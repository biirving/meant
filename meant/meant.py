import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from attention import attention
from xPosAttention import xPosAttention
from temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer

# because
MAX_SEQ_LENGTH = 3333

# should the vision encoder encode temporal information?
class visionEncoder(nn.Module):
    def __init__(self, dim, num_heads):
        """
        The initial encoder for extracting relevant features from the multimodal input.
        """
        super(visionEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # so, the xPos embeddings will focus on the pixel case
        self.posEmbed = RotaryEmbedding(
        dim = math.floor(dim/num_heads/2),
        freqs_for='pixel')
        self.encode = nn.ModuleList([nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    attention(num_heads, dim, self.posEmbed), 
                                    nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim)])

    def forward(self, input):
        inter = input
        for mod in self.encode:
            inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        # then another residual connection before the output is processed
        return inter + final_resid

# separate encoder module, because might make changes to structure
class languageEncoder(nn.Module):
    def __init__(self, dim, num_heads):
        """
        Encoder to extract language inputs. Virtually identical to visual encoder, except that it utitilizes 
        the xPos embeddings rather than the base rotary embeddings
        """
        super(languageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # so, the xPos embeddings will focus on the pixel case
        self.xPos = RotaryEmbedding(
            dim = 48,
            #use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            #xpos_scale_base=2
        )

        self.encode = nn.ModuleList([nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    xPosAttention(num_heads, dim, self.xPos), 
                                    nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim)])

    def forward(self, input):
        inter = input
        for mod in self.encode:
            inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        return inter + final_resid



class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n = 196

        self.temp_embeding = torch.randn(1, lag, dim)
        self.lag = lag
        self.temp_encode = nn.ModuleList([nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), 
                                            nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])
        #self.clsstoken = nn.Parameter()
                    
    def forward(self, input):
        b, l, d = input.shape
        print(self.temp_embeding.shape)
        input += self.temp_embeding
        for encode in self.temp_encode:
            input = encode(input)
        return input


class meant(nn.Module):
    def __init__(self, text_dim, image_dim, price_dim, height, width, patch_res, lag, num_classes, embedding, num_heads= 8, num_encoders = 1, channels=3):
        """
        Args
            dim: The dimension of the input to the encoder
            num_heads: The number of attention heads for the attention mechanism
            height: The height of the images being processed
            width: The width of the images being processed
            patch_res: The dimension of each image patch
            channels: The number of channels in the images being processed
            num_classes: The number of classes for the mlp output head
        
        returns: A classification vector, of size num_classes
        """
        super(meant, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim + image_dim + price_dim
        self.num_heads = num_heads

        # for the image component of the encoder
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))

        # pretrained language embedding from hugging face model
        self.embedding = nn.ModuleList([embedding, nn.Linear(1024, 768)])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        #self.cls_token = nn.Parameter(torch.randn(1, lag, 1, image_dim))

        # the patch embedding for the image
        # we have to apply it to every image in the lag period
        # c = channel
        # h = height
        # w = width
        # b = batch
        # l = lag period (how many images are being processed for each input)
        self.patchEmbed = nn.Sequential(
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim),)

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads) for i in range(num_encoders)])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads) for i in range(num_encoders)])

        self.temporal_encoding = temporalEncoder(150528, num_heads, lag)
        self.mlpHead = nn.ModuleList([nn.LayerNorm(image_dim), nn.Linear(image_dim, num_classes)])

        # image projection


        # text projection

        # we have a classtoken for the temporal input
        self.classtkn = nn.Parameter(torch.randn(1, image_dim))

    def forward(self, tweets, images, prices):

        _batch = images.shape[0]

        # embed multiple days of information with a forloop
        words = tweets
        for mod in self.embedding:
            words = mod(words)

        words = rearrange(words, '(b l) s d -> b l s d', b = _batch)
        for encoder in self.languageEncoders:
            words = encoder.forward(words)

        image = self.patchEmbed(images)
        for encoder in self.visionEncoders:
            image = encoder.forward(image)

        # flatten the word and image inputs for concatenation
        words = torch.flatten(words, start_dim = 2, end_dim = -1)
        image = torch.flatten(image, start_dim = 2, end_dim = -1)

        # so how are we going to deal with batch size
        # lag period, sequence length, sequence dim
        print('words', words.shape)
        print('image', image.shape)

        # lets bring this temporal attention home
        # maybe these should be projected into a more managable state before being fed to an attention architecture
        temporal_input = torch.cat((words, image), dim = 2)

        # where concatenation happens?
        output = self.temporal_encoding(image)
        for mod in self.mlpHead:
            output = mod(output)
        return output



