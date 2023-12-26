import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from .attention import attention
from .xPosAttention import xPosAttention
from .temporal import temporal
from .rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# okay, lets run these experiments
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
# should we have temporal encoding baked into each encoder?
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
            use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
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

# how does this scale to deal with an arbitrary lag period
# lets make this multimodal temporal model, shall we?
class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n = 196

        # this is the positional embedding for the temporal encoder
        self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))
        self.lag = lag
        self.temp_encode = nn.ModuleList([#nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), 
                                            #nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])
        self.temp_encode = self.temp_encode.to(torch.float32)
        

    def forward(self, x):
        b, l, d = x.shape
        temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        x += temp_embed
        x = x.double()
        count = 0
        for mod in self.temp_encode:           
            x = mod(x)
            count+=1
        return x


class meant(nn.Module):
    def __init__(self, text_dim, image_dim, price_dim, height, width, patch_res, lag, num_classes, embedding, num_heads= 8, num_encoders = 1, channels=4):
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
        self.embedding = nn.ModuleList([embedding])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        #self.cls_token = nn.Parameter(torch.randn(1, lag, 1, image_dim))

        # the patch embedding for the image
        # we have to apply it to every image in the lag period
        # c = channel
        # h = height
        # w = width
        # b = batch
        # f = the number of frames we are processing
        self.patchEmbed = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim, dtype=torch.float32),)

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads) for i in range(num_encoders)])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads) for i in range(num_encoders)])

        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)]).to(torch.float32)

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

        # each component has a class token
        self.img_classtkn = nn.Parameter(torch.randn(1, lag, 1, image_dim))

        # how does this work with the lag period
        self.txt_classtkn = nn.Parameter(torch.randn(1, lag, 1, text_dim))

        # haven't decided on this dimensionality as of yet
        #self.temp_classtkn = nn.Parameter(torch.randn(1, image_dim))

    def forward(self, tweets, images, prices):
        _batch = images.shape[0]

        words = tweets
        embedded = []
        for word in words:
            for mod in self.embedding:
                word = mod(word)
            embedded.append(word)
        words = torch.concat(embedded, dim=0)
            

        words = rearrange(words, '(b l) s d -> b l s d', b = _batch)
        txt_classtkn = repeat(self.txt_classtkn, '1 l 1 d -> b l 1 d', b = _batch)
        words = torch.cat((txt_classtkn, words), dim = 2)
        for encoder in self.languageEncoders:
            words = encoder.forward(words)

        # the datatype correction depends on the system?
        image = self.patchEmbed(images.double())

        img_classtkn = repeat(self.img_classtkn, '1 l 1 d -> b l 1 d', b = _batch)
        image = torch.cat((img_classtkn, image), dim = 2)
        for encoder in self.visionEncoders:
            image = encoder.forward(image)

        # then we take the class tokens from both encoders
        temporal_input = torch.cat((words[:, :, 0, :], image[:, :, 0, :], prices), dim = 2)
        temporal = temporal_input.to(torch.float32)

        for encoder in self.temporal_encoding:
            temporal = encoder.forward(temporal)
        # we process temporal output
        for mod in self.mlpHead:
            temporal = mod(temporal)
        return temporal.squeeze(dim=1)        