import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from .attention import attention
from .xPosAttention import xPosAttention
from .xPosAttention_flash import xPosAttention_flash
from .temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer
from utils import RMSNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_SEQ_LENGTH = 3333

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
        
        # why use the kosmos architecture
        self.encode = nn.ModuleList([RMSNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    attention(num_heads, dim, self.posEmbed), 
                                    RMSNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([RMSNorm(dim), nn.Linear(dim, dim), nn.GELU(), RMSNorm(dim), nn.Linear(dim, dim)])

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

class languageEncoder_v2(nn.Module):
    def __init__(self, dim, num_heads, embeddings=None):
        """
        Encoder to extract language inputs. Virtually identical to visual encoder, except that it utitilizes 
        the xPos embeddings rather than the base rotary embeddings
        """
        super(languageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # should we remove the positional encoding?
        self.embeddings = embeddings

        # so, the xPos embeddings will focus on the pixel case
        self.xPos = RotaryEmbedding(
            dim = 48,
            use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            #xpos_scale_base=2
        )

        self.encode = nn.ModuleList([RMSNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    xPosAttention_flash(num_heads, dim, self.xPos), 
                                    RMSNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([RMSNorm(dim), nn.Linear(dim, dim), nn.GELU(), RMSNorm(dim), nn.Linear(dim, dim)])

    def forward(self, words):
        # if you put the embeddings in here, you can't stack the encoders
        # the problem is that we stack embeddings?
        if embeddings is not None:
            for mod in self.embeddings:
                words = mod(words)
            words = rearrange(words, '(b l) s d -> b l s d', b = _batch)
        else:
            inter = words 
        for mod in self.encode:
            inter = mod(inter)
        inter = inter + words
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

        # this is the positional embedding for the temporal encoder
        # does this positional encoding need to be repeated?
        self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))


        self.lag = lag

        self.temp_encode = nn.ModuleList([RMSNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), 
                                            RMSNorm(dim), 
                                            nn.Linear(dim, dim)])
        

    def forward(self, x):
        b, l, d = x.shape
        # the repeated temporal embedding is not good
        temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


class meant_v2(nn.Module):
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
        super(meant_v2, self).__init__()


        # recent additions for editing purposes
        self.lag = lag
        self.text_dim = text_dim
        self.image_dim = image_dim

        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim + image_dim 
        self.num_heads = num_heads

        # for the image component of the encoder
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))
        self.embedding = nn.ModuleList([embedding])
        #self.embedding_alt = nn.Linear(1, 768)
        #self.cls_token = nn.Parameter(torch.randn(1, lag, 1, image_dim))
        self.patchEmbed = nn.Sequential(
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))
        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads) for i in range(num_encoders)])

        self.languageEncoders = nn.ModuleList([
            languageEncoder(text_dim, num_heads, embeddings=embedding) if i == 0 else languageEncoder(text_dim, num_heads)
            for i in range(num_encoders)
        ])
        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)])
        self.mlpHead = nn.ModuleList([RMSNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])
        self.img_classtkn = nn.Parameter(torch.randn(1, lag, 1, image_dim))
        self.txt_classtkn = nn.Parameter(torch.randn(1, lag, 1, text_dim))

        #self.temp_classtkn = nn.Parameter(torch.randn(1, image_dim))

    def forward(self, tweets, images):
        _batch = images.shape[0]
        words = tweets.view(_batch * self.lag, tweets.shape[2])
        for encoder in self.languageEncoders:
            words = encoder.forward(words)
        
        image = self.patchEmbed(images)

        for encoder in self.visionEncoders:
            image = encoder.forward(image)

        temporal = torch.cat((torch.mean(words, dim=2), torch.mean(image, dim=2)), dim = 2)

        for encoder in self.temporal_encoding:
            temporal = encoder.forward(temporal)

        if torch.isnan(temporal).any():
            raise ValueError('Nans encountered in the temporal encoder')        
            sys.exit()

        for mod in self.mlpHead:
            temporal = mod(temporal)
        return temporal.squeeze(dim=1)        