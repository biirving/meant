import sys, os
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from .attention import attention
from .flash_attention import flash_attention
from .xPosAttention import xPosAttention
from .xPosAttention_flash import xPosAttention_flash
from .temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer
from utils import RMSNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

# separate encoder module, because might make changes to structure
# should we have temporal encoding baked into each encoder?

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the CUDA device
    cuda_device_name = torch.cuda.get_device_name(0)

    # Check if the device name contains "Ampere" or a later architecture
    if "Ampere" in cuda_device_name or "A100" in cuda_device_name:
        ampere = True
    else:
        ampere = False
else:
    print("CUDA is not available on this system.")
    ampere = False

class languageEncoder(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, flash=False):
        """
        Encoder to extract language inputs. Virtually identical to visual encoder, except that it utitilizes 
        the xPos embeddings rather than the base rotary embeddings
        """
        super(languageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.xPos = RotaryEmbedding(
            dim = 48,
            use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            #xpos_scale_base=2
        )
        if flash and ampere:
            attention = xPosAttention_flash(num_heads, dim, self.xPos) 
        else:
            if flash and not ampere and torch.cuda.is_available():
                print(f"The {cuda_device_name} GPU is not from the Ampere series or later. Flash attention not supported.")
            elif flash:
                print('Cuda not supported. Cannot use flash attention.')
            attention = xPosAttention(num_heads, dim, self.xPos)
        self.encode = nn.ModuleList([RMSNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    attention, 
                                    RMSNorm(dim), 
                                    nn.Dropout(dropout),
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([RMSNorm(dim), nn.Linear(dim, dim), nn.GELU(), RMSNorm(dim), nn.Dropout(), nn.Linear(dim, dim)])

    def forward(self, input, attention_mask=None):
        inter = input
        for mod in self.encode:
            if type(mod).__name__ == 'xPosAttention':
                inter = mod(inter, attention_mask)
            else:
                inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        return inter + final_resi

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
        

    def forward(self, x):
        b, l, d = x.shape
        # the temporal embedding is the positional embedding?
        # use a rotary emebdding here too?
        temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


# the meant model without image inputs
class meant_tweet(nn.Module):
    def __init__(self, text_dim, price_dim, lag, num_classes, embedding, flash=False, num_heads=8, num_encoders = 1, channels=4):
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
        super(meant_tweet, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim
        self.num_heads = num_heads

        # pretrained language embedding from hugging face model
        # what if we have already used the flair embeddings
        self.embedding = nn.ModuleList([embedding])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=flash) for i in range(num_encoders)])
        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)])

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

        self.lag = lag

    def forward(self, tweets, attention_mask=None):
        _batch = tweets.shape[0]
        words = tweets.view(_batch * self.lag, tweets.shape[2])
        attention_mask = attention_mask.view(_batch * self.lag, attention_mask.shape[2])
        for mod in self.embedding:
            words = mod(words)
        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask=attention_mask)
        words = rearrange(words, '(b l) s d -> b l s d', b = _batch)
        # mean pooling works way better
        temporal = torch.mean(words, dim=2)

        for encoder in self.temporal_encoding:
            temporal = encoder.forward(temporal)

        # we process temporal output
        for mod in self.mlpHead:
            temporal = mod(temporal)

        # should we squeeze at the end?
        return temporal.squeeze(dim=1)        