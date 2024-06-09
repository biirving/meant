import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.xPosAttention import xPosAttention
from src.meant.temporal import temporal
from src.utils.rms_norm import RMSNorm
from src.utils.torchUtils import weights_init
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

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
    def __init__(self, dim, num_heads, dropout=0.4, flash=False):
        """
        Encoder to extract language inputs. Virtually identical to visual encoder, except that it utilizes 
        the xPos embeddings rather than the base rotary embeddings
        """
        super(languageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.xPos = RotaryEmbedding(
            dim=48,
            use_xpos=True,  # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
            #xpos_scale_base=2
        )
        if flash and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            attention = xPosAttention_flash(num_heads, dim, self.xPos) 
        else:
            if flash and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
                print(f"The {torch.cuda.get_device_name(0)} GPU is not from the Ampere series or later. Flash attention not supported.")
            elif flash:
                print('Cuda not supported. Cannot use flash attention.')
            attention = xPosAttention(num_heads, dim, self.xPos)
        
        self.encode = nn.ModuleList([
            RMSNorm(dim), 
            nn.Linear(dim, dim), 
            attention, 
            RMSNorm(dim), 
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        ])
        
        self.encode2 = nn.ModuleList([
            RMSNorm(dim), 
            nn.Linear(dim, dim), 
            nn.GELU(), 
            RMSNorm(dim), 
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        ])

    def forward(self, input, attention_mask=None):
        inter = input
        for mod in self.encode:
            if isinstance(mod, xPosAttention):
                inter = mod(inter, attention_mask)
            else:
                inter = mod(inter)
        inter = inter + input
        final_resid = inter
        for mod in self.encode2:
            inter = mod(inter)
        return inter + final_resid


class temporalEncoderNew(nn.Module):
    def __init__(self, dim, num_heads, lag, s, dropout=0.0):
        super(temporalEncoderNew, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n = 196
        self.s = s

        # this is the positional embedding for the temporal encoder
        self.temp_embedding = nn.Parameter(torch.randn(1, lag, s, dim))
        self.lag = lag
        self.temp_encode = nn.ModuleList([
            RMSNorm(dim),
            nn.Linear(dim, dim).float(), 
            temporal(num_heads, dim).float(), 
            RMSNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim).float()
        ])

    def forward(self, x):
        b, l, s, d = x.shape
        temp_embed = repeat(self.temp_embedding, '1 l s d -> b l s d', b=b)
        x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag, dropout=0.0):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # positional encoding
        self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))
        self.temp_encode = nn.ModuleList([
            RMSNorm(dim), 
            nn.Linear(dim, dim), 
            temporal(num_heads, dim), 
            RMSNorm(dim), 
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        ])

    def forward(self, x):
        b, l, d = x.shape
        temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b=b)
        x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


class meantTweetPrice(nn.Module):
    def __init__(self, text_dim, price_dim, lag, num_classes, embedding, 
    sequence_length=128, flash=False, num_heads=8, num_encoders=1, 
    num_temporal_encoders=1, channels=4, pool='mean'):
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
        super(meantTweetPrice, self).__init__()

        self.lag=lag
        self.price_dim=price_dim
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim + price_dim
        self.num_heads = num_heads

        # pretrained language embedding from hugging face model
        self.embedding = nn.ModuleList([embedding])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        #self.cls_token = nn.Parameter(torch.randn(1, lag, 1, image_dim))

        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=flash) for i in range(num_encoders)])

        # why is this fucked up
        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag) for i in range(num_temporal_encoders)])

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

        # how does this work with the lag period
        self.txt_classtkn = nn.Parameter(torch.randn(1, lag, 1, text_dim))

        # haven't decided on this dimensionality as of yet
        #self.temp_classtkn = nn.Parameter(torch.randn(1, image_dim))
        #self.apply(weights_init)

        self.pool = pool


    def forward(self, tweets, prices, attention_mask=None):
        _batch = tweets.shape[0]
        words = tweets.view(_batch * self.lag, tweets.shape[2])

        if attention_mask is not None:
            attention_mask = attention_mask.view(_batch * self.lag, attention_mask.shape[2])

        for mod in self.embedding:
            words = mod(words)

        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask=attention_mask)
        words = rearrange(words, '(b l) s d -> b l s d', b = _batch)

        # mean pooling works way better
        #if self.pool == 'mean':
        #    temporal = torch.mean(words, dim=2)
        #else:
        #    temporal = words[:, :, 0, :]
        #temporal = words

        #temporal_input = torch.cat((temporal, prices.view(_batch, self.lag, 1, self.price_dim).expand(_batch, self.lag, temporal.shape[2], self.price_dim)), dim = 3)
        #temporal_input = temporal_input.to(torch.float32)

        temporal_input = torch.cat((torch.mean(words, dim=2), prices), dim=2)

        for encoder in self.temporal_encoding:
            output = encoder.forward(temporal_input)

        # Take the mean over the sequence? Or the tokens? 
        #output = torch.mean(output, dim=1)

        for mod in self.mlpHead:
            output = mod(output)

        return output.squeeze(dim=1)
