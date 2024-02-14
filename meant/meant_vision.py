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
MAX_SEQ_LENGTH = 3333

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class visionEncoder(nn.Module):
    # we should pretrain the patch embeddings, right?
    def __init__(self, dim, num_heads, flash=False):
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
        
        if flash and ampere:
            atten = flash_attention(num_heads, dim, self.posEmbed) 
        else:
            if flash and not ampere and torch.cuda.is_available():
                print(f"The {cuda_device_name} GPU is not from the Ampere series or later. Flash attention not supported.")
            elif flash:
                print('Cuda not supported. Cannot use flash attention.')
            atten = attention(num_heads, dim, self.posEmbed)

        self.encode = nn.ModuleList([RMSNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    atten, 
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
        temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        x += temp_embed
        count = 0
        for mod in self.temp_encode:           
            x = mod(x)
            count+=1
        return x

class meant_vision(nn.Module):
    def __init__(self, image_dim, price_dim, height, width, patch_res, lag, num_classes, flash=False, num_heads= 8, num_encoders = 1, channels=4):
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
        super(meant_vision, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = image_dim
        self.num_heads = num_heads

        # for the image component of the encoder
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))

        # the patch embedding for the image
        # we have to apply it to every image in the lag period
        # c = channel
        # h = height
        # w = width
        # b = batch
        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=flash) for i in range(num_encoders)])

        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)]).to(torch.float32)

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

        # haven't decided on this dimensionality as of yet
        #self.temp_classtkn = nn.Parameter(torch.randn(1, image_dim))

    def forward(self, images):

        _batch = images.shape[0]
        images = rearrange(images, 'b l c h w -> (b l) c h w')
        images = self.patchEmbed(images)
        for encoder in self.visionEncoders:
            images = encoder.forward(images)
        images = rearrange(images, '(b l) n d -> b l n d', b = _batch)
        temporal = torch.mean(images, dim=2)
        for encoder in self.temporal_encoding:
            temporal = encoder.forward(temporal)
        for mod in self.mlpHead:
            temporal = mod(temporal)
        return temporal.squeeze(dim=1)        
