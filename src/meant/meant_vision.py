import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.flash_attention import flash_attention
from src.meant.xPosAttention import xPosAttention
from src.meant.xPosAttention_flash import xPosAttention_flash
from src.meant.temporal import temporal
from src.meant.timesformer_pytorch import TimeSformer
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer
from src.utils.rms_norm import RMSNorm 

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
        for mod in self.temp_encode:           
            x = mod(x)
        return x

class meant_vision(nn.Module):
    def __init__(self, image_dim, price_dim, height, width, patch_res, lag, num_classes, flash=False, num_heads= 8, num_encoders = 1, channels=3):
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
        
        self.dim = price_dim 
        self.image_dim = image_dim
        self.output_dim = price_dim + image_dim
        self.num_heads = num_heads
        self.channels = channels
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))

        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))

        self.timesformer = TimeSformer(dim=image_dim, 
        image_size=224,
        patch_size=patch_res,
        num_frames=lag, 
        num_classes=num_classes,
        depth=1,
        heads=8,
        dim_head=64,
        attn_dropout = 0.1,
        ff_dropout = 0.1
        )

        # replace this with the TimeSFormer
        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=flash) for i in range(num_encoders)])

        self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)]).to(torch.float32)

        self.image_proj = nn.Sequential(nn.Linear(981, 1), nn.LayerNorm(1), nn.GELU())

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.image_dim), nn.Linear(self.image_dim, num_classes), nn.Sigmoid()])


        # haven't decided on this dimensionality as of yet
        #self.temp_classtkn = nn.Parameter(torch.randn(1, image_dim))

    def forward(self, **kwargs):
        images = kwargs.get('pixels')
        prices = kwargs.get('prices')

        _batch = images.shape[0]

        images = self.timesformer.meant_forward(images)
        images = rearrange(images, 'b p d -> b d p') 
        #images = rearrange(images, 'b l c h w -> (b l) c h w')
        #images = self.patchEmbed(images)
        #for encoder in self.visionEncoders:
         #   images = encoder.forward(images)
#        images = rearrange(images, '(b l) n d -> b l d n', b = _batch)
        images = self.image_proj(images)
        images = images.squeeze(dim=-1)

        #temporal = prices

        #temporal = torch.cat((temporal, prices), dim=2)
        #temporal = torch.mean(images, dim=2)

        #for encoder in self.temporal_encoding:
        #    temporal = encoder.forward(temporal)

        #itemporal = temporal.squeeze(dim=1)
        #temporal = torch.cat((temporal, images), dim=1)
        temporal = images

        for mod in self.mlpHead:
            temporal = mod(temporal)
        
        # need to try this
        return temporal

