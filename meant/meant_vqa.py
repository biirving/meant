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

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

# should the vision encoder encode temporal information?
class visionEncoder(nn.Module):
    # we should pretrain the patch embeddings, right?
    def __init__(self, dim, num_heads, flash=False):
        """
        The initial encoder for extracting relevant features from the multimodal input.
        """
        super(visionEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
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

# separate encoder module, because might make changes to structure
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
        return inter + final_resid

# how does this scale to deal with an arbitrary lag period
# lets make this multimodal temporal model, shall we?
class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # positional encoding
        self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))
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


class meant_vqa(nn.Module):
    # how to make the height and width dynamic?
    # need to solve this to make collate function work
    # and to understand vits?
    def __init__(self, text_dim, image_dim, price_dim, height, width, patch_res, lag, num_classes, embedding, flash=False, num_heads= 8, num_encoders = 1, channels=4):
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
        super(meant_vqa, self).__init__()

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

        # pretrained language embedding from hugging face model
        # what if we have already used the flair embeddings
        self.embedding = nn.ModuleList([embedding])
        #self.embedding_alt = nn.Linear(1, 768)

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
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim))

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads, flash=flash) for i in range(num_encoders)])
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads, flash=flash) for i in range(num_encoders)])

        #self.temporal_encoding = nn.ModuleList([temporalEncoder(self.dim, num_heads, lag)])
        self.multimodal_embedding = nn.Sequential(nn.Linear(1, self.dim), nn.GELU(), RMSNorm(self.dim), nn.Linear(self.dim, self.dim))
        self.multimodal_encoding = nn.ModuleList([visionEncoder(self.dim, num_heads, flash=flash)])

        # output head
        self.mlpHead = nn.ModuleList([RMSNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

    def forward(self, tweets, images, attention_mask=None):
        _batch = images.shape[0]
        words = tweets
        for mod in self.embedding:
            words = mod(words)

        # perhaps change these back?
        for encoder in self.languageEncoders:
            words = encoder.forward(words, attention_mask)

        images = self.patchEmbed(images)
        for encoder in self.visionEncoders:
            images = encoder.forward(images)

        # how can we translate our temporal attention mechanism

        # this is not gonna cut it
        # there should be some sort of cross attention from the inputs
        temporal = torch.cat((torch.mean(words, dim=1), torch.mean(images, dim=1)), dim = 1)

        #temporal = temporal.unsqueeze(dim=2)
        #temporal = self.multimodal_embedding(temporal)

#        for encoder in self.multimodal_encoding:
  #          temporal = encoder.forward(temporal)

        for mod in self.mlpHead:
            temporal = mod(temporal)

        return temporal        