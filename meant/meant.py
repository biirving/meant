import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from attention import attention
from temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
# because
MAX_SEQ_LENGTH = 3333

# two separate encoders? or one encoder?
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
        dim = math.floor(dim/num_heads/2))
        #self.xPos = RotaryEmbedding(math.floor(dim/num_heads/2), use_xpos = True)
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


class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n = 196
        self.temp_embeding = torch.randn(1, lag, self.n, dim)
        self.lag = lag
        self.temp_encode = nn.ModuleList([nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), 
                                            nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])
    def forward(self, input):
        b, l, n, _ = input.shape
        print(self.temp_embeding[:, :(n + 1)].shape)
        input += self.temp_embeding[:, :(n + 1)]
        for encode in self.temp_encode:
            input = encode(input)
        return input

# WE PROCESS THE TWO MODALITIES WITH DIFFERENT ENCODERS
# first strategy: work with vision and language separately. I need a clear path forwards.
class meant(nn.Module):
    def __init__(self, text_dim, image_dim, price_dim, height, width, patch_res, lag, num_classes, num_heads= 8, num_encoders = 1, channels=3):
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

        # the patch embedding for the image
        # we have to apply it to every image in the lag period
        # c = channel
        # h = height
        # w = width
        # b = batch
        # l = lag period (how many images are being processed for each input)
        self.patchEmbed = nn.Sequential(
            # not using the patch dimenions? instead creating one huge vector
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim),)

        self.visionEncoders = nn.ModuleList([visionEncoder(image_dim, num_heads) for i in range(num_encoders)])
        self.temporal_encoding = temporalEncoder(image_dim, num_heads, lag)
        self.mlpHead = nn.ModuleList([nn.LayerNorm(image_dim), nn.Linear(image_dim, num_classes)])

    def forward(self, tweets, images, prices):
        image_embed = self.patchEmbed(images)
        #input = torch.cat((tweets, image_embed, prices), dim = 3)
        input = image_embed
        for encoder in self.visionEncoders:
            input = encoder.forward(input)
        output = self.temporal_encoding(input)
        for mod in self.mlpHead:
            output = mod(output[:, 0, :])
        return output