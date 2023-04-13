import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from attention import attention
from temporal import temporal
from rotary_embedding_torch import RotaryEmbedding

# because
MAX_SEQ_LENGTH = 3333

class encoder(nn.Module):
    def __init__(self, dim, num_heads):
        """
        The initial encoder for extracting relevant features from the multimodal input.
        """
        super(encoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.xPos = RotaryEmbedding(dim)
        self.encode = nn.ModuleList([nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim), 
                                    attention(num_heads, dim, self.xPos), 
                                    nn.LayerNorm(dim), 
                                    nn.Linear(dim, dim)])
        self.encode2 = nn.ModuleList([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim)])
    
    def forward(self, input):
        inter = self.encode(input)
        output = self.encode2(inter + input)
        return output


class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temp_embeding = torch.randn(1, lag, dim)
        self.lag = lag
        self.temp_encode = nn.ModuleList([nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), 
                                            nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])
    def forward(self, input):
        inter = self.temp_embeding.view(input.shape[0], self.lag, self.dim) + input
        return self.temp_encode(inter)


class meant(nn.Module):
    def __init__(self, text_dim, image_dim, price_dim, num_heads, height, width, patch_res, lag, num_classes, num_encoders = 5, channels=3):
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
            Rearrange('b l c (h p1) (w p2) -> b l (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, image_dim),)
        
        self.encoders = nn.ModuleList([encoder(self.dim, num_heads)] * num_encoders)
        self.temporal_encoding = temporalEncoder(self.dim, num_heads, lag)

        # temporal dim?
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes)])

    def forward(self, tweets, images, prices):
        image_embed = self.patchEmbed(images)
        input = torch.cat((tweets, image_embed, prices), dim = 2)
        initial_encode = self.encoders(input)
        temporal_encode = self.temporal_encoding(initial_encode)
        output = self.mlpHead(temporal_encode)
        return output