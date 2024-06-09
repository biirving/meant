import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.xPosAttention import xPosAttention
from src.meant.temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math
from transformers import AutoModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# okay, lets run these experiments
# because
MAX_SEQ_LENGTH = 3333

# how does this scale to deal with an arbitrary lag period
class temporalEncoder(nn.Module):
    def __init__(self, dim, num_heads, lag, use_rot_embed=False):
        super(temporalEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        if use_rot_embed:
            self.xPos = RotaryEmbedding(
                dim = 48,
                use_xpos = True,   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
                #xpos_scale_base=2
            )
        else:
            self.xPos = None
        # positional encoding


        #self.temp_cls_token = nn.Parameter(torch.randn)
        # Definitely matters how this is initialized
        #self.temp_embedding = nn.Parameter(torch.randn(1, lag, dim))
        self.temp_encode = nn.ModuleList([#nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim), 
                                            temporal(num_heads, dim), #, rot_embed=self.xPos), 
                                            #nn.LayerNorm(dim), 
                                            nn.Linear(dim, dim)])

        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, l, d = x.shape
        # the repeated temporal embedding is not good
        # how does this contribute to the batch being 
        # the same value for all of them?
        #temp_embed = repeat(self.temp_embedding, '1 l d -> b l d', b = b)
        #x += temp_embed
        for mod in self.temp_encode:           
            x = mod(x)
        return x


class meant_price(nn.Module):
    def __init__(self, price_dim, lag, num_classes, num_heads= 8, num_encoders = 1, channels=4):
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
        super(meant_price, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.lag = lag
        self.dim = price_dim
        self.num_heads = num_heads
        self.temporal_encoding = nn.ModuleList([temporalEncoder(price_dim, num_heads, lag)])

        # output head
        self.mlpHead = nn.ModuleList([nn.LayerNorm(price_dim), nn.Linear(price_dim, num_classes), nn.Sigmoid()])

    def forward(self,  **kwargs):
        prices = kwargs.get('prices')
        _batch = prices.shape[0]
        temporal_input = prices 
        for encoder in self.temporal_encoding:
            output = encoder.forward(temporal_input)
        for mod in self.mlpHead:
            output = mod(output)
        return output.squeeze(dim=1)
