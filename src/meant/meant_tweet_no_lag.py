import sys, os
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from src.meant.attention import attention
from src.meant.xPosAttention import xPosAttention
from src.meant.temporal import temporal
from rotary_embedding_torch import RotaryEmbedding
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LENGTH = 3333



# For debugging:
# Lets get the model to return something other than all 0s

class languageEncoder(nn.Module):
    def __init__(self, dim, num_heads):
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


# the meant model without image inputs
class meant_tweet_no_lag(nn.Module):
    def __init__(self, text_dim, embedding, num_classes=2, num_heads=8, num_encoders = 1, channels=4):
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
        super(meant_tweet_no_lag, self).__init__()
        
        # concatenation strategy: A simple concatenation to feed the multimodal information into the encoder.
        self.dim = text_dim
        self.num_heads = num_heads

        # pretrained language embedding from hugging face model
        # what if we have already used the flair embeddings
        self.embedding = nn.ModuleList([embedding])

        # classification token for the image component. Will be passed to the temporal attention mechanism
        self.languageEncoders = nn.ModuleList([languageEncoder(text_dim, num_heads) for i in range(num_encoders)])

        # output head
        # using the sigmoid!
        self.mlpHead = nn.ModuleList([nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes), nn.Sigmoid()])

        # how does this work with the lag period
        self.txt_classtkn = nn.Parameter(torch.randn(1, 1, text_dim))

    def forward(self, **kwargs):
        tweets = kwargs.get('input_ids')
        _batch = tweets.shape[0]

        words = tweets
        for mod in self.embedding:
            words = mod(words)

        #txt_classtkn = repeat(self.txt_classtkn, '1 1 d -> b 1 d', b = _batch)
        #words = torch.cat((txt_classtkn, words), dim = 1)
        for encoder in self.languageEncoders:
            words = encoder.forward(words)

        # the temporal input is just  the tweet
        #temporal = words[:, 0, :].view(_batch, 768)
        temporal = words.mean(dim=1).view(_batch, 768)

        for mod in self.mlpHead:
            temporal = mod(temporal)

        # should we squeeze at the end?
        return temporal.squeeze(dim=1)        