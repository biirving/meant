import torch
from torch import nn
from einops import rearrange
import math
import torch

"""
A temporal attention mechanism.
"""

# attend to the antecent information over time?
class temporal(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, sequence_length=512, mask=False, droput=0., rot_embed=None):
        super(temporal, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        if self.Dh == 0:
            self.Dh = 1
        self.dropout = nn.Dropout(droput)
        self.mask = mask

        self.softmax = nn.Softmax(dim = -1)
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        self.atten_size = self.Dh * self.num_heads

        # We should add these here
        #self.lang_proj = nn.Sequential(nn.Linear(seq_len, 1), nn.LayerNorm(1), nn.GELU())
        #self.image_proj = nn.Sequential(nn.Linear(196, 1), nn.LayerNorm(1), nn.GELU())

        self.q = nn.Linear(self.dim, self.atten_size)
        self.v = nn.Linear(self.dim, self.atten_size)
        self.k = nn.Linear(self.dim, self.atten_size)
        self.xPos=rot_embed

    def forward(self, input, attention_mask=None):
        b, l, _ = input.shape

        # why is my entire batch returning the same output

        # Something here is not working. Figure out what it is? Perhaps?
        q_mat, k_mat, v_mat = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h = self.num_heads), 
                                                        (self.q(input[:, -1, :]).view(b, 1, self.atten_size), self.k(input), self.v(input)))


        if self.xPos is not None:
            q_mat, k_mat = self.xPos.rotate_queries_and_keys(q_mat, k_mat)
        
        # Compute attention scores using dot product of queries and keys
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / math.sqrt(self.Dh)

        @torch.jit.script_if_tracing
        def applyMask():
            if(self.mask):
                mask = torch.tril(torch.ones(scores.size(-1), scores.size(-1))).unsqueeze(0).unsqueeze(1).to(input.device)
                scores = scores.masked_fill(mask == 0, float('-inf'))
        applyMask()

        #if attention_mask is not None:
         #   print(attention_mask.shape)
        #    attention_mask = 1 - attention_mask.unsqueeze(dim=2)
         #   print(scores.shape)
          #  scores = scores + attention_mask * -1e9

        # Apply softmax to get attention 
        weights = torch.softmax(scores, dim=-1)

        inter = torch.matmul(weights, v_mat)

        inter = rearrange(inter, 'b h l d -> b (l h d)')
        output = self.multi_mad(inter)
        return output
