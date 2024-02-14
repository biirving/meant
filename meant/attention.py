import torch
from torch import nn
from einops import rearrange
import math
import torch
from rotary_embedding_torch import RotaryEmbedding

"""
A classic attention mechanism with support for axial-2D embeddings (images)
"""
class attention(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, pos_emb:RotaryEmbedding, mask=False, droput=0.):
        super(attention, self).__init__()

        # what is the dimension of the attention head
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.pos_emb = pos_emb
        self.mask = mask

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        
    def forward(self, input):
        q_mat, k_mat, v_mat = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), 
                                                        (self.q(input), self.v(input), self.k(input)))
 
        q_mat = self.pos_emb.rotate_queries_or_keys(q_mat)
        k_mat = self.pos_emb.rotate_queries_or_keys(k_mat)

        # Compute attention scores using dot product of queries and keys
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / math.sqrt(self.Dh * self.num_heads)

        # for tracing: trace call cannot deal with control flow
        @torch.jit.script_if_tracing
        def applyMask():
            if(self.mask):
                # Create a mask matrix to prevent attending to future positions
                mask = torch.tril(torch.ones(scores.size(-1), scores.size(-1))).unsqueeze(0).unsqueeze(1).to(input.device)
                scores = scores.masked_fill(mask == 0, float('-inf'))
        applyMask()

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)
        # Apply attention weights to values
        inter = torch.matmul(weights, v_mat)
        # reshape for the linear layer
        inter = rearrange(inter, 'b h n d -> b n (h d)')
        output = self.multi_mad(inter)
        output = self.dropout(output)
        return output
