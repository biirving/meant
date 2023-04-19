import torch
from torch import nn
from einops import rearrange
import math
import torch

"""
A temporal attention mechanism.
"""
class temporal(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, mask=False, droput=0.):
        super(temporal, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.mask =mask

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        
    # the input will be in the shape of: 
    # batch, lag, vector
    def forward(self, input):
        # q, k, v matrices
        # we want to use the query vector for the target day only, and attend to the entire input from this vector
        q_mat = rearrange(self.q(input[:, input.shape[1] - 1]), 'b (h d) -> b h d', h = self.num_heads)
        v_mat = rearrange(self.k(input), 'b l (h d) -> b l h d', h = self.num_heads)
        k_mat = rearrange(self.v(input), 'b l (h d) -> b l h d', h = self.num_heads)
        
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
        inter = rearrange(inter, 'b l h d -> b l (h d)')
        output = self.multi_mad(inter)
        return output