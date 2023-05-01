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
        
    # batch, lag, vector
    def forward(self, input):
        # q, k, v matrices
        # the queries should attend to one another.
        # the key difference with this mechanism is that the attention component focuses primarily on the target day.
        # so we should generate queries for the target day only
        print(input.shape)
        q_mat, k_mat, v_mat = map(lambda t: rearrange(t, 'b l n (h d) -> b l h n d', h = self.num_heads), 
                                                        (self.q(input), self.v(input), self.k(input)))

        # Compute attention scores using dot product of queries and keys
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 3, 4)) / math.sqrt(self.Dh * self.num_heads)

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
        inter = rearrange(inter, 'b l h n d -> b l n (h d)')
        output = self.multi_mad(inter)
        return output