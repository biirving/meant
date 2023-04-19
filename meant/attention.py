import torch
from torch import nn
from einops import rearrange
import math
import torch
from rotary_embedding_torch import RotaryEmbedding

"""
A classic attention mechanism with xPos embedding support.
"""
class attention(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, xPos:RotaryEmbedding, mask=False, droput=0.):
        super(attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.xPos = xPos
        self.mask =mask

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        
    def forward(self, input):
        # q, k, v matrices
        q_mat = rearrange(self.q(input), 'b l (h d) -> b l h d', h = self.num_heads)
        v_mat = rearrange(self.k(input), 'b l (h d) -> b l h d', h = self.num_heads)
        k_mat = rearrange(self.v(input), 'b l (h d) -> b l h d', h = self.num_heads)
        print(q_mat.shape)
        # apply positional embeddings before softmax function
        q_mat, k_mat = self.xPos.rotate_queries_and_keys(q_mat, k_mat)
        print(q_mat.shape)
        print(q_mat)
        print('k mat', k_mat.shape)
        # Compute attention scores using dot product of queries and keys
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / math.sqrt(self.Dh * self.num_heads)
        print('scores', scores.shape)

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