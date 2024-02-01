import torch
from torch import nn
from einops import rearrange
import math
import torch
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

"""
A classic attention mechanism with xPos embedding support.
"""
class xPosAttention_flash(nn.Module):
    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, xPos:RotaryEmbedding, mask=True, droput=0.):
        super(xPosAttention_flash, self).__init__()

        # what is the dimension of the attention head
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.xPos = xPos
        self.mask = mask

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        self.q = nn.Linear(self.dim, self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads)
        #self.qkv = nn.Linear(self.dim, self.Dh * self.num_heads * 3)


    def forward(self, input):
        _batch = input.shape[0]
        q_mat, k_mat, v_mat = map(lambda t: rearrange(t, 'b l s (h d) -> (b l) s h d', h = self.num_heads), 
                                                        (self.q(input), self.v(input), self.k(input)))
        # Apply out xPos rotary embeddings
        q_mat, k_mat = self.xPos.rotate_queries_and_keys(q_mat, k_mat)
        scores = flash_attn_func(q_mat, k_mat, v_mat, dropout_p=0.0, softmax_scale=None, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        inter = rearrange(scores, '(b l) s h d -> b l s (h d)', b=_batch)
        output = self.multi_mad(inter)
        return output