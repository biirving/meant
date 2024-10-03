import torch
from torch import nn
from einops import rearrange
import math
import torch
from rotary_embedding_torch import RotaryEmbedding
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
"""
A classic attention mechanism with xPos embedding support.
"""
class flash_attention(nn.Module):

    # the default values in the original paper for num_heads and dim are 5 and 50 respectively
    def __init__(self, num_heads, dim, pos_emb:RotaryEmbedding, mask=False, droput=0.):
        super(flash_attention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim/self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.pos_emb = pos_emb
        self.mask = mask

        self.softmax = nn.Softmax(dim = -1)
        self.multi_mad = nn.Linear(self.num_heads * self.Dh, self.dim)

        self.q = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.v = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        self.k = nn.Linear(self.dim, self.Dh * self.num_heads).float()
        
    def forward(self, input):
        # should change to flash attention 
        q_mat, k_mat, v_mat = map(lambda t: rearrange(t, 'b s (h d) -> b s h d', h = self.num_heads), 
                                                        (self.q(input), self.v(input), self.k(input)))
        # apply rotary embeddings
        q_mat = self.pos_emb.rotate_queries_or_keys(q_mat)
        k_mat = self.pos_emb.rotate_queries_or_keys(k_mat)
        scores = flash_attn_func(q_mat.half(), k_mat.half(), v_mat.half(), dropout_p=0.0, softmax_scale=1/self.dim, causal=self.mask,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        inter = rearrange(scores, 'b s h d -> b s (h d)')
        output = self.multi_mad(inter)
        return output
