import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os 

class dividedSpaceTimeAttention(nn.Module):
    def __init__(self, num_heads, dim, n, num_frames, self, num_heads, pos_emb:RotaryEmbedding, mask=False, droput=0.):
        super(dividedSpaceTimeAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n = int(n)
        self.Dh = int(self.dim/self.num_heads)
        self.num_frames = num_frames
        self.pos_emb = pos_emb
        self.mask = mask
        # softmax within each head
        self.softmax = nn.Softmax(dim = -1)
        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.multimad_temp = nn.Linear(self.num_heads * 3 * self.Dh, self.dim) 
        self.multi_mad_final = nn.Linear(self.num_heads * 3 * self.Dh, self.dim)


    
    def forward(self, input):
        # q, k, v matrices
        q_mat = rearrange(self.q(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
        k_mat = rearrange(self.v(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
        v_mat = rearrange(self.k(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)

        # will this work with the increased number of dimensions
        q_mat = self.pos_emb.rotate_queries_or_keys(q_mat)
        k_mat = self.pos_emb.rotate_queries_or_keys(k_mat)

        # the class token has to attend to the entire input (don't need multiple heads)
        # So the q matrix will be the query of the first row (the class token) which will
        # 'attend' to all of the key values
        q_mat_cls_tkn = rearrange(self.q(input[:, 0:1, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
        v_mat_cls_tkn = rearrange(self.v(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
        k_mat_cls_tkn = rearrange(self.k(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
        inter_cls_tkn = self.softmax(torch.matmul(q_mat_cls_tkn, torch.transpose(k_mat_cls_tkn, 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
        inter_cls_tkn = torch.matmul(inter_cls_tkn, v_mat_cls_tkn)

        # First, we calculate temporal attention, multiplying the q vector being processed by every k vector at that 
        # frame in subsequent time steps
        # at this point, the q matrix contains all of the query vectors to be processed
        # but each row will be multiplied by a different set of the k vectors, because they are being compared to the other
        # keys at that timeframe
        temporal = self.softmax(torch.matmul(q_mat[:, :, 1::self.n, :], torch.transpose(k_mat[:, :, 1::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
        temporal = torch.matmul(temporal, v_mat[:, :, 1::self.n, :])
        temporal = torch.sum(temporal, 2, keepdim = True)
        
        # temporal calculation
        for x in range(2, self.n):
            # get all of the patches at that frame
            inter = self.softmax(torch.matmul(q_mat[:, :, x::self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
            inter = torch.sum(inter, 2, keepdim = True)
            #inter = inter.repeat(1, 1, self.num_frames, 1)
            temporal = torch.cat((temporal, inter), 2)
        
        temporal = temporal.repeat(1, 1, self.num_frames, 1)
        temporal_input = self.multimad_temp(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))

        q_mat = rearrange(self.q(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
        v_mat = rearrange(self.k(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
        k_mat = rearrange(self.v(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)

        temporal = self.softmax(torch.matmul(q_mat[:, :, 0:self.n, :], torch.transpose(k_mat[:, :, 0:self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
        temporal = torch.matmul(temporal, v_mat[:, :, 0:self.n, :])
        temporal = torch.sum(temporal, 2, keepdim = True)

        # spatial calculation
        for x in range(1, self.num_frames):
            inter = self.softmax(torch.matmul(q_mat[:, :, x:x+self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
            inter = torch.sum(inter, 2, keepdim = True)
            temporal = torch.cat((temporal, inter), 2)

        temporal = temporal.repeat(1, 1, self.n, 1)
        temporal = torch.cat((inter_cls_tkn, temporal), 2)
        output = self.multi_mad_final(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))
        return output
