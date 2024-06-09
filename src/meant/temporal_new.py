import torch
from torch import nn
from einops import rearrange
import math
from src.utils.torchUtils import weights_init

class temporal_2(nn.Module):
    def __init__(self, num_heads, dim, sequence_length=128, mask=False, droput=0., lag=5, rot_embed=None):
        super(temporal_2, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.Dh = int(self.dim / self.num_heads)
        self.dropout = nn.Dropout(droput)
        self.mask = mask

        self.softmax = nn.Softmax(dim=-1)
        self.multi_mad = nn.Linear(lag * self.num_heads * self.Dh, self.dim)

        self.sequence_length = sequence_length

        self.atten_size = self.Dh * self.num_heads

        self.q = nn.Linear(self.dim, self.atten_size)
        self.v = nn.Linear(self.dim, self.atten_size)
        self.k = nn.Linear(self.dim, self.atten_size)

        self.xPos = rot_embed
        self.apply(weights_init)

    def forward(self, input):
        b, l, _, _ = input.shape

        q_mat, v_mat, k_mat = map(lambda t: rearrange(t, 'b l s (h d) -> b l h s d', h=self.num_heads), 
                                  (self.q(input[:, l - 1, :, :]).view(b, 1, self.sequence_length, self.atten_size),
                                   self.v(input[:, l - 1, :, :]).view(b, 1, self.sequence_length, self.atten_size), 
                                   self.k(input)))

        
        if self.xPos is not None:
            q_mat, k_mat = self.xPos.rotate_queries_and_keys(q_mat, k_mat)

        #print(torch.matmul(q_mat, torch.transpose(k_mat, 3, 4)))
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 3, 4)) / math.sqrt(self.Dh)

        if self.mask:
            mask = torch.tril(torch.ones(scores.size(-1), scores.size(-1))).unsqueeze(0).unsqueeze(1).to(input.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Use numerically stable softmax
        scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
        scores = scores - scores_max
        weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        inter = torch.matmul(weights, v_mat)
        # Reshape for the linear layer
        inter = rearrange(inter, 'b l h s d -> b s (l h d)')
        output = self.multi_mad(inter)
        return output
