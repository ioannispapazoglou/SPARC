
"""

Input Embedding Block

SPARC by DurdagiLab
# v2.2.9

"""


import torch
import torch.nn as nn



class InterSignalEmbeder(nn.Module):
    def __init__(self, in_dim, d_model=128):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, d_model)
        
        self.chainemb = nn.Embedding(2, d_model)   # 0 = A, 1 = B
        nn.init.zeros_(self.chainemb.weight)

    def forward(self, batch):
        a = self.norm(batch['a_inter'].float())
        b = self.norm(batch['b_inter'].float())

        a = self.linear(a) + self.chainemb.weight[0]
        b = self.linear(b) + self.chainemb.weight[1]
        
        batch['a_interin'] = a * batch['a_seqmsk'].unsqueeze(-1).float()
        batch['b_interin'] = b * batch['b_seqmsk'].unsqueeze(-1).float()

        return batch



class IntraSignalEmbeder(nn.Module):
    def __init__(self, in_dim, d_model=128):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, d_model)

    def forward(self, batch):
        a = self.norm(batch['a_intra'].float())
        b = self.norm(batch['b_intra'].float())

        a = self.linear(a)
        b = self.linear(b)
        
        batch['a_intrain'] = a * batch['a_seqmsk'].unsqueeze(-1).float()
        batch['b_intrain'] = b * batch['b_seqmsk'].unsqueeze(-1).float()

        return batch