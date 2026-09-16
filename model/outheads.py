
"""

Output Heads

SPARC by DurdagiLab
# v2.2.9

"""


import torch 
import torch.nn as nn 



class ContactHead(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            nn.GELU(),
            nn.LayerNorm(c_hidden),
            nn.Linear(c_hidden, 1),
        )

    def forward(self, batch):
        batch['c_logits'] = self.head(batch['c_pair']).squeeze(-1)  # [B, N, N]

        return batch