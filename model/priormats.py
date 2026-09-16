
""" 

Matrix Generation Block

SPARC by DurdagiLab
# v2.2.9

"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class OuterProduct(nn.Module):
    def __init__(self, d_model, c_pair, mode='inter'):
        super().__init__()
        
        self.d_model, self.c_pair, self.mode = d_model, c_pair, mode
        
        self.pair_out = nn.Linear(d_model * d_model, c_pair)

    def _opm_efficient(self, a, b):
        W = self.pair_out.weight.view(self.c_pair, self.d_model, self.d_model)
        aW = torch.einsum("bic,pcd->bipd", a, W)
        return torch.einsum("bipd,bjd->bijp", aW, b) + self.pair_out.bias

    def forward(self, batch):

        key = {'inter': ('a_interin', 'b_interin', 'c_inter'),
               'intra': ('a_intrain', 'b_intrain', 'c_intra')}[self.mode]
        batch[key[2]] = self._opm_efficient(batch[key[0]], batch[key[1]])

        return batch