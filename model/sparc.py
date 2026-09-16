
"""

Model Assembly

SPARC by DurdagiLab
# v2.2.9

"""


import torch 
import torch.nn as nn

from utils.batch_utils import batch_on_the_fly, batch_to_device

from model.embeddings import IntraSignalEmbeder, InterSignalEmbeder
from model.priormats import OuterProduct
from model.triangles import TriangleAttBlock
from model.outheads import ContactHead



class SPARC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.intra_n_bins=config['intra_n_bins']
        self.intra_max_dist=config['intra_max_dist']
        self.th_dst=config['con_threshold']
        
        # embedders
        self.inter2emb = InterSignalEmbeder(in_dim=config['d_mint'], d_model=config['d_opm'])
        self.intra2emb = IntraSignalEmbeder(in_dim=config['d_esm3'], d_model=config['d_opm'])

        self.inter2mat = OuterProduct(d_model=config['d_opm'], c_pair=config['c_pair'], mode='inter')
        self.intra2mat = OuterProduct(d_model=config['d_opm'], c_pair=config['c_pair'], mode='intra')

        #self.types2mat = nn.Embedding(2, config['c_pair']) # 2 for homo or hetero

        # triagles
        self.intraemb = nn.Linear(config['intra_n_bins'], config['h_trimul'])
        self.intrabias = nn.Linear(config['intra_n_bins'], config['n_heads'], bias=False)
        
        self.blocks = nn.ModuleList([
            TriangleAttBlock(c_pair=config['c_pair'], h_trimul=config['h_trimul'], h_ptrans=config['h_ptrans'], 
                             n_heads=config['n_heads'], dropout=config['dropout'], use_checkpoint=config['use_checkpoint'])
            for _ in range(config['n_blocks'])
        ])

        # contacts
        self.dsthead = ContactHead(c_in=config['c_pair'], c_hidden=config['h_conhead'])


    def forward(self, batch):
        
        # mk batch
        device = next(self.parameters()).device
        batch = batch_to_device(batch, device)
        batch = batch_on_the_fly(batch, intra_n_bins=self.intra_n_bins, 
                                 intra_max_dist=self.intra_max_dist, th_dst=self.th_dst)
        
        # embedders 
        batch = self.inter2emb(batch)
        batch = self.intra2emb(batch)

        batch = self.inter2mat(batch)
        batch = self.intra2mat(batch)
        #dimertype = self.types2mat(batch['is_homo']).unsqueeze(1).unsqueeze(1)   # [B, 1, 1, c_pair]

        pairmsk = batch['a_seqmsk'][:, :, None, None].float() * batch['b_seqmsk'][:, None, :, None].float()
        batch['c_pair'] = (batch['c_inter'] + batch['c_intra']) * pairmsk

        # clearance
        #del dimertype
        for k in ['a_mint', 'b_mint', 'a_esm3', 'b_esm3', 'a_interin', 'b_interin', 'a_intrain', 'b_intrain',
                  'c_inter', 'c_intra']:
            batch.pop(k, None)

        # intra-distances
        a_pair = self.intraemb(batch['a_pair']) * (batch['a_seqmsk'][:, :, None, None] * batch['a_seqmsk'][:, None, :, None])
        b_pair = self.intraemb(batch['b_pair']) * (batch['b_seqmsk'][:, :, None, None] * batch['b_seqmsk'][:, None, :, None])
        
        aa_mask = (batch['a_seqmsk'][:, None, :, None] * batch['a_seqmsk'][:, None, None, :]).float()  # [B, 1, Na, Na]
        bb_mask = (batch['b_seqmsk'][:, None, :, None] * batch['b_seqmsk'][:, None, None, :]).float()  # [B, 1, Nb, Nb]

        aa_bias = self.intrabias(batch['a_pair']).permute(0, 3, 1, 2) * aa_mask
        bb_bias = self.intrabias(batch['b_pair']).permute(0, 3, 1, 2) * bb_mask
        
        # clearance
        for k in ['a_pair', 'b_pair']:
            batch.pop(k, None)

        # triangles
        for blk in self.blocks:
            batch = blk(batch, a_pair, b_pair, aa_bias, bb_bias, pairmsk)

        # contacts
        batch = self.dsthead(batch)

        return batch


def countparams(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters total: {total:,d}\nParameters train: {trainable:,d}")