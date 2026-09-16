
""" 

Triangles Update Block

SPARC by DurdagiLab
# v2.2.9

"""


import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint



class TriangleUpdateOutgoing(nn.Module): 
    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.c_hidden = c_hidden

        self.proj_ab = nn.Linear(c_pair, c_hidden)
        self.gate_ab = nn.Linear(c_pair, c_hidden)

        self.layer_norm_in = nn.LayerNorm(c_pair)
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        self.linear_z = nn.Linear(c_hidden, c_pair)
        self.linear_g = nn.Linear(c_pair, c_pair)

    def forward(self, intermap, intramap):
        # intermap: [B, Na, Nb, C]
        # intramap: [B, N, N, C] - with N = Na or Nb

        ab_normed = self.layer_norm_in(intermap)
        ab = torch.sigmoid(self.gate_ab(ab_normed)) * self.proj_ab(ab_normed)

        # Outgoing: intra @ inter — "Known chain's structure informs the interface"
        # [B, Na, Na, c_h] @ [B, Na, Nb, c_h] -> [B, Na, Nb, c_h]
        update = torch.einsum('bijc,bjkc->bikc', intramap, ab)

        update = self.layer_norm_out(update)
        update = self.linear_z(update)
        gate = torch.sigmoid(self.linear_g(ab_normed))

        return intermap + gate * update



class TriangleUpdateIncoming(nn.Module): 
    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.c_hidden = c_hidden

        self.proj_ab = nn.Linear(c_pair, c_hidden)
        self.gate_ab = nn.Linear(c_pair, c_hidden)

        self.layer_norm_in = nn.LayerNorm(c_pair)
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        self.linear_z = nn.Linear(c_hidden, c_pair)
        self.linear_g = nn.Linear(c_pair, c_pair)

    def forward(self, intermap, intramap):
        # intermap: [B, Na, Nb, C]
        # intramap: [B, N, N, C] - with N = Na or Nb

        ab_normed = self.layer_norm_in(intermap)
        ab = torch.sigmoid(self.gate_ab(ab_normed)) * self.proj_ab(ab_normed)

        # Incoming: inter @ intra — "Known structure refines the interface"
        # [B, Na, Nb, c_h] @ [B, Nb, Nb, c] -> [B, Na, Nb, c_h]
        update = torch.einsum('bikc,bkjc->bijc', ab, intramap)

        update = self.layer_norm_out(update)
        update = self.linear_z(update)
        gate = torch.sigmoid(self.linear_g(ab_normed))

        return intermap + gate * update



class IntraInfTriangleUpdate(nn.Module):
    def __init__(self, c_pair, c_hidden):
        super().__init__()

        # Outgoing: chain structure -> interface
        self.outgoing_a = TriangleUpdateOutgoing(c_pair, c_hidden)
        self.outgoing_b = TriangleUpdateOutgoing(c_pair, c_hidden)
        # Incoming: interface <- partner structure
        self.incoming_a = TriangleUpdateIncoming(c_pair, c_hidden)
        self.incoming_b = TriangleUpdateIncoming(c_pair, c_hidden)

        for m in (self.outgoing_a, self.outgoing_b, self.incoming_a, self.incoming_b):
            nn.init.zeros_(m.linear_z.weight); nn.init.zeros_(m.linear_z.bias)

    def forward(self, c, a_pair, b_pair):
        # AB @ BB (incoming) = AB'
        c = self.incoming_b(c, b_pair)
        # AA @ AB' (outgoing) = AB''
        c = self.outgoing_a(c, a_pair)
        # Permute AB'' = BA
        c = c.permute(0, 2, 1, 3).contiguous()
        # BA @ AA (incoming) = BA'
        c = self.incoming_a(c, a_pair)
        # BB @ BA' (outgoing) = BA''
        c = self.outgoing_b(c, b_pair)
        # Permute BA'' = AB'''
        return c.permute(0, 2, 1, 3).contiguous()



class AxisAttention(nn.Module):
    def __init__(self, c_pair, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        assert c_pair % n_heads == 0
        self.head_dim = c_pair // n_heads

        self.norm = nn.LayerNorm(c_pair)
        
        self.q = nn.Linear(c_pair, c_pair, bias=False)
        self.k = nn.Linear(c_pair, c_pair, bias=False)
        self.v = nn.Linear(c_pair, c_pair, bias=False)
        self.o = nn.Linear(c_pair, c_pair)
        
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)   # identity at init

    def forward(self, x, bias, mask=None):
        # x:        [B_eff, L, C]
        # tri_bias: [B_eff, H, L, L]  additive per-head bias from the intra map
        # mask:     [B_eff, L]        1 = real, 0 = pad
        B, L, _ = x.shape
        H, d = self.n_heads, self.head_dim

        xn = self.norm(x)
        q = self.q(xn).view(B, L, H, d).transpose(1, 2)     # [B, H, L, d]
        k = self.k(xn).view(B, L, H, d).transpose(1, 2)
        v = self.v(xn).view(B, L, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)   # [B, H, L, L]
        attn = attn + bias
        if mask is not None:
            keypad = (mask == 0)[:, None, None, :]          # [B,1,1,L]
            attn = attn.masked_fill(keypad, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                         # [B, H, L, d]
        out = out.transpose(1, 2).reshape(B, L, H * d)
        
        return self.o(out)



class AxialAttentionLayer(nn.Module):
    def __init__(self, c_pair, n_heads, dropout=0.1, chunk=None):
        super().__init__()
        
        self.row_attn = AxisAttention(c_pair, n_heads, dropout)
        self.col_attn = AxisAttention(c_pair, n_heads, dropout)
        
        self.chunk = chunk   # chunking at inference

    def _axis(self, attn, x, bias, mask, R):
        # x: [B*R, L, C]  bias: [B, H, L, L]  mask: [B, L] or None  R: repeats per batch elem
        B, H, L, _ = bias.shape
        BR = x.shape[0]

        if self.chunk is None or self.training:
            bias_e = bias.unsqueeze(1).expand(B, R, H, L, L).reshape(BR, H, L, L)               # [B*Na, H, Nb, Nb] or [B*Nb, H, Na, Na]
            mask_e = None if mask is None else mask[:, None, :].expand(B, R, L).reshape(BR, L)  # [B*Na, Nb] or [B*Nb, Na]
            return attn(x, bias_e, mask_e)

        outs = []
        for s in range(0, BR, self.chunk):
            e = min(s + self.chunk, BR)
            bidx = torch.arange(s, e, device=x.device) // R      # row -> batch element
            outs.append(attn(x[s:e], bias[bidx], None if mask is None else mask[bidx]))
        return torch.cat(outs, dim=0)
    
    def forward(self, pair, aa_bias, bb_bias, mask_a=None, mask_b=None):
        # pair:    [B, Na, Nb, C]
        # aa_bias: [B, H, Na, Na]
        # bb_bias: [B, H, Nb, Nb]
        # mask_a:  [B, Na]
        # mask_b:  [B, Nb]
        B, Na, Nb, C = pair.shape
        H = self.row_attn.n_heads

        # === Row attention: attend along Nb, biased by BB ===
        row_in = pair.reshape(B * Na, Nb, C)                                    # [B*Na, Nb, C]
        row_out = self._axis(self.row_attn, row_in, bb_bias, mask_b, Na)        # [B*Na, Nb, C]

        pair = pair + row_out.reshape(B, Na, Nb, C)
        
        # substituted by _axis 
        #bb_exp = bb_bias.unsqueeze(1).expand(B, Na, H, Nb, Nb).reshape(B * Na, H, Nb, Nb)  # [B*Na, H, Nb, Nb]
        #row_mask = None
        #if mask_b is not None:
        #    row_mask = mask_b[:, None, :].expand(B, Na, Nb).reshape(B * Na, Nb)  # [B*Na, Nb]
        #pair = pair + self.row_attn(row_in, bb_exp, row_mask).reshape(B, Na, Nb, C)

        # === Column attention: attend along Na, biased by AA ===
        col_in = pair.permute(0, 2, 1, 3).reshape(B * Nb, Na, C)                # [B*Nb, Na, C]
        col_out = self._axis(self.col_attn, col_in, aa_bias, mask_a, Nb)        # [B*Nb, Na, C]

        pair = pair + col_out.reshape(B, Nb, Na, C).permute(0, 2, 1, 3)
        
        # substituted by _axis 
        #aa_exp = aa_bias.unsqueeze(1).expand(B, Nb, H, Na, Na).reshape(B * Nb, H, Na, Na)  # [B*Nb, H, Na, Na]
        #col_mask = None
        #if mask_a is not None:
        #    col_mask = mask_a[:, None, :].expand(B, Nb, Na).reshape(B * Nb, Na)  # [B*Nb, Na]
        #pair = pair + self.col_attn(col_in, aa_exp, col_mask).reshape(B, Nb, Na, C).permute(0, 2, 1, 3)
        
        return pair



class PairTransition(nn.Module):
    def __init__(self, c_pair, c_hidden):
        super().__init__()

        self.c_pair = c_pair
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(c_pair)
        self.linear_1 = nn.Linear(self.c_pair, self.c_hidden)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.c_hidden, c_pair)
        nn.init.zeros_(self.linear_2.weight); nn.init.zeros_(self.linear_2.bias)

    def forward(self, c, mask):

        upd = self.layer_norm(c)
        upd = self.linear_1(upd)
        upd = self.relu(upd)
        upd = self.linear_2(upd)
        upd = upd * mask

        return c + upd   # residual
    
    
    
class TriangleAttBlock(nn.Module):
    def __init__(self, c_pair, h_trimul, h_ptrans, n_heads, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.trimul = IntraInfTriangleUpdate(c_pair, h_trimul)
        self.axattn = AxialAttentionLayer(c_pair, n_heads, dropout)
        self.ptransit = PairTransition(c_pair, h_ptrans)

    def _run(self, z, a_pair, b_pair, aa_bias, bb_bias, pairmsk, mask_a, mask_b):
        z = self.trimul(z, a_pair, b_pair)
        z = self.axattn(z, aa_bias, bb_bias, mask_a, mask_b)
        z = self.ptransit(z, pairmsk)
        return z

    def forward(self, batch, a_pair, b_pair, aa_bias, bb_bias, pairmsk):
        if self.use_checkpoint and self.training:
            batch['c_pair'] = checkpoint(self._run, batch['c_pair'],
                                         a_pair, b_pair, aa_bias, bb_bias,
                                         pairmsk, batch['a_seqmsk'], batch['b_seqmsk'],
                                         use_reentrant=False)
        else:
            batch['c_pair'] = self._run(batch['c_pair'], a_pair, b_pair,
                                        aa_bias, bb_bias, pairmsk, batch['a_seqmsk'], batch['b_seqmsk'])
        return batch