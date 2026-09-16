
# SPARC v_2.2.9
# plm_utils

# Developed by Durdagi Lab (https://www.durdagilab.com/)

import sys
import torch
import numpy as np

from esm.pretrained import ESM3_sm_open_v0
from esm.sdk.api import ESMProtein, SamplingConfig


def load_esm3(device):
    
    print("[INFO]:: Loading ESM-3...")
    esm3 = ESM3_sm_open_v0(device=device).eval()
    print(f"[INFO]:: Loaded!")
    
    return esm3

def insert_gaps_coords(crd, gap_mask):
    """Insert NaN rows where gap_mask is True."""
    L_full = len(gap_mask)
    out = np.full((L_full,) + crd.shape[1:], np.nan, dtype=np.float32)
    out[~np.array(gap_mask)] = crd
    return out

def insert_gaps_sasa(sasa, gap_mask):
    """Insert None where gap_mask is True."""
    out = []
    j = 0
    for g in gap_mask:
        if g:
            out.append(None)
        else:
            out.append(float(sasa[j]))
            j += 1
    return out

def insert_gaps_ss(ss_str, gap_mask):
    """Insert <pad> where gap_mask is True."""
    out = []
    j = 0
    for g in gap_mask:
        if g:
            out.append("<pad>")
        else:
            out.append(ss_str[j])
            j += 1
    return "".join(out)

@torch.no_grad()
def embed_esm3(model, inputs, gaps, device):

    embs = {}
    for cid in ["a", "b"]:
            
        seq  = inputs[f'{cid}_seq']   # full seq (with gap AAs)
        crd  = inputs[f'{cid}_crd']   # structure coords (no gaps)
        sasa = inputs[f'{cid}_sasa']  # sasa (no gaps)
        ss   = inputs[f'{cid}_ss']    # ss string (no gaps)
        gap_mask = np.array(gaps[f'{cid}_gaps'], dtype=bool)

        # expand structure tracks to full length (seq filled, str masked)
        crd_full  = insert_gaps_coords(crd, gap_mask)
        sasa_full = insert_gaps_sasa(sasa, gap_mask)
        ss_full   = insert_gaps_ss(ss, gap_mask)

        coords = torch.from_numpy(crd_full).to(device)

        protein = ESMProtein(
            sequence=seq,
            coordinates=coords,
            sasa=sasa_full,
            secondary_structure=ss_full,
        )
        pt  = model.encode(protein)
        out = model.forward_and_sample(
            pt, SamplingConfig(return_per_residue_embeddings=True),
        )
        emb = out.per_residue_embedding[1:-1].float().cpu().numpy()
        
        # crop back to resolved positions only
        emb = emb[~gap_mask]
        embs[cid] = emb.astype(np.float32)
        
    return embs["a"], embs["b"]

"""
@torch.no_grad()
def embed_esm3(model, inputs, device):

    embs = {}
    for cid in ["a", "b"]:
            
        seq, crd, sasa, ss = inputs[f'{cid}_seq'], inputs[f'{cid}_crd'], inputs[f'{cid}_sasa'], inputs[f'{cid}_ss']
        coords = torch.from_numpy(np.asarray(crd, dtype=np.float32)).to(device)

        protein = ESMProtein(
            sequence=seq, # str
            coordinates=coords, # numpy / tensor 
            sasa=np.asarray(sasa, dtype=np.float32).tolist(), # list of floats
            secondary_structure=ss, # str
        )
        pt  = model.encode(protein)
        out = model.forward_and_sample(
            pt, SamplingConfig(return_per_residue_embeddings=True),
        )
        embs[cid] = out.per_residue_embedding[1:-1].float().cpu().numpy().astype(np.float32)
        
    return embs["a"], embs["b"]
"""

def load_mint(src_path, cfg_path, ckpt_path, device):
    sys.path.insert(0, src_path)
    from mint.data import Alphabet
    from mint.helpers.extract import load_config, MINTWrapper

    print("[INFO]:: Loading MINT...")
    cfg = load_config(cfg_path)
    model = MINTWrapper(cfg, ckpt_path, device=device).eval()
    alphabet = Alphabet.from_architecture("ESM-1b")
    print("[INFO]:: Loaded!")
    return model, alphabet

@torch.no_grad()
def embed_mint(mint, alphabet, inputs, gaps, device):
    # a/b_seq are the full seqs (no matter the structural gaps)
    ta = alphabet.encode("<cls>" + inputs['a_seq'].replace("J","L") + "<eos>")
    tb = alphabet.encode("<cls>" + inputs['b_seq'].replace("J","L") + "<eos>")
    toks = torch.tensor(ta + tb, dtype=torch.int64)[None].to(device)
    cids = torch.cat([
        torch.zeros(len(ta), dtype=torch.int32),
        torch.ones(len(tb), dtype=torch.int32),
    ])[None].to(device)
    
    m = mint.model
    h = m(toks, cids, repr_layers=[33])["representations"][33][0]
    real = ~toks[0].eq(m.cls_idx) & ~toks[0].eq(m.eos_idx) & ~toks[0].eq(m.padding_idx)
    c = cids[0]
    emb_a = h[real & (c == 0)].float().cpu().numpy()
    emb_b = h[real & (c == 1)].float().cpu().numpy()
    
    # crop back to resolved positions only
    gap_a = np.array(gaps['a_gaps'], dtype=bool)
    gap_b = np.array(gaps['b_gaps'], dtype=bool)
    emb_a = emb_a[~gap_a]
    emb_b = emb_b[~gap_b]
    
    return emb_a, emb_b

"""
@torch.no_grad()
def embed_mint(mint, alphabet, inputs, device):
    ta = alphabet.encode("<cls>" + inputs['a_seq'].replace("J","L") + "<eos>")
    tb = alphabet.encode("<cls>" + inputs['b_seq'].replace("J","L") + "<eos>")
    toks = torch.tensor(ta + tb, dtype=torch.int64)[None].to(device)
    cids = torch.cat([
        torch.zeros(len(ta), dtype=torch.int32),
        torch.ones(len(tb), dtype=torch.int32),
    ])[None].to(device)
    
    m = mint.model
    h = m(toks, cids, repr_layers=[33])["representations"][33][0]
    real = ~toks[0].eq(m.cls_idx) & ~toks[0].eq(m.eos_idx) & ~toks[0].eq(m.padding_idx)
    c = cids[0]
    emb_a = h[real & (c == 0)].float().cpu().numpy()
    emb_b = h[real & (c == 1)].float().cpu().numpy()
    return emb_a, emb_b
"""