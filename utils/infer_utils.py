
# SPARC v_2.2.9
# batch_utils

# Developed by Durdagi Lab (https://www.durdagilab.com/)

import json
import torch 
import numpy as np 

from sparc.model import SPARC 


def print_sparc_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███████╗██████╗  █████╗ ██████╗  ██████╗                       ║
║   ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝                       ║
║   ███████╗██████╔╝███████║██████╔╝██║                            ║
║   ╚════██║██╔═══╝ ██╔══██║██╔══██╗██║                            ║
║   ███████║██║     ██║  ██║██║  ██║╚██████╗                       ║
║   ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                       ║
║                                                                  ║
║   Sequence-based Protein Assessment for Residue Contacts         ║
║                                                                  ║
║ ──────────────────────────────────────────────────────────────── ║
║                                                                  ║
║   Durdagi Lab | Bahcesehir University                            ║
║   A project funded by EU (RETORNA ITN)                           ║
║                                                                  ║
║   Version: 2.2.8+                                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    
def load_sparc(cfg_path, checkpoint, device):
    with open(cfg_path) as f:
        config = json.load(f)

    print("[INFO]:: Loading SPARC...")
    sparc = SPARC(config).to(device)
    ckpt = torch.load(checkpoint, map_location=config['device'], weights_only=False)
    sparc.load_state_dict(ckpt['model_state_dict'])
    print(f"[INFO]:: Loaded!")
    
    return sparc


def featurize(inputs, is_homo):
    La = inputs["a_crd"].shape[0]
    Lb = inputs["b_crd"].shape[0]
    d = {
        "a_crd":      torch.from_numpy(inputs["a_crd"]).unsqueeze(0),
        "a_msk":      torch.from_numpy(inputs["a_msk"]).unsqueeze(0),
        "a_esm3":     torch.from_numpy(inputs["a_esm3"]).unsqueeze(0),
        "a_mint":     torch.from_numpy(inputs["a_mint"]).unsqueeze(0),
        "a_seqmsk":   torch.ones(1, La),
        "b_crd":      torch.from_numpy(inputs["b_crd"]).unsqueeze(0),
        "b_msk":      torch.from_numpy(inputs["b_msk"]).unsqueeze(0),
        "b_esm3":     torch.from_numpy(inputs["b_esm3"]).unsqueeze(0),
        "b_mint":     torch.from_numpy(inputs["b_mint"]).unsqueeze(0),
        "b_seqmsk":   torch.ones(1, Lb),
        "is_homo":    torch.tensor([int(is_homo)], dtype=torch.long),
    }
    return d


def pred2hdockrests(batch, inputs, top_k=None, output_path="hdock_restraints.txt"):
    a_res, b_res = inputs["a_res"], inputs["b_res"]
    mask = batch["a_seqmsk"][0].unsqueeze(1) * batch["b_seqmsk"][0].unsqueeze(0)
    probs = (torch.sigmoid(batch['c_logits'][0]) * mask)

    if top_k is not None:
        _, idx = probs.flatten().topk(min(top_k, (probs > 0).sum().item()))
        pairs = torch.stack([idx // probs.shape[1], idx % probs.shape[1]], dim=1)
    else:
        pairs = (probs > 0.5).nonzero(as_tuple=False)

    lines = [f"{a_res[i].item()}:{inputs['a_cid']} {b_res[j].item()}:{inputs['b_cid']} 8" for i, j in pairs.tolist()]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def savepreds(features, output_path):
    pred = torch.sigmoid(features['c_logits'][0]).detach().cpu().numpy()
    gt = features['gt_pair'][0].detach().cpu().numpy()
    np.savez_compressed(output_path, pred=pred, gt=gt)
