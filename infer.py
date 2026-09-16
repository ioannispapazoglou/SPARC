
# SPARC v_2.2.9
# infer.py

import os, glob, torch, warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sparc.utils.pdb_utils import pdbprocess
from sparc.utils.plm_utils import load_esm3, embed_esm3, load_mint, embed_mint
from sparc.utils.infer_utils import featurize, load_sparc, pred2hdockrests
from sparc.utils.eval_utils import evaluate_single, collect_results

SPARC_CKPT = "model.pt"
SPARC_CFG  = "config.json"

MINT_SRC   = "mint"
MINT_CFG   = "esm2_t33_650M_UR50D.json"
MINT_CKPT  = "mint.ckpt"

device = torch.device("cuda")
is_homo = False

# load models once
esm3 = load_esm3(device)
mint, alphabet = load_mint(MINT_SRC, MINT_CFG, MINT_CKPT, device)
sparc = load_sparc(SPARC_CFG, SPARC_CKPT, device)

results_b = {}
results_u = {}

folders = sorted(glob.glob("./"))

for folder in tqdm(folders, desc="Folders"):
    folder = folder.rstrip('/')
    code = os.path.basename(folder)
    
    for suffix, seq_mode in [("b", "fromalign"), ("u", "fromalign")]:
        try:
            r_pdb = os.path.join(folder, f"A_{suffix}.pdb")
            l_pdb = os.path.join(folder, f"B_{suffix}.pdb")
            out   = os.path.join(folder, f"restraints_{suffix}.txt")
            
            inputs, gaps = pdbprocess(r_pdb, l_pdb, seq=seq_mode)
            inputs['a_esm3'], inputs['b_esm3'] = embed_esm3(esm3, inputs, gaps, device)
            inputs['a_mint'], inputs['b_mint'] = embed_mint(mint, alphabet, inputs, gaps, device)
            
            features = featurize(inputs, is_homo)
            with torch.no_grad():
                features = sparc(features)
            pred2hdockrests(features, inputs, top_k=None, output_path=out)
            
            row = evaluate_single(features)
            if row is not None:
                target = results_b if suffix == 'b' else results_u
                target[code] = row
            
        except Exception as e:
            print(f"[ERROR]:: {code}_{suffix}: {e}")

del esm3, mint, alphabet, sparc
torch.cuda.empty_cache()

df_b = collect_results(results_b)
df_u = collect_results(results_u)
df_b.to_csv("./ins/sparc_eval_bound.csv", index=False)
df_u.to_csv("./ins/sparc_eval_unbound.csv", index=False)

print(f"[INFO]:: Bound:   {len(df_b)} complexes, mean AUPRC={df_b['auprc'].mean():.4f}")
print(f"[INFO]:: Unbound: {len(df_u)} complexes, mean AUPRC={df_u['auprc'].mean():.4f}")
print("[INFO]:: Successfully finished! Bye.")