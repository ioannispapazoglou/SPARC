
# SPARC v_2.2.9
# pdb_utils

# Developed by Durdagi Lab (https://www.durdagilab.com/)

import torch
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from biotite.structure import annotate_sse


atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_TO_IDX['X'] = 20

AF2_RESTYPES = "ARNDCQEGHILKMFPSTWYVX"

one_to_three = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK',
}


def load_structure(pdb_path):
    f = pdb_io.PDBFile.read(pdb_path)
    structure = f.get_structure(model=1, extra_fields=["b_factor"])
    chain = structure[struc.filter_amino_acids(structure)]
    chain = chain[~np.isin(chain.element, ["H", "D"])]
    
    chains = np.unique(chain.chain_id)
    if len(chains) != 1:
        raise ValueError(f"Expected single chain, got {len(chains)}: {chains}")
    
    return chain


def get_chainidx(chain):
    return np.unique(chain.chain_id)[0]


def get_seq_from_chain(chain):
    res_names = struc.get_residues(chain)[1]
    return ''.join(three_to_one.get(r, 'X') for r in res_names)


def get_seq_from_fasta(pdb_path):
    fa_path = pdb_path.replace('.pdb', '.fa')
    for line in open(fa_path):
        if not line.startswith('>'):
            return line.strip()


def get_seq_from_align(pdb_path):
    fa_path = pdb_path.replace('.pdb', '.fa')
    ref, tgt = None, None
    hdr = None
    for line in open(fa_path):
        if line.startswith('>'):
            hdr = line[1:].strip()
        else:
            if '_ref' in hdr: ref = line.strip()
            elif '_tgt' in hdr: tgt = line.strip()
    gap_mask = [c == '-' for c in tgt]  # True where residue is missing
    return ref, gap_mask


def get_gap_mask(pdb_seq, fasta_seq):
    p_i, gap_mask = 0, []
    for aa in fasta_seq:
        if p_i < len(pdb_seq) and pdb_seq[p_i] == aa:
            gap_mask.append(False)
            p_i += 1
        else:
            gap_mask.append(True)
    return gap_mask


def get_atom37(chain):
    res_ids_unique = struc.get_residues(chain)[0]
    num_res = len(res_ids_unique)

    all_atom_positions = np.zeros([num_res, atom_type_num, 3], dtype=np.float32)
    all_atom_mask = np.zeros([num_res, atom_type_num], dtype=np.float32)

    # map every atom's name -> atom37 slot (-1 if not in atom_order)
    atom_slots = np.array([atom_order.get(a, -1) for a in chain.atom_name], dtype=np.int64)
    valid = atom_slots >= 0

    # map every atom's res_id -> row index in unique list
    res_rows = np.searchsorted(res_ids_unique, chain.res_id)
    # guard: searchsorted can point past end / to wrong id for stray res_ids
    res_rows_clip = np.clip(res_rows, 0, num_res - 1)
    valid &= res_ids_unique[res_rows_clip] == chain.res_id
    res_rows = res_rows_clip

    r = res_rows[valid]
    s = atom_slots[valid]
    all_atom_positions[r, s] = chain.coord[valid]
    all_atom_mask[r, s] = 1.0

    return all_atom_positions, all_atom_mask, res_ids_unique.astype(np.int64)


def build_atom_array(seq_str, crd, msk, chain_id="A"):
    msk = np.asarray(msk, dtype=bool)
    res_idx, atom_idx = np.where(msk)
    coords   = crd[res_idx, atom_idx, :]
    atoms    = np.array(atom_types)[atom_idx]
    resnames = np.array([one_to_three.get(seq_str[i], "UNK") for i in res_idx])

    aa = struc.AtomArray(len(coords))
    aa.coord     = coords.astype(np.float32)
    aa.atom_name = atoms
    aa.res_name  = resnames
    aa.res_id    = (res_idx + 1).astype(int)
    aa.chain_id  = np.array([chain_id] * len(coords))
    aa.element   = np.array([a[0] for a in atoms])
    aa.hetero    = np.zeros(len(coords), dtype=bool)
    return aa


def compute_ss(chain):
    sse_3 = annotate_sse(chain)
    mapping = {'a': 'H', 'b': 'E', 'c': 'C', '': 'C'}
    return "".join(mapping[c] for c in sse_3)


def sasa_gpu(coords, mask, probe_radius=1.4, n_points=32, chunk_size=512):
    dev = coords.device

    radii = torch.tensor([1.55,1.70,1.70,1.70,1.52,1.70,1.70,1.70,1.52,1.52,
                          1.80,1.70,1.70,1.70,1.55,1.55,1.52,1.52,1.80,1.70,
                          1.70,1.70,1.70,1.55,1.55,1.55,1.52,1.52,1.70,1.55,
                          1.55,1.52,1.70,1.70,1.70,1.55,1.52], device=dev)

    radii_full = radii.unsqueeze(0).expand(mask.shape[0], 37)
    bmask = mask.bool()
    flat_coords = coords[bmask]
    flat_radii = radii_full[bmask]

    M = flat_coords.shape[0]
    r = flat_radii + probe_radius

    n = n_points
    indices = torch.arange(n, device=dev, dtype=torch.float32)
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)
    theta = torch.pi * (1 + 5**0.5) * indices
    unit_points = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)

    exposed_frac = torch.zeros(M, device=dev)
    atom_ids = torch.arange(M, device=dev)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk_coords = flat_coords[start:end]
        chunk_r = r[start:end]

        sp = chunk_coords[:, None, :] + chunk_r[:, None, None] * unit_points[None, :, :]
        diff = sp[:, :, None, :] - flat_coords[None, None, :, :]
        dist_sq = (diff ** 2).sum(-1)
        r_sq = r[None, None, :] ** 2

        # self-exclusion without the (chunk, n, M) bool tensor
        is_self = (atom_ids[start:end][:, None] == atom_ids[None, :])  # (chunk, M)
        buried = ((dist_sq < r_sq) & ~is_self[:, None, :]).any(dim=-1)
        exposed_frac[start:end] = (~buried).float().mean(dim=1)

    atom_area = 4 * torch.pi * r**2 * exposed_frac

    N_res = mask.shape[0]
    atom_res_idx = torch.arange(N_res, device=dev).unsqueeze(1).expand_as(mask)[bmask]
    per_res_sasa = torch.zeros(N_res, device=dev)
    per_res_sasa.scatter_add_(0, atom_res_idx, atom_area)

    return per_res_sasa


def compute_sasa(crd, msk, chunk_size=512):
    cs = chunk_size
    crd_t = torch.from_numpy(crd).cuda()
    msk_t = torch.from_numpy(msk.astype(np.float32)).cuda()
    while True:
        try:
            sasa = sasa_gpu(crd_t, msk_t, n_points=32, chunk_size=cs).cpu().numpy().astype(np.float32)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cs //= 2
            if cs < 16:
                raise RuntimeError("OOM even at chunk_size=16")
            print(f"apo OOM, retrying chunk_size={cs}")
    del crd_t, msk_t
    return sasa


def pdbprocess(a_path, b_path, seq='fromfasta'):
    a_chain = load_structure(a_path)
    b_chain = load_structure(b_path)

    a_cid = get_chainidx(a_chain)
    b_cid = get_chainidx(b_chain)

    if seq == 'frompdb':
        
        print('[CAUTION]:: In case of structure having gaps, provide a uniprot fasta file.')
        a_seq = get_seq_from_chain(a_chain)
        b_seq = get_seq_from_chain(b_chain)
        
        a_gap_msk = [False] * len(a_seq)
        b_gap_msk = [False] * len(b_seq)
        
    elif seq == 'fromfasta':
        
        a_seq = get_seq_from_fasta(a_path)
        b_seq = get_seq_from_fasta(b_path)
        
        a_gapped_seq = get_seq_from_chain(a_chain)
        b_gapped_seq = get_seq_from_chain(b_chain)
        
        a_gap_msk = get_gap_mask(a_gapped_seq, a_seq)
        b_gap_msk = get_gap_mask(b_gapped_seq, b_seq)
        
    elif seq == 'fromalign':
        
        a_seq, a_gap_msk = get_seq_from_align(a_path)
        b_seq, b_gap_msk = get_seq_from_align(b_path)
        
    a_crd, a_msk, a_res = get_atom37(a_chain)
    b_crd, b_msk, b_res = get_atom37(b_chain)

    a_ss = compute_ss(a_chain)
    b_ss = compute_ss(b_chain)

    a_sasa = compute_sasa(a_crd, a_msk)
    b_sasa = compute_sasa(b_crd, b_msk)
    
    return {
        "a_chain": a_chain, "b_chain": b_chain,
        "a_cid":   a_cid,   "b_cid":   b_cid,
        "a_seq":   a_seq,   "b_seq":   b_seq,
        "a_crd":   a_crd,   "b_crd":   b_crd,
        "a_msk":   a_msk,   "b_msk":   b_msk,
        "a_res":   a_res,   "b_res":   b_res,
        "a_ss":    a_ss,    "b_ss":    b_ss,
        "a_sasa":  a_sasa,  "b_sasa":  b_sasa,
    }, {"a_gaps": a_gap_msk, "b_gaps": b_gap_msk}
    

"""
def pdbprocess(a_path, b_path):
    a_chain = load_structure(a_path)
    b_chain = load_structure(b_path)

    a_cid = get_chainidx(a_chain)
    b_cid = get_chainidx(b_chain)

    a_seq = get_seq_from_chain(a_chain)
    b_seq = get_seq_from_chain(b_chain)

    a_crd, a_msk, a_res = get_atom37(a_chain)
    b_crd, b_msk, b_res = get_atom37(b_chain)

    a_ss = compute_ss(a_chain)
    b_ss = compute_ss(b_chain)

    a_sasa = compute_sasa(a_crd, a_msk)
    b_sasa = compute_sasa(b_crd, b_msk)
    
    return {
        "a_chain": a_chain, "b_chain": b_chain,
        "a_cid":   a_cid,   "b_cid":   b_cid,
        "a_seq":   a_seq,   "b_seq":   b_seq,
        "a_crd":   a_crd,   "b_crd":   b_crd,
        "a_msk":   a_msk,   "b_msk":   b_msk,
        "a_res":   a_res,   "b_res":   b_res,
        "a_ss":    a_ss,    "b_ss":    b_ss,
        "a_sasa":  a_sasa,  "b_sasa":  b_sasa,
    }
"""