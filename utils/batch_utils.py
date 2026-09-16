
# SPARC v_2.2.9
# batch_utils

# Developed by Durdagi Lab (https://www.durdagilab.com/)

import torch 


def batch_to_device(batch, device):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def atom37_to_intrapair(atom37, ca_index=1):
    """
    atom37:      (B, N, 37, 3) tensor
    Returns:     (B, N, N) tensor
    """
    ca = atom37[:, :, ca_index, :]                          # (B, N, 3)
    return torch.cdist(ca, ca)                              # (B, N, N)


def atom37_to_mindst(atom37_a, atom37_b, mask_a=None, mask_b=None):
    """
    atom37_a: (B, N_a, 37, 3)
    atom37_b: (B, N_b, 37, 3)
    mask_a:   (B, N_a, 37)
    mask_b:   (B, N_b, 37)
    Returns:  (B, N_a, N_b) min heavy-atom distance
    """
    B, Na, A, _ = atom37_a.shape
    Nb = atom37_b.shape[1]

    # (B, Na*37, 3) and (B, Nb*37, 3)
    a = atom37_a.reshape(B, Na * A, 3)
    b = atom37_b.reshape(B, Nb * A, 3)

    d = torch.cdist(a, b)                       # (B, Na*37, Nb*37)
    d = d.reshape(B, Na, A, Nb, A)              # split atom dims back

    if mask_a is not None or mask_b is not None:
        if mask_a is None:
            mask_a = torch.ones(B, Na, A, dtype=torch.bool, device=a.device)
        if mask_b is None:
            mask_b = torch.ones(B, Nb, A, dtype=torch.bool, device=b.device)
        mask_a = mask_a.bool()
        mask_b = mask_b.bool()
        # (B, Na, A, Nb, A)
        valid = mask_a[:, :, :, None, None] & mask_b[:, None, None, :, :]
        d = d.masked_fill(~valid, float('inf'))
    return d.amin(dim=(2, 4))                   # (B, Na, Nb)


def scalar_to_rbf(x, num_bins, max_val):
    """
    dist: (..., ) any shape
    Returns: (..., num_bins)
    """
    bins = torch.linspace(0, max_val, num_bins, device=x.device)
    sigma = max_val / num_bins  # width of each Gaussian
    diff = x.unsqueeze(-1) - bins
    return torch.exp(-diff ** 2 / (2 * sigma ** 2))


def batch_on_the_fly(batch, intra_n_bins=32, intra_max_dist=22.0, th_dst=8.0):
    
    # labels
    batch['gt_pair'] = atom37_to_mindst(batch['a_crd'], batch['b_crd'], 
                                        batch['a_msk'], batch['b_msk'])                      # (B, N, N)
    
    # features
    batch['a_pair'] = atom37_to_intrapair(batch['a_crd'])                                    # (B, N, N)        
    batch['b_pair'] = atom37_to_intrapair(batch['b_crd'])                                    # (B, N, N)

    batch['a_pair'] = scalar_to_rbf(batch['a_pair'], intra_n_bins, max_val=intra_max_dist)   # (B, N, N, c_pair)
    batch['b_pair'] = scalar_to_rbf(batch['b_pair'], intra_n_bins, max_val=intra_max_dist)   # (B, N, N, c_pair)
    
    return batch