import torch
from typing import Dict

def build_hetero_adj(positions: torch.Tensor, r_comm: float, n_scouts: int, n_interc: int) -> Dict[str, torch.Tensor]:
    """builds per-edge-type binary adj matrix for HetGAT style architectures 
        returns dict w keys like scout to scout , scout to intercept etc.
        each val is a binary mask of shape (batch, n_src, n_dst)
        convention: mask[b, src, dst] = 1 means src sent to dst in env b
        no row norm - attention handles weighting
    """
    n_s = n_scouts
    n_i = n_interc
        
    #calculate dist 
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    dist = torch.linalg.vector_norm(diff, dim=-1)

    connected = (dist <= r_comm).float()

    #zero out self loops 
    eye = torch.eye(n_s + n_i, device=positions.device).unsqueeze(0)
    connected = connected * (1.0 - eye)

    #slice in to quadrants by agent type 
    #rows = source, col = dest
    adj = {
        'scout_to_scout': connected[:, :n_s, :n_s],
        'scout_to_interc': connected[:, :n_s, n_s:], 
        'interc_to_scout': connected[:, n_s:, :n_s],
        'interc_to_interc': connected[:, n_s:, n_s:]
    }

    # ssn edges -> all agents connected duting training 
    batch = positions.shape[0]
    adj['scout_to_ssn'] = torch.ones(batch, n_s, 1, device=positions.device)
    adj['interc_to_ssn'] = torch.ones(batch, n_i, 1, device=positions.device)

    return adj

       