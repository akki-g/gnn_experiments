import torch
import torch.nn as nn
from torch.distributions import Normal

from preProcessing import ClassPreprocessor
from hetgatLayer import HetGATLayer
from graph import build_hetero_adj

class HetNetPolicy(nn.Module):
    """
    Full policy network
    owns: per-class preprocessors, HetGAT stack, per-class action heads.
    does not own the critic - CTDE
    """

    def __init__(self, 
                obs_dim_scout: int, 
                obs_dim_interc: int,
                state_dim_scout: int,
                state_dim_interc: int,
                action_dim: int,
                hidden_dim: int = 64,
                n_heads: int = 4,
                head_dim: int = 16, 
                n_layers: int = 3,
                ssn_input_dim: int = 5,
                r_comm: float = 1.0
                ):
        super().__init__()

        self.r_comm = r_comm
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.action_dim = action_dim

        self.preprcess_scout = ClassPreprocessor(obs_dim_scout, state_dim_scout, hidden_dim)
        self.preprcess_interc = ClassPreprocessor(obs_dim_interc, state_dim_interc, hidden_dim)

        # ssn input projection to match node feat dim
        gat_input_dim = 2 * hidden_dim
        self.ssn_proj = nn.Linear(ssn_input_dim, gat_input_dim)

        # hetgat stack
        self.hetgatLayers = nn.ModuleList()
        for l in range(n_layers):
            d_in = gat_input_dim if l == 0 else n_heads * head_dim
            # all layers use concat for continuous action
            self.hetgatLayers.append(
                HetGATLayer(
                    d_in=d_in,
                    d_out=head_dim,
                    n_heads=n_heads,
                    concat_heads=True # concat action head handles the rest
                )
            )

        final_feat_dim = n_heads * head_dim
        self.action_head_scout = nn.Sequential(
            nn.Linear(final_feat_dim, final_feat_dim),
            nn.ReLU(),
            nn.Linear(final_feat_dim, action_dim)
        )

        self.action_head_interc = nn.Sequential(
            nn.Linear(final_feat_dim, final_feat_dim),
            nn.ReLU(),
            nn.Linear(final_feat_dim, action_dim)
        )

        self.log_std_scout = nn.Parameter(torch.zeros(action_dim))
        self.log_std_interc = nn.Parameter(torch.zeros(action_dim))

    
    def forward(self,
                obs_scout: torch.Tensor, # (B, n_s, obs_dim_s)
                state_scout: torch.Tensor, # (B, n_s, state_dim_s)
                obs_interc: torch.Tensor, # (B, n_i, obs_dim_i)
                state_interc: torch.Tensor, # (B, n_i, state_dim_i)
                positions: torch.Tensor, # (B, n_defend, 2)
                ssn_input: torch.Tensor, #(B, ssn_input_dim)
                hidden_scout: dict, 
                hidden_interc: dict,
                n_scouts: int,
                n_interc: int
            ):
        """
        full forward pass
        obs -> preprocessing -> hetGAT stack -> action dists.

        returns: 
            scout_dist: normal dist, batch shape 
            interc_dist: normal dist, batch shape
            h_ssn: SSN embedding for critic 
            new_hidden_s: updated LSTM states for scouts
            new_hidden_i: updated LSTM states for interc
        """

        # preprocessing 
        h_scout, new_hidden_s = self.preprcess_scout(
            obs_scout, state_scout, hidden_scout
        )
        h_interc, new_hidden_i = self.preprcess_interc(
            obs_interc, state_interc, hidden_interc
        )

        # ssn feats
        h_ssn = self.ssn_proj(ssn_input).unsqueeze(1) # (B, 1, d)

        # build adj graph
        adj = build_hetero_adj(positions, self.r_comm, n_scouts, n_interc)
        
        # hetGAT stack
        for layer in self.hetgatLayers:
            h_scout, h_interc, h_ssn = layer(h_scout, h_interc, h_ssn, adj)

        # action heads
        scout_mean = self.action_head_scout(h_scout)
        interc_mean = self.action_head_interc(h_interc)

        scout_std = self.log_std_scout.exp().expand_as(scout_mean)
        interc_std = self.log_std_interc.exp().expand_as(interc_mean)

        scout_dist = Normal(scout_mean, scout_std)
        interc_dist = Normal(interc_mean, interc_std)

        return scout_dist, interc_dist, h_ssn, new_hidden_s, new_hidden_i





