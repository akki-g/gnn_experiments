import torch
import torch.nn as nn
from torch.distributions import Normal

from HetGAT.preProcessing import ClassPreprocessor
from HetGAT.hetgatLayer import HetGATLayer
from HetGAT.graph import build_hetero_adj

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

        self.preprocess_scout = ClassPreprocessor(obs_dim_scout, state_dim_scout, hidden_dim)
        self.preprocess_interc = ClassPreprocessor(obs_dim_interc, state_dim_interc, hidden_dim)

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

        # LayerNorm after each HetGAT layer (applied to scout and interc separately)
        self.ln_scout = nn.ModuleList([nn.LayerNorm(n_heads * head_dim) for _ in range(n_layers)])
        self.ln_interc = nn.ModuleList([nn.LayerNorm(n_heads * head_dim) for _ in range(n_layers)])

        final_feat_dim = n_heads * head_dim
        #project mean agent feats into ssn space for critic enrichment
        self.ssn_enrich = nn.Linear(2*final_feat_dim, final_feat_dim)
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
        h_scout, new_hidden_s = self.preprocess_scout(
            obs_scout, state_scout, hidden_scout
        )
        h_interc, new_hidden_i = self.preprocess_interc(
            obs_interc, state_interc, hidden_interc
        )

        # ssn feats
        h_ssn = self.ssn_proj(ssn_input).unsqueeze(1) # (B, 1, d)

        # build adj graph
        adj = build_hetero_adj(positions, self.r_comm, n_scouts, n_interc)
        
        # hetGAT stack with residual connections and LayerNorm
        for i, layer in enumerate(self.hetgatLayers):
            h_scout_new, h_interc_new, h_ssn = layer(h_scout, h_interc, h_ssn, adj)
            # Apply LayerNorm
            h_scout_new = self.ln_scout[i](h_scout_new)
            h_interc_new = self.ln_interc[i](h_interc_new)
            # Residual connection (skip layer 0 — dimension changes from 2*hidden_dim to n_heads*head_dim)
            if i > 0:
                h_scout_new = h_scout_new + h_scout
                h_interc_new = h_interc_new + h_interc
            h_scout = h_scout_new
            h_interc = h_interc_new

        #enrich ssn with mean feats for better val estimation
        scout_mean_feat = h_scout.mean(dim=1, keepdim=True)
        interc_mean_feat = h_interc.mean(dim=1, keepdim=True)
        agent_summary = torch.cat([scout_mean_feat, interc_mean_feat], dim=-1)
        h_ssn = h_ssn + self.ssn_enrich(agent_summary) # residual enrichment

        # action heads
        scout_mean = self.action_head_scout(h_scout)
        interc_mean = self.action_head_interc(h_interc)

        scout_std = self.log_std_scout.clamp(min=-5.0, max=2.0).exp().expand_as(scout_mean)
        interc_std = self.log_std_interc.clamp(min=-5.0, max=2.0).exp().expand_as(interc_mean)

        scout_dist = Normal(scout_mean, scout_std)
        interc_dist = Normal(interc_mean, interc_std)

        return scout_dist, interc_dist, h_ssn, new_hidden_s, new_hidden_i
    
    @staticmethod
    def sample_action(dist):
        """Sample action from Normal distribution. No tanh — VMAS clamps via max_speed.

        Returns:
            action: raw sample from distribution
            log_prob: log probability summed over action dims -> (B, n_agents)
        """
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob





