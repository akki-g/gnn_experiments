import torch 
import torch.nn as nn

class HetNetCritic(nn.Module):
    """
    per class critic
    takes SSN embeddins (global state summary) and optionally per agent embeddings.
    outputs value estimated used as baselines in policy grad
    """

    def __init__(self, 
                ssn_dim: int, # SSN embedding dim 
                agent_dim: int = 0, # agent embedding dim 0 = per-class node
                hidden_dim: int = 64,
                mode: str = 'per_class' # cetralized, per-class, or per agent
                ):
        super().__init__()
        self.mode = mode

        if mode == 'centralized':
            self.value_head = nn.Sequential(
                nn.Linear(ssn_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        elif mode == 'per_class':
            # one head per class, both read from ssn
            self.value_head_scout = nn.Sequential(
                nn.Linear(ssn_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.value_head_interc = nn.Sequential(
                nn.Linear(ssn_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # concat SSN + agent embedding, class specific heads
            self.value_head_scout = nn.Sequential(
                nn.Linear(ssn_dim+agent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.value_head_interc = nn.Sequential(
                nn.Linear(ssn_dim+agent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, 
                h_ssn: torch.Tensor, # (B, 1, ssn_dim)
                h_scout: torch.Tensor = None, # (B, n_s, agent_dim)
                h_interc: torch.Tensor = None # (B, n_i, agent_dim)
                ) -> dict[torch.Tensor]:
        
        """ Returns dict w value estimates
            centralized -> {'all': (B, 1)}
            per-class -> {'scout: (B, 1), 'interc': (B, 1)}
            per-agent -> {'scout': (B, n_s), 'interc': (B, n_i)}
        """

        ssn = h_ssn.squeeze(1) # (B, ssn_dim)

        if self.mode == 'centralized':
            return {'all': self.value_head(ssn)}
        
        elif self.mode == 'per_class':
            return {'scout': self.value_head_scout(ssn), 'interc': self.value_head_interc(ssn)}

        elif self.mode == 'per_agent':
            B, n_s, _ = h_scout.shape
            B, n_i, _ = h_interc.shape

            ssn_s = ssn.unsqueeze(1).expand(-1, n_s, -1) # (B, n_s, ssn_dim)
            ssn_i = ssn.unsqueeze(1).expand(-1, n_i, -1) # (B, n_i, ssn_dim)

            scout_input = torch.cat([ssn_s, h_scout], dim=-1)
            interc_input = torch.cat([ssn_i, h_interc], dim=-1)

            return {'scout': self.value_head_scout(scout_input), 'interc':self.value_head_interc(interc_input)}
        