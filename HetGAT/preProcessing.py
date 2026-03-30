import torch 
from torch.nn import functional as F
import torch.nn as nn


class ClassPreprocessor(nn.Module):
    """per-class feat preprocessing: fc + lstm for state and obs"""

    def __init__(self, obs_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.fc_obs = nn.Linear(obs_dim, hidden_dim)
        self.lstm_obs = nn.LSTMCell(hidden_dim, hidden_dim)

        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.lstm_state = nn.LSTMCell(hidden_dim, hidden_dim)


    def init_hidden(self, batch_size: int, n_agents: int, device: torch.device):
        """call at ep start, returns dict of hidden state"""
        # flatten batch * agents into one batch dim for lstm cell 
        flat = batch_size * n_agents

        return {
            'obs_h': torch.zeros(flat, self.hidden_dim, device=device),
            'obs_c': torch.zeros(flat, self.hidden_dim, device=device),
            'state_h': torch.zeros(flat, self.hidden_dim, device=device),
            'state_c': torch.zeros(flat, self.hidden_dim, device=device)
        }
    
    def forward(self, obs: torch.Tensor, # (batch, n_agents, obs_dim)
                state: torch.Tensor, # (batch, n_agents, state_dim)
                hidden: dict, # from init or prev step
                ) -> tuple[torch.Tensor, dict]:
    
        """ returns: 
            node_feats: (batch, n_agents, 2*hidden_dim)
            new_hidden: updated LSTM states
        """

        B, N, _ = obs.shape

        # flatten for lstm cell
        obs_flat = obs.reshape(B*N, -1)
        state_flat = state.reshape(B*N, -1)

        # obs pathway
        obs_fc = F.relu(self.fc_obs(obs_flat))
        obs_h, obs_c = self.lstm_obs(obs_fc, (hidden['obs_h'], hidden['obs_c']))

        # state pathway
        state_fc = F.relu(self.fc_state(state_flat))
        state_h, state_c = self.lstm_state(state_fc, (hidden['state_h'], hidden['state_c']))

        # concat and reshape back
        node_feat = torch.cat([obs_h, state_h], dim=-1)
        node_feat = node_feat.reshape(B, N, -1)

        new_hidden = {
            'obs_h': obs_h, 'obs_c': obs_c, 
            'state_h': state_h, 'state_c': state_c
        }

        return node_feat, new_hidden
        
