from torch.distributions import Normal
import torch
import torch.nn as nn

from gnn.graph_conv import GraphConv
from gnn.obs_encode import ObservationEncoder
from gnn.action import ActionHead

class CommPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, F_feat, G_feat, K):
        super(CommPolicy, self).__init__()
        self.obsEncoder = ObservationEncoder(obs_dim, hidden_dim, F_feat)
        self.graphConv = GraphConv(F_feat, G_feat, K)
        self.mean_head = ActionHead(G_feat, hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable exploration

    def forward(self, obs, S):
        device = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device=device, dtype=torch.float32)
        if not torch.is_tensor(S):
            S = torch.as_tensor(S, dtype=torch.float32, device=device)
        else:
            S = S.to(device=device, dtype=torch.float32)

        obs_encode = self.obsEncoder(obs)
        agg_feats = self.graphConv(obs_encode, S)
        mean = self.mean_head(agg_feats)
        return mean

    def get_actions(self, obs, S):
        mean = self.forward(obs, S)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, obs, S, actions):
        mean = self.forward(obs, S)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean
