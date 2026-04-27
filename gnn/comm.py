from torch.distributions import Normal, Categorical
import torch
import torch.nn as nn

from gnn.graph_conv import GraphConv
from gnn.obs_encode import ObservationEncoder
from gnn.action import ActionHead

class CommPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, F_feat, G_feat, K,
                 discrete: bool = False, n_actions: int = 5):
        super(CommPolicy, self).__init__()
        self.discrete = discrete
        self.n_actions = n_actions
        out_dim = n_actions if discrete else action_dim
        self.obsEncoder = ObservationEncoder(obs_dim, hidden_dim, F_feat)
        self.graphConv = GraphConv(F_feat, G_feat, K)
        self.mean_head = ActionHead(G_feat, hidden_dim, out_dim)
        if not discrete:
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
        if self.discrete:
            logits = self.forward(obs, S)
            dist = Categorical(logits=logits)
            action = dist.sample()           # (...,) long
            log_prob = dist.log_prob(action)  # (...,) scalar
            entropy = dist.entropy()          # (...,) scalar
            return action, log_prob, entropy
        else:
            mean = self.forward(obs, S)
            std = self.log_std.clamp(min=-5.0, max=2.0).exp().expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action, log_prob, entropy

    def evaluate_actions(self, obs, S, actions):
        if self.discrete:
            logits = self.forward(obs, S)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), logits
        else:
            mean = self.forward(obs, S)
            std = self.log_std.clamp(min=-5.0, max=2.0).exp().expand_as(mean)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, mean
