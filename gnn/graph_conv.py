import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, F_in, G_out, K):
        super(GraphConv, self).__init__()
        self.F = F_in
        self.G = G_out
        self.K = K

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(F_in, G_out)) for _ in range(K)]
        )
        for p in self.weights:
            nn.init.xavier_uniform_(p)

    def forward(self, X, S):
        Z = X
        accum = X.new_zeros(*X.shape[:-1], self.G)
        for k in range(self.K):
            accum += torch.matmul(Z, self.weights[k])
            Z = torch.matmul(S, Z)
        return accum