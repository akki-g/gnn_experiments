import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, F_in, G_out, K):
        super(GraphConv, self).__init__()
        self.F = F_in
        self.G = G_out
        self.K = K

        self.weight = nn.Parameter(torch.empty(K, F_in, G_out))
        nn.init.xavier_uniform_(self.weight.reshape(K*F_in, G_out))
        self.weight.data = self.weight.data.reshape(K, F_in, G_out)
        

    def forward(self, X, S):
        Z = X
        Z_hops = [Z]
        for k in range(1, self.K):
            Z = torch.matmul(S, Z)
            Z_hops.append(Z)

        Z_stack = torch.stack(Z_hops, dim=0)
        accum = torch.einsum(
            'k...nf,kfg->...ng', Z_stack, self.weight
        )
        return accum
        