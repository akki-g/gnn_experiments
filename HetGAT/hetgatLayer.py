import torch
from torch.nn import functional as F
import torch.nn as nn


class HetGATHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_out = d_out

        # node type weights: self transform + attention receiver side
        self.W_scout = nn.Linear(d_in, d_out, bias=False)
        self.W_interc = nn.Linear(d_in, d_out, bias=False)
        self.W_ssn = nn.Linear(d_in, d_out, bias=False)

        # edge-type weights: message transforms
        self.W_s2s = nn.Linear(d_in, d_out, bias=False)
        self.W_s2i = nn.Linear(d_in, d_out, bias=False)
        self.W_i2i = nn.Linear(d_in, d_out, bias=False)
        self.W_i2s = nn.Linear(d_in, d_out, bias=False)
        self.W_s2ssn = nn.Linear(d_in, d_out, bias=False)
        self.W_i2ssn = nn.Linear(d_in, d_out, bias=False)

        # attention vectors: one per edge type
        #shape (2*d_out) becaues they dot w [W_recv * h_j || W_edge *h_k]
        self.a_s2s = nn.Parameter(torch.empty(2*d_out))
        self.a_s2i = nn.Parameter(torch.empty(2*d_out))
        self.a_i2s = nn.Parameter(torch.empty(2*d_out))
        self.a_i2i = nn.Parameter(torch.empty(2*d_out))
        self.a_s2ssn = nn.Parameter(torch.empty(2*d_out))
        self.a_i2ssn = nn.Parameter(torch.empty(2*d_out))


        self.attn_dropout = 0.1
        self._init_weights()
    
    def _init_weights(self):
        """xavier uniform for weights for attention vectors"""

        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                bound = 1.0 / (2*self.d_out) ** 0.5
                # attention vectors scale like Xavier for fan_in = 2*d_out
                nn.init.uniform_(param, -bound, bound)
    
    def _attend(
            self,
            h_src: torch.Tensor, # (B, N_src, d_in) encoded node feature tensors from MLP
            h_dst: torch.Tensor, # (B, N_dst, d_in)
            W_edge: nn.Linear, # d_in -> d_out (transforms sender)
            W_node: nn.Linear, # d_in -> d_out (transforms receiver - same as self transform)
            a: nn.Parameter, # (2*d_out, )
            mask: torch.Tensor, # (B, N_src, N_dst) 1=connected
    ) -> torch.Tensor:
        """
        Compute attention-weighted aggregation for one edge type

        returns: (B, N_dst, d_out)

        changed to use gatv2 style attention: leaky relu before dot product w attention vecto 
        more expressive than v1 
        """

        src_t = W_edge(h_src)
        dst_t = W_node(h_dst)

        N_src = h_src.shape[1]
        N_dst = h_dst.shape[1]

        # pairwise sum of proj features
        pairwise = src_t.unsqueeze(2) + dst_t.unsqueeze(1)

        # gat v2: leaky relu before dotting w attention
        pairwise = F.leaky_relu(pairwise, negative_slope=0.2)

        #split attention vector and dot: a is (2*d_out,) but we only need first d_out elements
        a_vec = a[:self.d_out]
        e = (pairwise * a_vec).sum(dim=-1)

        # mask and softmax over sources
        e = e.masked_fill(mask==0, float('-inf'))
        alpha = F.softmax(e, dim=1)
        alpha - alpha.masked_fill(torch.isnan(alpha), 0.0)

        if self.training:
            alpha = F.dropout(alpha, p=self.attn_dropout, training=True)

        out = torch.einsum('bsd, bsf -> bdf', alpha, src_t)

        return out
        
    
    def forward(self, h_scout, h_interc, h_ssn, adj):
        """
        h_scout: (B, n_s, d_in) encoded node feats from MLP
        h_interc: (B, n_i, d_in)
        h_ssn: (B, 1, d_in) None during exec
        adj: dict of binary masks

        returns: (h_scout_new, h_interc_new, h_ssn_new) each (B, n, d_out)
        """

        # self transfrom computed once and reused
        self_s = self.W_scout(h_scout)
        self_i = self.W_interc(h_interc)

        # scout update (eq.5)
        agg_s2s = self._attend(
            h_scout, h_scout,
            self.W_s2s, self.W_scout, self.a_s2s,
            adj['scout_to_scout']
        )
        agg_i2s = self._attend(
            h_interc, h_scout,
            self.W_i2s, self.W_scout, self.a_i2s,
            adj['interc_to_scout']
        )
        h_scout_new = F.elu(self_s + agg_s2s + agg_i2s)

        # interc update (eq.7)
        agg_i2i = self._attend(
            h_interc, h_interc,
            self.W_i2i, self.W_interc,
            self.a_i2i,
            adj['interc_to_interc']
        )
        agg_s2i = self._attend(
            h_scout, h_interc,
            self.W_s2i, self.W_interc,
            self.a_s2i,
            adj['scout_to_interc']
        )
        h_interc_new = F.elu(self_i + agg_s2i + agg_i2i)

        ## ssn update - skip during exec
        h_ssn_new = None
        if h_ssn is not None:
            self_ssn = self.W_ssn(h_ssn)
            agg_s2ssn = self._attend(
                h_scout, h_ssn,
                self.W_s2ssn, self.W_ssn,
                self.a_s2ssn,
                adj['scout_to_ssn']
            )
            agg_i2ssn = self._attend(
                h_interc, h_ssn,
                self.W_i2ssn, self.W_ssn, self.a_i2ssn,
                adj['interc_to_ssn']
            )
            h_ssn_new = F.elu(self_ssn + agg_s2ssn + agg_i2ssn)


        return h_scout_new, h_interc_new, h_ssn_new


class HetGATLayer(nn.Module):
    """multi-head HetGAT layer. Wraps L independent HetGATHead modules"""
    def __init__(self, d_in: int, d_out: int, n_heads: int, concat_heads: bool = True):
        super().__init__()
        self.concat_heads = concat_heads
        self.heads = nn.ModuleList([
            HetGATHead(d_in=d_in, d_out=d_out) for _ in range(n_heads)
        ])

    def forward(self, h_scout, h_interc, h_ssn, adj):
        head_outs = [head(h_scout, h_interc, h_ssn, adj) for head in self.heads]

        hs_list = [o[0] for o in head_outs]
        hi_list = [o[1] for o in head_outs]

        if self.concat_heads:
            h_s = torch.cat(hs_list, dim=-1) # (B, n_s, K*d_put)
            h_i = torch.cat(hi_list, dim=-1) # (B, n_i, K*d_out)
        else:
            h_s = torch.stack(hs_list).mean(0) # (B, n_s, d_out)
            h_i = torch.stack(hi_list).mean(0) # (B, n_i, d_out)

        h_ssn_out = None
        if head_outs[0][2] is not None:
            hssn_list = [o[2] for o in head_outs]
            if self.concat_heads:
                h_ssn_out = torch.cat(hssn_list, dim=-1)
            else:
                h_ssn_out = torch.stack(hssn_list).mean(0)

        return h_s, h_i, h_ssn_out
            
        
