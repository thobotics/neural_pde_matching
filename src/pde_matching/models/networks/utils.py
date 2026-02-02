import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.typing import Optional
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def get_gcn_norm(edge_index, edge_weight: Optional[torch.Tensor] = None):
    """
    Normalize the adjacency matrix of a graph using GCN normalization.
    """
    if edge_weight is not None:
        edge_weight_distance = edge_weight[..., -1].unsqueeze(-1)
    else:
        edge_weight_distance = None

    edge_index, edge_weight_distance = gcn_norm(
        edge_index, edge_weight=edge_weight_distance, add_self_loops=False
    )

    if edge_weight is not None:
        edge_weight = torch.cat(
            [
                edge_weight[..., :-1],
                edge_weight_distance,
            ],
            dim=-1,
        )
    return edge_index, edge_weight


def time_embedding(t, d_model):
    """Standard positional embedding for scalar time t (batchable)."""
    position = t.unsqueeze(-1)  # [..., 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=t.dtype, device=t.device)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(t.shape + (d_model,), dtype=t.dtype, device=t.device)
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe


@torch.jit.script
def batch_dot_product(W, x, batch, X_pad, pad_pos):
    if len(W.shape) == 2 or W.shape[0] == 1:
        W = W if len(W.shape) == 2 else W[0]
        return x @ W

    X_pad = X_pad.clone()
    X_pad[batch, pad_pos] = x
    Yp = torch.bmm(X_pad, W)
    y = Yp[batch, pad_pos]
    return y


def pre_compute_padding_x_batch(x_shape, batch):
    N, H = x_shape
    B = batch.max().item() + 1
    counts = torch.bincount(batch, minlength=B)
    max_c = int(counts.max().item())
    starts = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
    pos = torch.arange(N, device=batch.device) - starts[batch]
    Xp = torch.zeros((B, max_c, H), device=batch.device)
    return Xp, pos


class Time2DynamicsMatrixLowRank(nn.Module):
    """
    Low-rank version of Time2DynamicsMatrix.

    A(t) = S(t) + R(t)
      S(t) = U_S diag(d_S(t)) U_S^T          (symmetric, rank <= r_s)
      R(t) = sum_k w_k(t) [ u_k v_k^T - v_k u_k^T ]   (skew, rank <= 2 r_r)
    with shared bases U_S, U_R, V_R (time-invariant nn.Parameters).

    The time MLP outputs exactly `hidden_dim` scalars:
      coeffs = [d_S (length r_s), w_R (length r_r)], where r_r = hidden_dim - r_s.
    """

    def __init__(
        self,
        hidden_dim: int,  # matrix size c
        rank: int = 64,  # symmetric rank
        weight_matrix_scale: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert 0 < rank <= hidden_dim, "rank must be in (0, hidden_dim]"
        self.hidden_dim = hidden_dim
        self.r_s = rank
        self.r_r = rank
        self.d_model = hidden_dim
        self.weight_matrix_scale = weight_matrix_scale
        self.eps = eps

        # Shared bases (time-invariant)
        self.U_S = nn.Parameter(torch.empty(hidden_dim, self.r_s, dtype=torch.float32))
        self.U_R = nn.Parameter(torch.empty(hidden_dim, self.r_r, dtype=torch.float32))
        self.V_R = nn.Parameter(torch.empty(hidden_dim, self.r_r, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim * 2, self.r_s + self.r_r),  # outputs d_S and w_R
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.U_S, a=math.sqrt(5))
        init.kaiming_uniform_(self.U_R, a=math.sqrt(5))
        init.kaiming_uniform_(self.V_R, a=math.sqrt(5))

    @staticmethod
    def _sym_from_US(U, d):  # S = U diag(d) U^T
        B = d.shape[0]
        U_exp = U.unsqueeze(0).expand(B, -1, -1)  # [B, c, r_s]
        Sd = U_exp * d.unsqueeze(1)  # [B, c, r_s]
        return Sd @ U_exp.transpose(1, 2)  # [B, c, c]

    @staticmethod
    def _skew_from_UV(U, V, w):  # sum_k w_k (u_k v_k^T - v_k u_k^T)
        B = w.shape[0]
        Ue = U.unsqueeze(0).expand(B, -1, -1)  # [B, c, r_r]
        Ve = V.unsqueeze(0).expand(B, -1, -1)  # [B, c, r_r]
        Uw = Ue * w.unsqueeze(1)  # [B, c, r_r]
        return (Uw @ Ve.transpose(1, 2)) - (Ve @ Uw.transpose(1, 2))  # [B, c, c]

    def forward(self, t: torch.Tensor, symmetric_only: bool = False) -> torch.Tensor:
        """
        t: scalar or [B]
        returns A(t): [B, hidden_dim, hidden_dim]
        """
        t_shape = t.shape
        if len(t.shape) > 1:
            t = t.view(-1)  # flatten to [B]

        emb = time_embedding(t, self.d_model)  # [B, d_model]
        coeffs = self.net(emb)  # [B, hidden_dim]
        d_S, w_R = coeffs.split([self.r_s, self.r_r], dim=-1)

        S = self._sym_from_US(self.U_S, d_S)  # [B, c, c]

        if not symmetric_only:
            R = self._skew_from_UV(self.U_R, self.V_R, w_R)  # [B, c, c]
            A = S + R
        else:
            A = S

        norm = A.norm(p="fro", dim=(1, 2), keepdim=True) + self.eps
        A_norm = self.weight_matrix_scale * A / norm

        if len(t_shape) > 1:
            A_norm = A_norm.view(*t_shape, self.hidden_dim, self.hidden_dim)

        return A_norm  # [B, c, c]


class Time2DynamicsMatrix(nn.Module):
    def __init__(self, hidden_dim=64, weight_matrix_scale=10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_matrix_scale = weight_matrix_scale
        self.sym_dim = hidden_dim * (hidden_dim + 1) // 2
        self.anti_dim = hidden_dim * (hidden_dim - 1) // 2
        self.total_dim = self.sym_dim + self.anti_dim

        self.d_model = hidden_dim

        # MLP
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_dim),
        )

    def upper_to_symmetric(self, vec):
        """Map upper triangle vector to symmetric matrix"""
        B = vec.size(0)
        S = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=vec.device)
        idx = torch.triu_indices(self.hidden_dim, self.hidden_dim)
        S[:, idx[0], idx[1]] = vec
        S[:, idx[1], idx[0]] = vec  # mirror
        return S

    def upper_to_antisymmetric(self, vec):
        """Map upper triangle vector to anti-symmetric matrix"""
        B = vec.size(0)
        R = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=vec.device)
        idx = torch.triu_indices(self.hidden_dim, self.hidden_dim, offset=1)
        R[:, idx[0], idx[1]] = vec
        R[:, idx[1], idx[0]] = -vec  # skew
        return R

    def forward(self, t):
        """
        Input: t ∈ ℝ or shape [B]
        Output: A ∈ ℝ^{B x c x c}
        """
        emb = time_embedding(t, self.d_model)  # [B, hidden_dim]
        out = self.net(emb)  # [B, total_dim]
        sym_vec = out[:, : self.sym_dim]
        anti_vec = out[:, self.sym_dim :]

        S = self.upper_to_symmetric(sym_vec)
        R = self.upper_to_antisymmetric(anti_vec)
        A = S + R  # A(t)
        norm = A.norm(p="fro", dim=(1, 2), keepdim=True) + 1e-6
        return self.weight_matrix_scale * A / norm  # [B, c, c]
