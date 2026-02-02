import math
import torch
import torch_geometric.nn.inits as inits
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init, Linear, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor, Tensor, Optional
from .utils import (
    get_gcn_norm,
    time_embedding,
    batch_dot_product,
    pre_compute_padding_x_batch,
    Time2DynamicsMatrixLowRank,
)


class ExternalForcing(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        dtype: torch.dtype = torch.float32,
        activ_fun=None,
        aggr="add",
        **kwargs,
    ):
        if aggr == "AttentionalAggregation":
            gate_nn = Sequential(Linear(in_channels, in_channels), torch.nn.ReLU())
            aggr_kwargs = {"gate_nn": gate_nn}
        else:
            gate_nn = None
            aggr_kwargs = {}

        super().__init__(node_dim=0, aggr=aggr, aggr_kwargs=aggr_kwargs)
        self.gate_nn = gate_nn

        self.nf = in_channels
        self.dtype = dtype
        self.activation = getattr(torch, activ_fun)
        self.hidden_dim = in_channels
        self.external_force = nn.Linear(
            self.hidden_dim, in_channels, bias=False, dtype=self.dtype
        )
        self.lin_msg = nn.Linear(in_channels, in_channels, bias=False, dtype=self.dtype)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        if t is not None:
            t_embedding = time_embedding(
                torch.tensor(t, device=x.device, dtype=x.dtype), d_model=self.hidden_dim
            )
            state = x + t_embedding
        else:
            state = x
        external = self.external_force(state)
        external = self.activation(
            self.propagate(edge_index, x=external, edge_weight=edge_weight)
        )
        return external

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        mess = self.lin_msg(x_j - x_i)
        if edge_weight is None:
            return mess
        else:
            if edge_weight.shape[-1] == x_i.shape[-1]:
                return mess + edge_weight
            else:
                return mess + edge_weight.view(-1, 1)

    def reset_parameters(self):
        inits.glorot(self.external_force)


class EnergyMLP(Module):
    def __init__(
        self,
        in_channels: int,
        bias: bool = True,
        activ_fun: str = "tanh",
        dtype: torch.dtype = torch.float32,
        weight_matrix_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.bias = Parameter(torch.empty(in_channels, dtype=dtype)) if bias else None
        self.activation = getattr(torch, activ_fun)
        self.dtype = dtype
        self.W_func = Time2DynamicsMatrixLowRank(
            hidden_dim=in_channels, weight_matrix_scale=weight_matrix_scale
        )

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        x_conv: torch.Tensor,
        batch: Optional[torch.Tensor],
        W_batch: Optional[torch.Tensor],
        X_pad: Optional[torch.Tensor],
        pad_pos: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        if W_batch is None:
            W_batch = self.W_func(t)  # [B, c, c]
        else:
            W_batch = W_batch[t]
        x_non_linear = batch_dot_product(W_batch, x, batch, X_pad, pad_pos) + x_conv

        if self.bias is not None:
            x_non_linear = x_non_linear + self.bias
        x_non_linear = self.activation(x_non_linear)

        W_batch = W_batch.transpose(-1, -2)
        x = batch_dot_product(W_batch, x_non_linear, batch, X_pad, pad_pos)
        return x, x_non_linear

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)


class GraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        dtype: torch.dtype = torch.float32,
        aggr="add",
        weight_matrix_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(aggr=aggr)
        self.transpose_weight = False
        self.dtype = dtype
        self.V_msg = Parameter(torch.empty((in_channels, in_channels), dtype=dtype))
        self.V_func = Time2DynamicsMatrixLowRank(
            hidden_dim=in_channels, weight_matrix_scale=weight_matrix_scale
        )

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        V_batch: Optional[torch.Tensor] = None,
        X_pad: Optional[torch.Tensor] = None,
        pad_pos: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if V_batch is None:
            V_batch = self.V_func(t)
        else:
            V_batch = V_batch[t]
        x_transformed = batch_dot_product(V_batch, x, batch, X_pad, pad_pos)

        out = self.propagate(
            edge_index=edge_index, x=x_transformed, edge_weight=edge_weight
        )
        return out, V_batch, x_transformed

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j

    def reset_parameters(self):
        init.kaiming_uniform_(self.V_msg, a=math.sqrt(5))


class Second_GraphConv(MessagePassing):
    def __init__(self, conv, aggr="add"):
        super().__init__(aggr)

    def forward(
        self,
        x: torch.Tensor,
        x_transformed: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        X_pad: Optional[torch.Tensor] = None,
        pad_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        V = V.transpose(-1, -2)
        x = batch_dot_product(V, x, batch, X_pad, pad_pos)

        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j


class HamiltonianGradient(Module):
    def __init__(
        self,
        in_channels: int,
        bias: bool = True,
        activ_fun: str = "tanh",
        dtype: torch.dtype = torch.float32,
        aggr: str = "add",
        weight_matrix_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.conv = GraphConv(
            in_channels, dtype, aggr, weight_matrix_scale=weight_matrix_scale, **kwargs
        )
        self.second_conv = Second_GraphConv(self.conv, aggr, **kwargs)
        self.mlp = EnergyMLP(
            in_channels,
            bias,
            activ_fun,
            dtype,
            weight_matrix_scale=weight_matrix_scale,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        W_batch: Optional[torch.Tensor] = None,
        V_batch: Optional[torch.Tensor] = None,
        X_pad: Optional[torch.Tensor] = None,
        pad_pos: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_conv, V, x_transformed = self.conv(
            x, edge_index, edge_weight, batch, V_batch, X_pad, pad_pos, t=t
        )
        x_first, x_non_linear = self.mlp(x, x_conv, batch, W_batch, X_pad, pad_pos, t=t)
        x_second = self.second_conv(
            x_non_linear,
            x_transformed,
            edge_index,
            edge_weight,
            batch,
            V=V,
            X_pad=X_pad,
            pad_pos=pad_pos,
        )
        return x_first + x_second


class HamiltonianConv(Module):
    def __init__(
        self,
        in_channels: int,
        epsilon: float = 0.1,
        activ_fun: str = "tanh",
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        aggr: str = "add",
        weight_matrix_scale: float = 1.0,
        damping_factor: float = 1.0,
        external_force_factor: float = 0.0,
        using_symplectic_integrator: bool = False,
        ext_force_kwargs={},
        **kwargs,
    ) -> None:

        super().__init__()

        self.gradient_p = HamiltonianGradient(
            in_channels // 2,
            bias,
            activ_fun,
            dtype,
            aggr=aggr,
            weight_matrix_scale=weight_matrix_scale,
        )
        self.gradient_q = HamiltonianGradient(
            in_channels // 2,
            bias,
            activ_fun,
            dtype,
            aggr=aggr,
            weight_matrix_scale=weight_matrix_scale,
        )
        self.dampening = Parameter(torch.zeros(1, in_channels // 2, dtype=dtype))
        self.bias = bias
        self.dampening_factor = damping_factor
        self.external_force_factor = external_force_factor
        self.activ_fun = activ_fun
        self.external = ExternalForcing(
            in_channels // 2, dtype, activ_fun, **ext_force_kwargs
        )
        self.using_symplectic_integrator = using_symplectic_integrator

        self.nf = in_channels
        self.epsilon = torch.tensor(epsilon, dtype=dtype)

        torch.nn.init.kaiming_normal_(self.dampening)

    def pre_compute_matrix(
        self, x, t_span: torch.Tensor, batch: Optional[torch.Tensor] = None
    ):
        p_W_batch = self.gradient_p.mlp.W_func(t_span)  # [B, c, c]
        q_W_batch = self.gradient_q.mlp.W_func(t_span)

        p_V_batch = self.gradient_p.conv.V_func(t_span)  # [B, c, c]
        q_V_batch = self.gradient_q.conv.V_func(t_span)

        # p_V_batch = self.gradient_p.conv.V_func(t_span, symmetric_only=True)  # [B, c, c]
        # q_V_batch = self.gradient_q.conv.V_func(t_span, symmetric_only=True)  # [B, c, c]

        x_shape = (x.shape[0], self.nf // 2)
        X_pad, pad_pos = pre_compute_padding_x_batch(x_shape, batch)

        return p_W_batch, q_W_batch, p_V_batch, q_V_batch, X_pad, pad_pos

    def forward(
        self, t, x, edge_index, edge_weight=None, batch=None, pre_computed_matrix=None
    ) -> Tensor:
        p_n = x[..., : self.nf // 2]
        q_n = x[..., self.nf // 2 :]

        p_W_batch, q_W_batch, p_V_batch, q_V_batch, X_pad, pad_pos = pre_computed_matrix

        if self.using_symplectic_integrator:
            # Integrate position using updated momentum
            grad_p = self.gradient_p(
                p_n,
                edge_index,
                edge_weight,
                batch,
                p_W_batch,
                p_V_batch,
                X_pad,
                pad_pos,
                t=t,
            )
            q_n = q_n + self.epsilon * grad_p

            grad_q = self.gradient_q(
                q_n,
                edge_index,
                edge_weight,
                batch,
                q_W_batch,
                q_V_batch,
                X_pad,
                pad_pos,
                t=t,
            )
            p_n = p_n - self.epsilon * grad_q
        else:
            grad_p = self.gradient_p(
                p_n,
                edge_index,
                edge_weight,
                batch,
                p_W_batch,
                p_V_batch,
                X_pad,
                pad_pos,
                t=t,
            )
            grad_q = self.gradient_q(
                q_n,
                edge_index,
                edge_weight,
                batch,
                q_W_batch,
                q_V_batch,
                X_pad,
                pad_pos,
                t=t,
            )

            # Integrate momentum and position using current gradients
            q_n = q_n + self.epsilon * grad_p
            p_n = p_n - self.epsilon * grad_q

        p_n = p_n - self.epsilon * self.dampening_factor * self.dampening * grad_p
        p_n = p_n + self.epsilon * self.external_force_factor * self.external(
            q_n, edge_index, edge_weight, t=t
        )

        return torch.cat([p_n, q_n], dim=-1)


class IGNS(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        edge_dim: int = 3,
        num_layers: int = 3,
        num_enc_layers: int = 1,
        num_dec_layers: int = 1,
        dropout: float = 0.1,
        aggr: str = "mean",
        activation: str = "relu",
        mp_activation: str = "tanh",
        layer_norm: bool = False,
        shared_layers: bool = True,
        epsilon: float = 0.1,
        bias: bool = False,
        doubled_dim: bool = True,
        final_state: str = "pq",
        weight_matrix_scale: float = 1.0,
        damping_factor: float = 1.0,
        external_force_factor: float = 0.0,
        using_symplectic_integrator: bool = False,
        ext_force_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.doubled_dim = doubled_dim
        self.final_state = final_state
        self.shared_layers = shared_layers

        if self.doubled_dim:
            hidden_dim = hidden_dim * 2
        self.hidden_dim = hidden_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input projection
        self.input_proj = nn.Sequential(
            *(
                [nn.Linear(input_dim, hidden_dim), self.activation]
                + [nn.Linear(hidden_dim, hidden_dim), self.activation]
                * (num_enc_layers - 1)
            )
        )

        # Input projection for edge features
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2), self.activation, nn.Dropout(dropout)
        )

        # Graph convolution layers
        conv_kwargs = {
            "in_channels": hidden_dim,
            "epsilon": epsilon,
            "bias": bias,
            "aggr": aggr,
            "activ_fun": mp_activation,
            "weight_matrix_scale": weight_matrix_scale,
            "damping_factor": damping_factor,
            "external_force_factor": external_force_factor,
            "using_symplectic_integrator": using_symplectic_integrator,
            "ext_force_kwargs": ext_force_kwargs,
        }
        if shared_layers:
            # Use the same layer for all layers
            layer = HamiltonianConv(**conv_kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList(
                [HamiltonianConv(**conv_kwargs) for _ in range(num_layers)]
            )

        # Layer normalization
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.layer_norms.append(
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        self.activation,
                    )
                )
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Output projection
        output_hidden_dim = hidden_dim if self.final_state == "pq" else hidden_dim // 2

        self.output_proj = nn.Sequential(
            nn.Linear(output_hidden_dim, output_hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim, output_dim),
        )
        self.output_proj_p = nn.Sequential(
            nn.Linear(output_hidden_dim // 2, output_hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim, output_dim // 2),
        )
        self.output_proj_q = nn.Sequential(
            nn.Linear(output_hidden_dim // 2, output_hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(output_hidden_dim, output_dim // 2),
        )

    def prior_step(self, data: Batch, t: int, t_span: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Project to hidden dimension
        x = self.input_proj(x)
        with torch.no_grad():
            edge_index, edge_weight = get_gcn_norm(
                edge_index=edge_index, edge_weight=edge_weight
            )

        edge_weight = self.edge_proj(edge_weight) if edge_weight is not None else None

        pre_computed_matrix = self.layers[0].pre_compute_matrix(
            x, t_span.to(x.device).to(x.dtype), batch
        )

        return x, (edge_index, edge_weight, batch, pre_computed_matrix)

    def one_step(
        self, x, edge_index, edge_weight, batch, pre_computed_matrix, t: int
    ) -> torch.Tensor:
        layer = self.layers[0] if self.shared_layers else self.layers[t]
        x_new = layer(
            t=t,
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            pre_computed_matrix=pre_computed_matrix,
        )
        x_new = F.dropout(x_new, p=self.dropout, training=self.training)

        return x_new

    def output_step(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_state == "p":
            p = x[..., : self.hidden_dim // 2]
            return self.output_proj(p)
        elif self.final_state == "q":
            q = x[..., self.hidden_dim // 2 :]
            return self.output_proj(q)
        else:
            p = x[..., : self.hidden_dim // 2]
            q = x[..., self.hidden_dim // 2 :]
            output_p = self.output_proj_p(p)
            output_q = self.output_proj_q(q)
            return torch.cat([output_q, output_p], dim=-1)

    def forward(self, data: Batch, intermediate_output=False) -> torch.Tensor:
        x, edge_index, edge_weight, batch = self.prior_step(data, t=0)

        if intermediate_output:
            outputs = []

        # Initial layers with residual connections
        for i in range(self.num_layers):
            x = self.one_step(x, edge_index, edge_weight, batch, t=i)

            if intermediate_output:
                outputs.append(x)

        if intermediate_output:
            for j, output in enumerate(outputs):
                outputs[j] = self.output_step(output)
            return outputs
        else:
            return self.output_step(x)
