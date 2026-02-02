"""
Graph Couple Oscillator Networks (GraphCON) for displacement prediction.
Uses continuous-time dynamics with coupled oscillators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, init, Linear, Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from torch_geometric.typing import OptTensor, Tensor, Optional, Tuple

from .utils import get_gcn_norm


class GraphCONLayer(MessagePassing):
    """
    Graph Couple Oscillator Networks (GraphCON) layer.
    Implements continuous-time dynamics with position and velocity states.
    """

    def __init__(
        self,
        hidden_dim: int,
        act_fnc: nn.Module,
        dt: float = 0.1,
        alpha: float = 1.0,
        gamma: float = 1.0,
        aggr: str = "add",
    ):
        if aggr == "AttentionalAggregation":
            gate_nn = Sequential(Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
            aggr_kwargs = {"gate_nn": gate_nn}
        else:
            gate_nn = None
            aggr_kwargs = {}

        super().__init__(node_dim=0, aggr=aggr, aggr_kwargs=aggr_kwargs)
        self.gate_nn = gate_nn

        self.hidden_dim = hidden_dim
        self.dt = dt
        self.alpha = alpha  # Damping coefficient
        self.gamma = gamma  # Spring coefficient
        self.act_fnc = act_fnc

        # Linear transformations
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)
        self.lin_res = nn.Linear(hidden_dim, hidden_dim)

    def res_connection(self, x):
        """Residual connection similar to original GraphCON."""
        return self.lin_res(x) - self.lin(x)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: OptTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with position (x) and velocity (v) states.

        Args:
            x: Position state [N, hidden_dim]
            v: Velocity state [N, hidden_dim]
            edge_index: Graph edge indices [2, E]

        Returns:
            Updated (x, v) states
        """
        message = self.propagate(edge_index, x=self.lin(x), edge_weight=edge_weight)
        force = self.act_fnc(message + self.res_connection(x))
        # force = F.relu(message + self.res_connection(x))

        v_new = v + self.dt * (force - self.alpha * v - self.gamma * x)
        x_new = x + self.dt * v_new

        return x_new, v_new

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_weight: OptTensor
    ) -> torch.Tensor:
        mess = self.lin_msg(x_j - x_i)
        if edge_weight is None:
            return mess
        else:
            if edge_weight.shape[-1] == x_i.shape[-1]:
                return mess + edge_weight
            else:
                return mess + edge_weight.view(-1, 1)


class GraphCONModel(nn.Module):
    """
    Graph Couple Oscillator Networks (GraphCON) for displacement prediction.
    Uses continuous-time dynamics with coupled oscillators.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        edge_dim: int = 3,
        num_layers: int = 3,
        dt: float = 0.1,
        alpha: float = 1.0,
        gamma: float = 1.0,
        dropout: float = 0.1,
        layer_norm: bool = False,
        activation: str = "relu",
        mp_activation: str = "relu",
        use_gcn_norm: bool = False,
        shared_layers: bool = False,
        output_x_v: bool = False,
        aggr: str = "add",
        **ignored,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gcn_norm = use_gcn_norm
        self.output_x_v = output_x_v

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if mp_activation == "relu":
            self.mp_activation = nn.ReLU()
        elif mp_activation == "elu":
            self.mp_activation = nn.ELU()
        elif mp_activation == "tanh":
            self.mp_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown mp_activation: {mp_activation}")

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), self.activation, nn.Dropout(dropout)
        )

        # Input projection for edge features
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), self.activation, nn.Dropout(dropout)
        )

        # GraphCON layers
        if shared_layers:
            # Use the same layer for all layers
            con_layer = GraphCONLayer(
                hidden_dim, self.mp_activation, dt, alpha, gamma, aggr=aggr
            )
            self.con_layers = nn.ModuleList([con_layer for _ in range(num_layers)])
        else:
            self.con_layers = nn.ModuleList(
                [
                    GraphCONLayer(
                        hidden_dim, self.mp_activation, dt, alpha, gamma, aggr=aggr
                    )
                    for _ in range(num_layers)
                ]
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

        if output_x_v:
            self.output_proj_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim // 2),
            )
            self.output_proj_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim // 2),
            )
        else:
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def prior_step(
        self, data: Batch, t: int = 0, **ignored
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Project to hidden dimension
        x = self.input_proj(x)

        if self.use_gcn_norm:
            with torch.no_grad():
                edge_index, edge_weight = get_gcn_norm(
                    edge_index=edge_index, edge_weight=edge_weight
                )
        else:
            edge_weight = None

        edge_weight = self.edge_proj(edge_weight) if edge_weight is not None else None

        # Initialize velocity state
        v = torch.zeros_like(x)

        return (x, v), (edge_index, edge_weight)

    def one_step(
        self,
        xv: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        t: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, v = xv
        x, v = self.con_layers[t](x, v, edge_index, edge_weight=edge_weight)
        x = self.layer_norms[t](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return (x, v)

    def output_step(self, xv: Tuple[torch.Tensor]) -> torch.Tensor:
        x, v = xv
        if self.output_x_v:
            output_x = self.output_proj_x(x)
            output_v = self.output_proj_v(v)
            return torch.cat([output_x, output_v], dim=-1)
        return self.output_proj(x)

    def forward(self, data: Batch, intermediate_output=False) -> torch.Tensor:
        """Forward pass through GraphCON."""
        xv, edge_index, edge_weight = self.prior_step(data, t=0)

        if intermediate_output:
            outputs = []

        x, v = xv

        # Apply GraphCON layers
        for i in range(self.num_layers):
            x, v = self.one_step((x, v), edge_index, edge_weight=edge_weight, t=i)

            if intermediate_output:
                outputs.append(x)

        if intermediate_output:
            for j, output in enumerate(outputs):
                outputs[j] = self.output_step(output)
            return outputs
        else:
            return self.output_step(x)
