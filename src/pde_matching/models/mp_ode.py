import torch
import torch.nn as nn
from pde_matching.data.trajectory import DataTrajectory
from torchdiffeq import odeint

from .mgn import MeshGraphNet
from .model_builder import ModelBuilder
from .networks.mpnn import ProcessorLayer


@ModelBuilder.register("MessagePassingODE")
class MessagePassingODE(MeshGraphNet):
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        latent_dim,
        output_dim,
        num_messages,
        node_encoder_layers,
        edge_encoder_layers,
        node_decoder_layers,
        num_shooting,
        shared_processor: bool = False,
        autoregressive_integration: bool = False,
        integration_dt: float = 0.1,
        integration_method: str = "dopri5",
        **ignored,
    ):
        super(MeshGraphNet, self).__init__()

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_messages = num_messages
        self.shared_processor = shared_processor
        self.num_shooting = num_shooting
        self.autoregressive_integration = autoregressive_integration
        self.integration_dt = integration_dt
        self.integration_method = integration_method

        self.node_encoder = self._create_node_encoder(
            self.input_dim_node, self.hidden_dim, self.latent_dim, node_encoder_layers
        )
        self.edge_encoder = self._create_edge_encoder(
            self.input_dim_edge, self.hidden_dim, self.latent_dim, edge_encoder_layers
        )
        self.processor = self._create_processor(self.latent_dim, num_messages)
        self.decoder = self._create_decoder(
            self.latent_dim, self.hidden_dim, self.output_dim, node_decoder_layers
        )

    def _create_processor(self, latent_dim, num_messages=1):

        assert num_messages >= 1, "Number of message passing layers is not >=1"

        if self.shared_processor:
            processor = ProcessorLayer(latent_dim, latent_dim, update_edge=False)
        else:
            processor = nn.ModuleList()
            for _ in range(num_messages):
                processor.append(
                    ProcessorLayer(latent_dim, latent_dim, update_edge=False)
                )
        return processor

    def forward(
        self,
        trajectory: DataTrajectory,
        mask: torch.Tensor = None,
        dt: float = 0.1,
        **ignored,
    ) -> torch.Tensor:

        predictions = []
        norm_data = trajectory[0].normalize()
        initial_offset = norm_data.state_vector[0]

        node_attr = norm_data.input_vector_perturbed[0]
        edge_attr = norm_data.data_list[0].edge_attr

        latent = self.node_encoder(node_attr)
        edge_attr = self.edge_encoder(edge_attr)
        edge_index = norm_data.data_list[0].edge_index

        def ode_func(t, state):
            return self.one_step(
                latent=state,
                edge_attr=edge_attr,
                edge_index=edge_index,
                num_messages=self.num_messages,
            )

        t_eval = torch.linspace(
            0,
            self.integration_dt * (len(trajectory) - 1),
            len(trajectory),
            device=trajectory.device,
        )
        y0 = latent

        normalized_predictions = odeint(
            ode_func, y0, t_eval, method=self.integration_method
        )
        predictions = [
            initial_offset + norm_data.unnormalize(self.decoder(pred))
            for pred in normalized_predictions
        ]

        return torch.stack(predictions)

    def one_step(
        self,
        latent: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_messages: int = 0,
    ):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """

        if num_messages == 0:
            num_messages = self.num_messages

        if self.shared_processor:
            for _ in range(num_messages):
                latent, edge_attr = self.processor(latent, edge_index, edge_attr)
        else:
            for i in range(num_messages):
                latent, edge_attr = self.processor[i](latent, edge_index, edge_attr)

        return latent
