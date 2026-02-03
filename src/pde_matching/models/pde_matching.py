from typing import Optional

import torch
from pde_matching.data.trajectory import DataTrajectory
from .model_builder import ModelBuilder
from .networks import GraphCONModel, IGNSModel


@ModelBuilder.register("PDEMatching")
class PDEMatching(torch.nn.Module):

    DYNAMICS_REGISTRY = {
        "graphcon": GraphCONModel,
        "igns": IGNSModel,
    }

    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        output_dim,
        dynamic_name: str,
        dynamic_config: dict = None,
        warmup_steps: int = 0,
        stop_gradient_at_step: int = 0,
        use_relative_timestep: bool = True,
        num_substeps: int = 1,
        loss_linear_weighting: bool = False,
        **ignored,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.warmup_steps = warmup_steps
        self.stop_gradient_at_step = stop_gradient_at_step
        self.use_relative_timestep = use_relative_timestep
        self.num_substeps = num_substeps
        self.loss_linear_weighting = loss_linear_weighting

        if dynamic_config is None:
            dynamic_config = {}

        dynamic_config["num_layers"] = dynamic_config["num_layers"] + warmup_steps

        dynamics_cls = self.DYNAMICS_REGISTRY.get(dynamic_name)
        if dynamics_cls is None:
            raise ValueError(
                f"Unknown network: {dynamic_name}. Available: {list(self.DYNAMICS_REGISTRY.keys())}"
            )

        self.dynamics = dynamics_cls(
            input_dim=input_dim_node,
            edge_dim=input_dim_edge,
            output_dim=output_dim,
            **dynamic_config,
        )
        self.dynamics = torch.compile(self.dynamics)

    def reset(self, batch_size: int = 1, device: torch.device = "cuda"):
        pass

    def forward(
        self,
        trajectory: DataTrajectory,
        mask: torch.Tensor = None,
        dt: float = 0.1,
        **ignored,
    ) -> torch.Tensor:
        self.reset(trajectory.batch_size, trajectory.device)

        predictions = []

        norm_data = trajectory[0].normalize()
        initial_offset = norm_data.state_vector[0]
        graph = norm_data.data_list[0]
        graph.x = norm_data.input_vector_perturbed[0]

        use_abs_timestep = hasattr(graph, "timestep") and not self.use_relative_timestep

        if use_abs_timestep:
            t_span = torch.stack(
                [graph.timestep + i for i in range(len(trajectory) - 1)]
            )
        else:
            t_span = torch.arange(len(trajectory) - 1 + self.warmup_steps)

        init_time = 0

        state, args = self.dynamics.prior_step(graph, t=0, t_span=t_span)
        for t in range(len(trajectory) - 1 + self.warmup_steps):

            if use_abs_timestep:
                timestep = (
                    t - self.warmup_steps if t >= self.warmup_steps else init_time
                )
            else:
                timestep = t

            num_steps = self.num_substeps if t >= self.warmup_steps else 1
            for _ in range(num_steps):
                state = self.dynamics.one_step(state, *args, t=timestep)

            if t >= self.warmup_steps:
                output_state = self.dynamics.output_step(state)
                predicted_state = initial_offset + output_state
                predictions.append(norm_data.unnormalize(predicted_state))

                if (
                    self.training
                    and self.stop_gradient_at_step > 0
                    and t > 0
                    and t % self.stop_gradient_at_step == 0
                ):
                    state = state.detach()

        return torch.stack(predictions)

    def weight_coefficient(self, targets: DataTrajectory):
        if not self.loss_linear_weighting:
            return 1.0

        T, _, D = targets.state_vector.shape
        time_scale = torch.linspace(
            1.0 / T, 1.0, steps=T, device=targets.state_vector.device
        )
        time_scale = time_scale.view(T, 1, 1).expand(T, 1, D)
        return time_scale

    def loss(
        self,
        predictions: torch.Tensor,
        targets: DataTrajectory,
        mask: Optional[torch.Tensor] = None,
    ):
        mse_sample_wise = (predictions - targets.state_vector) ** 2

        if mask is not None:
            if mask.ndim == 1:
                mse_sample_wise = mse_sample_wise[:, mask]
            else:
                mse_sample_wise[..., : mask.shape[1]] = torch.where(
                    mask,
                    mse_sample_wise[..., : mask.shape[1]],
                    torch.zeros_like(mse_sample_wise[..., : mask.shape[1]]),
                )

        loss = torch.mean(mse_sample_wise.sum(-1))

        outputs = {
            "loss": loss,
            "mse_sample_wise": mse_sample_wise,
        }

        if targets.is_output_a_state_pair():
            mse_pos, mse_vel = targets.split(mse_sample_wise)
            mse_pos = torch.mean(mse_pos.sum(-1))
            mse_vel = torch.mean(mse_vel.sum(-1))

            outputs["mse_pos"] = mse_pos
            outputs["mse_vel"] = mse_vel
        else:
            outputs["mse_pos"] = loss
            outputs["mse_vel"] = 0.0

        return outputs
