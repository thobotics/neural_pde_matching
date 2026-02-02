from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from pde_matching.data.preprocessing import (
    MeanStdAccumulator,
    Standardizer,
    TrajectoryStandardizer,
)
from pde_matching.data.state import StateKey
from pde_matching.data.trajectory import DataTrajectory
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data.data import BaseData, Data
from tqdm import tqdm


class Env(ABC):
    envs = {}

    @classmethod
    def register(cls, env_name):
        def decorator(env_cls):
            cls.envs[env_name] = env_cls
            return env_cls

        return decorator

    @classmethod
    def get_env(cls, env_name):
        return cls.envs.get(env_name)

    @property
    def input_keys(self) -> List[StateKey]:
        raise NotImplementedError

    @property
    def output_keys(self) -> List[StateKey]:
        raise NotImplementedError

    @property
    def visualizer(self):
        raise NotImplementedError

    @property
    def dt(self) -> float:
        return None

    @property
    def standardize_by_trajectory(self) -> bool:
        if hasattr(self, "_standardize_by_trajectory"):
            return self._standardize_by_trajectory
        else:
            return False

    @property
    def standardizer(self) -> Standardizer:
        if self._standardizer is None:
            # Note that currently the standardizer is supported only for offline datasets

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            data = np.load(self.stats_file)
            u_standardizer = Standardizer(
                torch.from_numpy(data["u_mean"]).to(device),
                torch.from_numpy(data["u_std"]).to(device),
            )

            self._standardizer = {
                StateKey.STATE: u_standardizer,
            }

            u_dot_standardizer = None
            u_dot_dot_standardizer = None

            if "u_dot_mean" in data:
                u_dot_standardizer = Standardizer(
                    torch.from_numpy(data["u_dot_mean"]).to(device),
                    torch.from_numpy(data["u_dot_std"]).to(device),
                )

            if "u_dot_dot_mean" in data:
                u_dot_dot_standardizer = Standardizer(
                    torch.from_numpy(data["u_dot_dot_mean"]).to(device),
                    torch.from_numpy(data["u_dot_dot_std"]).to(device),
                )

            if (
                self.standardize_by_trajectory
                and u_dot_standardizer is not None
                and u_dot_dot_standardizer is not None
            ):
                trajectory_u_dot_stats = []
                trajectory_u_dot_dot_stats = []

                for t in range(self.window_size - 1):
                    trajectory_u_dot_stats.append(MeanStdAccumulator())
                    trajectory_u_dot_dot_stats.append(MeanStdAccumulator())

                for i in tqdm(range(self.len()), desc="Trajectory Standardizing"):
                    data = self.get(i)
                    init_u = data[0].u
                    init_u_dot = data[0].u_dot

                    for t in range(self.window_size - 1):
                        trajectory_u_dot_stats[t].add(
                            (data[t + 1].u - init_u).cpu().numpy()
                        )
                        trajectory_u_dot_dot_stats[t].add(
                            (data[t + 1].u_dot - init_u_dot).cpu().numpy()
                        )

                trajectory_u_dot_mean = []
                trajectory_u_dot_std = []
                trajectory_u_dot_dot_mean = []
                trajectory_u_dot_dot_std = []
                for t in range(self.window_size - 1):
                    u_dot_mean, u_dot_std = trajectory_u_dot_stats[t].mean_and_std()
                    u_dot_dot_mean, u_dot_dot_std = trajectory_u_dot_dot_stats[
                        t
                    ].mean_and_std()
                    trajectory_u_dot_mean.append(u_dot_mean)
                    trajectory_u_dot_std.append(u_dot_std)
                    trajectory_u_dot_dot_mean.append(u_dot_dot_mean)
                    trajectory_u_dot_dot_std.append(u_dot_dot_std)

                u_dot_standardizer = TrajectoryStandardizer(
                    mean=u_dot_standardizer.mean,
                    std=u_dot_standardizer.std,
                    trajectory_mean=torch.from_numpy(
                        np.stack(trajectory_u_dot_mean, axis=0)
                    ).to(device),
                    trajectory_std=torch.from_numpy(
                        np.stack(trajectory_u_dot_std, axis=0)
                    ).to(device),
                )
                u_dot_dot_standardizer = TrajectoryStandardizer(
                    mean=u_dot_dot_standardizer.mean,
                    std=u_dot_dot_standardizer.std,
                    trajectory_mean=torch.from_numpy(
                        np.stack(trajectory_u_dot_dot_mean, axis=0)
                    ).to(device),
                    trajectory_std=torch.from_numpy(
                        np.stack(trajectory_u_dot_dot_std, axis=0)
                    ).to(device),
                )

            self._standardizer[StateKey.STATE_DOT] = u_dot_standardizer
            self._standardizer[StateKey.STATE_DOT_DOT] = u_dot_dot_standardizer

        return self._standardizer

    @abstractmethod
    def loss_mask(self, input: Tensor) -> Tensor:
        return None

    @abstractmethod
    def output_mask(self, input: Tensor) -> Tensor:
        return None
    
    def additional_metrics(self, predictions: Tensor, targets: DataTrajectory, mask: Tensor) -> dict:
        return {}

    def render(self, *args, **kwargs):
        return self.visualizer.visualize(*args, **kwargs)

    def _extract_pos_from_state_or_graph(self, state: torch.Tensor, graph: Data) -> torch.Tensor:
        # Note that position is the first 3 elements of the state vector
        return state[:, :3]

    def update_data(
        self,
        trajectory: DataTrajectory,
        state: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        is_train: bool = False,
    ) -> DataTrajectory:
        new_trajectory = trajectory.clone()

        if mask is not None:
            # Don't update the masked values
            if trajectory.is_output_a_state_pair():
                state, state_dot = trajectory.split(state)

                # Clone the state and state_dot to prevent in-place modification
                state = state.clone()
                state_dot = state_dot.clone()

                state[~mask] = trajectory.u[0][~mask]
                new_trajectory.state_list[0].u = state

                state_dot[~mask] = trajectory.u_dot[0][~mask]
                new_trajectory.state_list[0].u_dot = state_dot
            else:
                # Clone the state to prevent in-place modification
                state = state.clone()
                state[~mask] = trajectory.u[0][~mask]
                new_trajectory.state_list[0].u = state

                if output is not None:
                    state_dot = output.clone()
                    state_dot[~mask] = trajectory.u_dot[0][~mask]
                    new_trajectory.state_list[0].u_dot = state_dot

        # Update the mesh with new positions
        graph = trajectory.data_list[0]
        graph.pos = self._extract_pos_from_state_or_graph(state, graph)
        new_trajectory.data_list[0] = self.post_update_data(graph)

        return new_trajectory

    def post_update_data(
        self,
        data: BaseData,
    ) -> BaseData:
        if hasattr(self, "transform") and self.transform is not None:
            # Recompute the edge attributes
            del data.edge_attr
            data = self.transform(data)

        return data

    # Rollout the trajectory starting from the initial state of the trajectory
    # Note that the trajectory may contain the ground truth of the future states.
    def rollout(
        self,
        model,
        ground_truth: DataTrajectory,
        mask: torch.Tensor = None,
        is_train: bool = False,
    ):
        trajectory = ground_truth.clone()
        model.reset(
            trajectory.state_list[0].u.size(0), trajectory.state_list[0].u.device
        )

        predictions = []
        state = trajectory.state_vector[0]

        outputs = []
        for t in range(len(trajectory) - 1):
            norm_data = trajectory[t].normalize()
            output = model.one_step(
                norm_data.data_list[0],
                norm_data.input_vector[0],
                norm_data.state_vector[0],
                norm_data.u_properties[0],
            )
            output = norm_data.unnormalize(output)
            state = trajectory[t].integrate(state, output, self.dt)

            trajectory[t + 1] = self.update_data(
                trajectory[t + 1], state, output, mask, is_train
            )
            predictions.append(state)
            outputs.append(output)

        return torch.stack(predictions), torch.stack(outputs)

    def rollout_with_action(
        self,
        model,
        ground_truth: DataTrajectory,
        action: torch.Tensor,
        mask: torch.Tensor = None,
        is_train: bool = False,
    ):
        trajectory = ground_truth.clone()
        model.reset(
            trajectory.state_list[0].u.size(0), trajectory.state_list[0].u.device
        )

        # Apply action after resetting the model
        # This also prevent cloning a learnable action
        trajectory.state_list[0].u_dot = action

        predictions = []
        state = trajectory.state_vector[0]

        outputs = []
        for t in range(len(trajectory) - 1):
            # Setting inplace=True to let gradient flow through the normalization
            # since we don't want to make the state as a leaf node
            norm_data = trajectory[t].normalize(inplace=True)
            output = model.one_step(
                norm_data.data_list[0],
                norm_data.input_vector[0],
                norm_data.state_vector[0],
                norm_data.u_properties[0],
            )
            output = norm_data.unnormalize(output)
            state = state + output

            trajectory[t + 1] = self.update_data(
                trajectory[t + 1], state, mask, is_train
            )
            predictions.append(state)
            outputs.append(output)

        return torch.stack(predictions), torch.stack(outputs)
