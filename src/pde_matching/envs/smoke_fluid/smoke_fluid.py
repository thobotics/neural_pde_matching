import enum
import os
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pde_matching.data.preprocessing import MeanStdAccumulator
from pde_matching.data.state import StateKey
from pde_matching.data.transforms import EdgeCategorical
from torch import Tensor
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import Cartesian, Compose, Distance
from tqdm import tqdm
from pde_matching.utils.common_utils import noise_like

from ..env import Env
from ..trajectory_dataset import TrajectoryDatasetMixin
from ..utils import EdgeFeatureScaler


class NodeType(enum.IntEnum):
    FLUID = 0
    BOUNDARY = 1
    SIZE = 2


class EdgeType(enum.IntEnum):
    FLUID_FLUID = 0
    FLUID_BOUNDARY = 1
    BOUNDARY_FLUID = 2
    BOUNDARY_BOUNDARY = 3
    SIZE = 4


@Env.register("SmokeFluidEnv")
class SmokeFluidEnv(TrajectoryDatasetMixin, Env):
    def __init__(
        self,
        root: str,
        window_size: int,
        window_shift: int,
        n_sequences: int = None,
        stage: str = "train",
        log: bool = True,
        input_keys: List[str] = None,
        output_keys: List[str] = None,
        training_noise: bool = False,
        training_noise_std: float = 0.0,
        standardize_by_trajectory: bool = False,
        **kwargs,
    ):
        transform = Compose(
            [
                EdgeCategorical(EdgeType.SIZE),
                Cartesian(norm=False),
                Distance(norm=False),
                EdgeFeatureScaler(
                    start_idx=EdgeType.SIZE, end_idx=EdgeType.SIZE + 3, scale_factor=1.0 / self.max_density
                ),
            ]
        )

        self.cache = {"idx": -1}
        self.stage = stage
        self.training_noise = training_noise
        self.training_noise_std = training_noise_std
        self._standardize_by_trajectory = standardize_by_trajectory
        self._n_sequences = n_sequences
        self._n_total_sequences = None
        self._standardizer = None
        self._visualizer = None
        self._input_keys = input_keys
        self._output_keys = output_keys
        self._visualizer_kwargs = kwargs.pop("visualizer", {})

        # Data split: 16 batches for train, 2 for val, 2 for test
        data_size = self.data_size
        self._train_size = data_size[0]
        self._val_size = data_size[1]
        self._test_size = data_size[2]

        super().__init__(
            root=root,
            window_size=window_size,
            window_shift=window_shift,
            transform=transform,
            log=log,
            **kwargs,
        )

    @property
    def raw_dirs_range_tuple(self) -> List[Tuple[int, int]]:
        # Ranges refer to batch file indices
        train_range = (0, self._train_size)
        val_range = (train_range[1], train_range[1] + self._val_size)
        test_range = (val_range[1], val_range[1] + self._test_size)
        return [train_range, val_range, test_range]

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return ["train/processed", "val/processed", "test/processed"]

    @property
    def stats_file(self) -> str:
        return Path(self.processed_dir) / "stats.npz"

    @property
    def trajectory_length(self) -> int:
        return 50
        # return 31

    @property
    def dt(self) -> float:
        # return 0.1
        return 1.0
    
    @property
    def sub_sampling_factor(self) -> int:
        return 4
    
    @property
    def max_density(self) -> float:
        return 1.0

    @property
    def node_type_dim(self) -> int:
        return NodeType.SIZE
    
    @property
    def data_size(self) -> List[int]:
        return [16, 2, 2]

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return None

    @property
    def n_total_sequences(self) -> int:
        if self._n_total_sequences is None:
            if self.stage == "train":
                self._n_total_sequences = self._train_size * 50  # Assuming 50 trajectories per batch
            elif self.stage == "val":
                self._n_total_sequences = self._val_size * 50
            elif self.stage == "test":
                self._n_total_sequences = self._test_size * 50

        return self._n_total_sequences

    @property
    def n_sequences(self) -> int:
        if self._n_sequences is None:
            return self.n_total_sequences
        else:
            return min(self._n_sequences, self.n_total_sequences)

    @property
    def input_keys(self) -> List[StateKey]:
        if self._input_keys is None:
            return [StateKey.PROPERTIES]
        else:
            return [StateKey[key] for key in self._input_keys]

    @property
    def output_keys(self) -> List[StateKey]:
        if self._output_keys is None:
            return [StateKey.STATE]
        else:
            return [StateKey[key] for key in self._output_keys]

    @property
    def visualizer(self):
        if self._visualizer is None:
            from .smoke_fluid_visualizer import SmokeFluidVisualizer
            self._visualizer = SmokeFluidVisualizer(self.grid_shape, self.window_size, **self._visualizer_kwargs)
        return self._visualizer
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (48, 48)

    def _compute_sdf_features(self, node_coords, boundary_nodes_set, nodes):
        """Compute SDF features directly on graph nodes."""
        n_nodes = len(nodes)
        sdf_values = np.zeros(n_nodes)
        sdf_dx = np.zeros(n_nodes)
        sdf_dy = np.zeros(n_nodes)
        
        # Get boundary node coordinates
        boundary_coords = []
        for i, (node_id, _) in enumerate(nodes):
            if node_id in boundary_nodes_set:
                boundary_coords.append(node_coords[i])
        boundary_coords = np.array(boundary_coords)
        
        # For each node, compute distance to nearest boundary
        for i, coord in enumerate(node_coords):
            node_id = nodes[i][0]
            
            if node_id in boundary_nodes_set:
                # Boundary nodes have zero distance
                sdf_values[i] = 0.0
                sdf_dx[i] = 0.0
                sdf_dy[i] = 0.0
            else:
                # Find nearest boundary node
                distances = np.sqrt(np.sum((boundary_coords - coord)**2, axis=1))
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                nearest_boundary = boundary_coords[min_idx]
                
                # Signed distance (positive for fluid nodes)
                sdf_values[i] = min_dist
                
                # Gradient points away from nearest boundary
                if min_dist > 0:
                    sdf_dx[i] = (coord[0] - nearest_boundary[0]) / min_dist
                    sdf_dy[i] = (coord[1] - nearest_boundary[1]) / min_dist
        
        return sdf_values, sdf_dx, sdf_dy

    def loss_mask(self, input: Tensor) -> Tensor:
        """Only predict fluid nodes, not boundary nodes."""
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.FLUID
        return mask

    def output_mask(self, input: Tensor) -> Tensor:
        """Only predict fluid nodes, not boundary nodes."""
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.FLUID
        return mask

    def control_params_mask(self, input: Tensor) -> Tensor:
        """No control parameters in this setup."""
        if input.ndim == 3:
            input = input[0]
        return torch.zeros(input.shape[0], dtype=torch.bool)

    def process(self):
        for stage in ["train", "val", "test"]:
            self._process_by_stage(stage)

    def _process_by_stage(self, stage):
        stage_dir = Path(self.processed_dir) / stage
        stage_dir.mkdir(exist_ok=True)

        u_stats = MeanStdAccumulator()
        u_dot_stats = MeanStdAccumulator()
        u_dot_dot_stats = MeanStdAccumulator()

        # Find batch files
        batch_files = list(Path(self.raw_dir).glob("*_batch_*_graph.pkl"))
        batch_files.sort(key=lambda x: int(x.stem.split("_batch_")[1].split("_")[0]))
        
        # Select subset based on stage
        if stage == "train":
            start, end = self.raw_dirs_range_tuple[0]
        elif stage == "val":
            start, end = self.raw_dirs_range_tuple[1]
        elif stage == "test":
            start, end = self.raw_dirs_range_tuple[2]
        
        stage_batch_files = batch_files[start:end]

        trajectory_counter = 0
        
        for batch_idx, batch_file in enumerate(
            tqdm(stage_batch_files, desc=f"Processing {stage}", unit=" batch files")
        ):
            with open(batch_file, "rb") as f:
                batch_data = pkl.load(f)

            for traj_data in batch_data:
                env_data_list = []
                
                # Extract graph structure (static for all timesteps)
                nodes = traj_data['nodes']
                edges = traj_data['edges']
                boundary_nodes = set(traj_data['boundary_nodes'])  # Get boundary node IDs
                density_data = np.array(traj_data['density_data'])  # (T, N)
                velocity_data = np.array(traj_data['velocity_data'])  # (T, N, 2)
                
                # Store grid information for visualization
                mask = np.array(traj_data['mask'])  # (H, W) - True for valid nodes
                grid_shape = traj_data['grid_shape']  # (H, W)
                
                n_nodes = len(nodes)
                n_timesteps = density_data.shape[0]
                
                # Create node coordinates and types
                node_coords = np.array([[node_data['x'], node_data['y']] for _, node_data in nodes])
                node_types = np.zeros((n_nodes, NodeType.SIZE))
                
                # Set node types based on boundary information
                for i, (node_id, _) in enumerate(nodes):
                    if node_id in boundary_nodes:
                        node_types[i, NodeType.BOUNDARY] = 1
                    else:
                        node_types[i, NodeType.FLUID] = 1
                
                # Compute SDF features
                sdf_values, sdf_dx, sdf_dy = self._compute_sdf_features(node_coords, boundary_nodes, nodes)
                
                # Combine node types with SDF features
                # node_properties = np.column_stack([
                #     node_types,
                #     sdf_values.reshape(-1, 1),
                #     sdf_dx.reshape(-1, 1), 
                #     sdf_dy.reshape(-1, 1)
                # ])
                node_properties = node_types
                # node_properties = np.column_stack([
                #     node_types,
                #     node_coords,
                # ])
                
                # Create edge index (make bidirectional)
                edge_array = np.array(edges)
                edge_index = torch.from_numpy(edge_array.T).long()
                edge_index = to_undirected(edge_index)  # Make bidirectional
                
                # Determine edge types based on connected node types
                edge_types = []
                for i in range(edge_index.shape[1]):
                    src_node = edge_index[0, i].item()
                    dst_node = edge_index[1, i].item()
                    src_is_boundary = node_types[src_node, NodeType.BOUNDARY] == 1
                    dst_is_boundary = node_types[dst_node, NodeType.BOUNDARY] == 1
                    src_is_fluid = node_types[src_node, NodeType.FLUID] == 1
                    dst_is_fluid = node_types[dst_node, NodeType.FLUID] == 1
                    
                    if src_is_boundary and dst_is_boundary:
                        edge_types.append(EdgeType.BOUNDARY_BOUNDARY)
                    elif src_is_fluid and dst_is_boundary:
                        edge_types.append(EdgeType.FLUID_BOUNDARY)
                    elif src_is_boundary and dst_is_fluid:
                        edge_types.append(EdgeType.BOUNDARY_FLUID)
                    else:
                        edge_types.append(EdgeType.FLUID_FLUID)
                
                edge_types = torch.tensor(edge_types, dtype=torch.long)

                if batch_idx == 0 and trajectory_counter == 0:
                    print(f"Number of nodes: {n_nodes}, timesteps: {n_timesteps}")
                    print("Node types distribution:", np.bincount(np.argmax(node_types, axis=1)))

                    print(f"Total edges: {edge_index.shape[1]}")
                    print(f"Edge types distribution: {torch.bincount(edge_types)}")
                
                for t in range(0, n_timesteps, self.sub_sampling_factor):  # Subsample every 4th timestep
                    # Current state: density + velocity (coordinates are static)
                    density = density_data[t] / self.max_density # (N,)
                    # velocity = velocity_data[t] / 400.0 # (N, 2)
                    
                    # State vector: only density + velocity (coordinates are static)
                    # state = np.column_stack([density.reshape(-1, 1), velocity])  # (N, 3)
                    state = density.reshape(-1, 1)  # (N, 1) - only density for prediction
                    
                    # Create graph data
                    graph_data = Data(
                        pos=torch.from_numpy(node_coords).float(),  # 2D coordinates (x, y) only
                        u=torch.from_numpy(state).float(),  # State (density, vx, vy) - only what we predict
                        properties=torch.from_numpy(node_properties).float(),
                        edge_index=edge_index.clone(),
                        edge_type=edge_types.clone(),
                        grid_mask=torch.from_numpy(mask).bool(),
                        grid_shape=grid_shape,
                        timestep=t,
                    )
                    
                    env_data_list.append(graph_data)
                    
                    # Update statistics
                    u_stats.add(graph_data.u.detach().cpu().numpy())

                # Compute u_dot (velocities from consecutive u states)
                for t, data in enumerate(env_data_list):
                    if t == len(env_data_list) - 1:
                        # For the last timestep, copy velocity from previous timestep
                        data.u_dot = env_data_list[t - 1].u_dot
                    else:
                        data.u_dot = (env_data_list[t + 1].u - data.u) / self.dt
                    u_dot_stats.add(data.u_dot.detach().cpu().numpy())

                # Compute u_dot_dot (accelerations from consecutive u_dot states)
                for t, data in enumerate(env_data_list):
                    if t == len(env_data_list) - 1:
                        data.u_dot_dot = env_data_list[t - 1].u_dot_dot
                    else:
                        data.u_dot_dot = (env_data_list[t + 1].u_dot - data.u_dot) / self.dt
                    u_dot_dot_stats.add(data.u_dot_dot.detach().cpu().numpy())
                
                # Save each trajectory as a separate .pt file
                torch.save(env_data_list, Path(stage_dir) / f"{trajectory_counter}.pt")
                trajectory_counter += 1

        # Save processed marker
        processed_file = Path(stage_dir) / "processed"
        processed_file.touch()

        # Save statistics for training stage
        if stage == "train":
            u_mean, u_std = u_stats.mean_and_std()
            u_dot_mean, u_dot_std = u_dot_stats.mean_and_std()
            u_dot_dot_mean, u_dot_dot_std = u_dot_dot_stats.mean_and_std()
            
            np.savez(
                self.stats_file,
                u_mean=u_mean,
                u_std=u_std,
                u_dot_mean=u_dot_mean,
                u_dot_std=u_dot_std,
                u_dot_dot_mean=u_dot_dot_mean,
                u_dot_dot_std=u_dot_dot_std,
            )

    def _extract_pos_from_state_or_graph(self, state: torch.Tensor, graph: Data) -> torch.Tensor:
        # For static graph, no change in position
        return graph.pos

    def post_update_data(self, data):
        # For static graph, no change in position
        return data
    
    def _add_noise_to_data(self, data_list):
        for i in range(len(data_list)):
            u_noise = noise_like(data_list[i].u, self.training_noise_std)
            u_dot_noise = noise_like(data_list[i].u_dot, self.training_noise_std)
            u_dot_dot_noise = noise_like(
                data_list[i].u_dot_dot, self.training_noise_std
            )

            # Noise is added to the position here before the transform
            # Note that the pos is always before the other physical quantities
            data_list[i].u_noise = u_noise
            data_list[i].u_dot_noise = u_dot_noise
            data_list[i].u_dot_dot_noise = u_dot_dot_noise


@Env.register("WaveEnv")
class WaveEnv(SmokeFluidEnv):
    @property
    def trajectory_length(self) -> int:
        return 51
    
    @property
    def dt(self) -> float:
        return 1.0
    
    @property
    def sub_sampling_factor(self) -> int:
        return 4
    
    @property
    def max_density(self) -> float:
        return 1.0
    
    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (48, 48)
    
    @property
    def data_size(self) -> List[int]:
        return [5, 2, 1]
    
    @property
    def interesting_indices(self) -> Tensor:
        return [1, 0, 3, 2, 4, 5, 7, 8, 9, 10, 11, 12]


@Env.register("WaveLongEnv")
class WaveLongEnv(WaveEnv):
    @property
    def trajectory_length(self) -> int:
        return 100
    
    @property
    def data_size(self) -> List[int]:
        return [5, 2, 1]
    
    @property
    def sub_sampling_factor(self) -> int:
        return 4


@Env.register("WaveLong200Env")
class WaveLong200Env(WaveLongEnv):
    @property
    def trajectory_length(self) -> int:
        return 200
    
    @property
    def sub_sampling_factor(self) -> int:
        return 2
    
    @property
    def data_size(self) -> List[int]:
        return [5, 2, 1]


@Env.register("KuramotoEnv")
class KuramotoEnv(SmokeFluidEnv):
    @property
    def trajectory_length(self) -> int:
        return 400

    @property
    def dt(self) -> float:
        return 0.1
    
    @property
    def sub_sampling_factor(self) -> int:
        return 1

    @property
    def max_density(self) -> float:
        return 10.0

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (40, 40)

    @property
    def data_size(self) -> List[int]:
        return [5, 2, 1]

    @property
    def interesting_indices(self) -> Tensor:
        return [5, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
