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
from pde_matching.envs.sphere_cloth_coupling.sphere_cloth_coupling_visualizer import (
    SphereClothCouplingVisualizer,
)
from torch import Tensor
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import Cartesian, Compose, Distance
from torch_geometric.nn import knn
from tqdm import tqdm

from ..env import Env
from ..trajectory_dataset import TrajectoryDatasetMixin
from ..utils import EdgeFeatureScaler, extract_triangular_faces
from .utils import faces_to_edges, create_cloth_grid, create_cloth_edges


class NodeType(enum.IntEnum):
    CLOTH = 0
    CLOTH_CORNER = 1  # Cloth corners (connect to sphere centers)
    SPHERE_CENTER = 2  # Sphere center nodes (3 spheres)
    SIZE = 3


class EdgeType(enum.IntEnum):
    CLOTH_CLOTH = 0  # Internal cloth connections
    SPHERE_CORNER = 1  # Connection from sphere centers to cloth corners
    CORNER_SPHERE = 2  # Connection from cloth corners to sphere centers
    SIZE = 3


@Env.register("SphereClothCouplingEnv")
class SphereClothCouplingEnv(TrajectoryDatasetMixin, Env):
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
        connection_radius: float = 0.3,
        mesh_type: str = "triangular",
        **kwargs,
    ):
        self._max_velocity = 10.0
        self._max_position = 10.0
        self._mesh_type = mesh_type

        transform = Compose(
            [
                EdgeCategorical(EdgeType.SIZE),
                Cartesian(norm=False),
                Distance(norm=False),
                EdgeFeatureScaler(
                    start_idx=EdgeType.SIZE, end_idx=EdgeType.SIZE + 4, scale_factor=1.0 / self._max_velocity
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

        self._train_size = self.datasets_size[0]
        self._val_size = self.datasets_size[1]
        self._test_size = self.datasets_size[2]

        super().__init__(
            root=root,
            window_size=window_size,
            window_shift=window_shift,
            transform=transform,
            log=log,
            **kwargs,
        )

    @property
    def datasets_size(self) -> List[int]:
        return [16, 2, 2]  # Note there are 50 trajectories per pickle file

    @property
    def raw_dirs_range_tuple(self) -> List[Tuple[int, int]]:
        # Ranges refer to pickle file indices (each pickle file contains 50 trajectories)
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

    @property
    def dt(self) -> float:
        return 0.1
    
    @property
    def node_type_dim(self) -> int:
        return NodeType.SIZE

    @property
    def n_total_sequences(self) -> int:
        if self._n_total_sequences is None:
            if self.stage == "train":
                self._n_total_sequences = self._train_size * 50  # 16 pickle files * 50 trajectories each
            elif self.stage == "val":
                self._n_total_sequences = self._val_size * 50   # 2 pickle files * 50 trajectories each
            elif self.stage == "test":
                self._n_total_sequences = self._test_size * 50  # 2 pickle files * 50 trajectories each

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
            self._visualizer = SphereClothCouplingVisualizer(self.trajectory_length, **self._visualizer_kwargs)
        return self._visualizer
    
    @property
    def interesting_indices(self) -> Tensor:
        return [2, 2, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    def loss_mask(self, input: Tensor) -> Tensor:
        """Only predict cloth nodes (both regular cloth and cloth corners)."""
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.CLOTH
        mask |= torch.argmax(input, dim=-1) == NodeType.SPHERE_CENTER
        return mask

    def output_mask(self, input: Tensor) -> Tensor:
        """Only predict cloth nodes (both regular cloth and cloth corners)."""
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.CLOTH
        mask |= torch.argmax(input, dim=-1) == NodeType.SPHERE_CENTER
        return mask

    def control_params_mask(self, input: Tensor) -> Tensor:
        """No control parameters in this simplified setup."""
        if input.ndim == 3:
            input = input[0]
        # Return empty mask - no control parameters
        return torch.zeros(input.shape[0], dtype=torch.bool)

    def _infer_cloth_grid_dimensions(self, cloth_positions):
        """Infer cloth grid dimensions from number of cloth points."""
        n_points = len(cloth_positions)
        n_rows = int(np.sqrt(n_points))
        n_cols = n_rows
        if n_rows * n_cols != n_points:
            raise ValueError(f"Cloth points ({n_points}) do not form a square grid")
        return n_rows, n_cols

    def _create_cloth_edges(self, cloth_positions):
        n_rows, n_cols = self._infer_cloth_grid_dimensions(cloth_positions)
        return create_cloth_edges(n_rows, n_cols, self._mesh_type)

    def _create_sphere_edges_from_faces(self, sphere_indices):
        return faces_to_edges(sphere_indices.reshape(-1, 3))

    def process(self):
        for stage in ["train", "val", "test"]:
            self._process_by_stage(stage)

    def _process_by_stage(self, stage):
        stage_dir = Path(self.processed_dir) / stage
        stage_dir.mkdir(exist_ok=True)

        position_stats = MeanStdAccumulator()
        velocity_stats = MeanStdAccumulator()

        u_stats = MeanStdAccumulator()
        u_dot_stats = MeanStdAccumulator()
        u_dot_dot_stats = MeanStdAccumulator()

        # Glob and sort pkl files by seed
        pkl_files = list(Path(self.raw_dir).glob("density=*.pkl"))
        pkl_files.sort(key=lambda x: int(x.stem.split("_seed=")[1]))
        
        # Select subset based on stage
        if stage == "train":
            start, end = self.raw_dirs_range_tuple[0]
        elif stage == "val":
            start, end = self.raw_dirs_range_tuple[1]
        elif stage == "test":
            start, end = self.raw_dirs_range_tuple[2]
        
        stage_pkl_files = pkl_files[start:end]

        trajectory_counter = 0
        
        for pkl_idx, pkl_file in enumerate(
            tqdm(stage_pkl_files, desc=f"Processing {stage}", unit=" pkl files")
        ):
            with open(pkl_file, "rb") as f:
                data = pkl.load(f)

            sphere_positions = data["sphere_positions"]  # (T, N, P_sphere * 3, 3)
            sphere_velocities = data["sphere_velocities"]  # (T, N, P_sphere * 3, 3)
            cloth_positions = data["cloth_positions"]  # (T, N, P_cloth, 3)
            cloth_velocities = data["cloth_velocities"]  # (T, N, P_cloth, 3)
            sphere_indices = data["sphere_indices"]  # Face indices
            sphere_mass = data["sphere_mass"]
            sphere_scale = data["sphere_scale"]
            num_spheres = data["num_spheres"] if "num_spheres" in data else 1

            num_envs = sphere_positions.shape[1]  # Should be 50
            
            for env_idx in range(num_envs):
                env_data_list = []
                
                # Get sphere initial positions for this environment (from t=0)
                sphere_initial_pos = sphere_positions[0, env_idx]  # (P_sphere * 3, 3)
                p_sphere_per_sphere = len(sphere_initial_pos) // num_spheres
                
                # Split into 3 spheres and compute their centers
                sphere_centers_initial = []
                for sphere_idx in range(num_spheres):
                    start_idx = sphere_idx * p_sphere_per_sphere
                    end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                    sphere_mesh = sphere_initial_pos[start_idx:end_idx]
                    sphere_center = np.mean(sphere_mesh, axis=0)
                    sphere_centers_initial.append(sphere_center)
                sphere_centers_initial = np.array(sphere_centers_initial)  # (3, 3)
                
                for t in range(0, len(sphere_positions), 2):
                    # Get cloth positions and velocities for this environment and timestep
                    cl_pos = cloth_positions[t, env_idx]  # (P_cloth, 3)
                    cl_vel = cloth_velocities[t, env_idx]  # (P_cloth, 3)
                    
                    # Get sphere mesh positions and compute centers
                    sphere_mesh_pos = sphere_positions[t, env_idx]  # (P_sphere * 3, 3)
                    sphere_centers_current = []
                    sphere_meshes = []
                    for sphere_idx in range(num_spheres):
                        start_idx = sphere_idx * p_sphere_per_sphere
                        end_idx = (sphere_idx + 1) * p_sphere_per_sphere
                        sphere_mesh = sphere_mesh_pos[start_idx:end_idx]
                        sphere_center = np.mean(sphere_mesh, axis=0)
                        sphere_centers_current.append(sphere_center)
                        sphere_meshes.append(sphere_mesh)
                    sphere_centers_current = np.array(sphere_centers_current)  # (3, 3)
                    
                    # Identify corner nodes (4 corners of cloth grid)
                    n_rows, n_cols = self._infer_cloth_grid_dimensions(cl_pos)
                    corner_indices = [0, n_cols-1, (n_rows-1)*n_cols, n_rows*n_cols-1]  # 4 corners
                    
                    # Node types and features (only create once for this environment)
                    if t == 0:
                        n_cloth = len(cl_pos)
                        n_total_nodes = n_cloth + num_spheres  # cloth + 3 sphere centers
                        
                        # Node types: cloth nodes + sphere center nodes
                        node_types = np.zeros((n_total_nodes, NodeType.SIZE))
                        
                        # Cloth nodes (default type)
                        node_types[:n_cloth, NodeType.CLOTH] = 1
                        # Override corners
                        for corner_idx in corner_indices:
                            node_types[corner_idx, NodeType.CLOTH] = 0
                            node_types[corner_idx, NodeType.CLOTH_CORNER] = 1
                        # Sphere center nodes
                        for i in range(num_spheres):
                            node_types[n_cloth + i, NodeType.SPHERE_CENTER] = 1
                        
                        # Create edges
                        # 1. Cloth-cloth connections (internal cloth structure)
                        cloth_edges = self._create_cloth_edges(cl_pos)
                        cloth_edge_types = np.full(len(cloth_edges), EdgeType.CLOTH_CLOTH)

                        cloth_edges, cloth_edge_types = to_undirected(
                            torch.from_numpy(cloth_edges.T).long(),
                            torch.from_numpy(cloth_edge_types).long(),
                        )
                        
                        # 2. Sphere centers to subsampled cloth connections
                        def subsample_cloth_nodes(n_rows, n_cols, subsample_factor):
                            """Subsample cloth nodes uniformly across the entire grid"""
                            selected = set()
                            
                            # Sample every subsample_factor nodes in both dimensions
                            # for i in range(0, n_rows, subsample_factor):
                            #     for j in range(0, n_cols, subsample_factor):
                            #         node_idx = i * n_cols + j
                            #         selected.add(node_idx)
                            
                            # Always include 4 corners
                            corners = [0, n_cols-1, (n_rows-1)*n_cols, n_rows*n_cols-1]
                            selected.update(corners)
                            
                            return list(selected)
                        
                        sampled_cloth_nodes = subsample_cloth_nodes(n_rows, n_cols, 4)

                        # for cloth_idx in sampled_cloth_nodes:
                        #     node_types[cloth_idx, NodeType.CLOTH] = 0
                        #     node_types[cloth_idx, NodeType.CLOTH_CORNER] = 1
                        
                        sphere_cloth_edges = []
                        sphere_node_indices = list(range(n_cloth, n_cloth + num_spheres))
                        for sphere_node_idx in sphere_node_indices:
                            for cloth_idx in sampled_cloth_nodes:
                                sphere_cloth_edges.append([sphere_node_idx, cloth_idx])
                                sphere_cloth_edges.append([cloth_idx, sphere_node_idx])
                        
                        sphere_cloth_edges = np.array(sphere_cloth_edges)
                        sphere_cloth_edge_types = np.full(len(sphere_cloth_edges), EdgeType.SPHERE_CORNER)

                        sphere_cloth_edges = torch.from_numpy(sphere_cloth_edges.T).long()
                        sphere_cloth_edge_types = torch.from_numpy(sphere_cloth_edge_types).long()
                        
                        # Combine all edges
                        all_edges = torch.cat([cloth_edges, sphere_cloth_edges], dim=1)
                        all_edge_types = torch.cat([cloth_edge_types, sphere_cloth_edge_types], dim=0)
                    
                    # Node positions: cloth positions + 3 sphere center positions
                    all_positions = np.vstack([cl_pos, sphere_centers_current])
                    
                    node_attr = torch.from_numpy(node_types).float()
                    edge_index = all_edges.clone()
                    edge_attr = all_edge_types.clone()

                    # Create graph data
                    graph_data = Data(
                        pos=torch.from_numpy(all_positions).float(),  # All node positions (cloth + 3 sphere centers)
                        u=torch.from_numpy(all_positions).float(),  # Current positions (cloth + 3 sphere centers)
                        properties=node_attr,
                        edge_index=edge_index,
                        edge_type=edge_attr,  # Edge types as long tensor
                        cloth_indices=torch.arange(n_cloth),  # Only cloth nodes (corners are included)
                        corner_indices=torch.tensor(corner_indices, dtype=torch.long),  # Corner node indices
                        sphere_center_indices=torch.tensor(sphere_node_indices, dtype=torch.long),  # 3 sphere center indices
                        cloth_faces=torch.from_numpy(self._create_cloth_faces(cl_pos)).long(),
                        # Visualization data (not part of graph)
                        sphere_mesh_pos=torch.from_numpy(sphere_mesh_pos).float(),  # Full sphere meshes for visualization
                        sphere_faces=extract_triangular_faces(sphere_indices),
                        sphere_centers_initial=torch.from_numpy(sphere_centers_initial).float(),  # For reference
                        num_spheres=num_spheres,
                        p_sphere_per_sphere=p_sphere_per_sphere,
                    )
                    
                    env_data_list.append(graph_data)
                    
                    # Update statistics for u (all positions, but only cloth will be predicted)
                    u_stats.add(graph_data.u.detach().cpu().numpy())

                # Compute u_dot (velocities from consecutive u states)
                for t, data in enumerate(env_data_list):
                    if t == len(env_data_list) - 1:
                        # For the last timestep, copy velocity from previous timestep
                        data.u_dot = env_data_list[t - 1].u_dot
                    else:
                        data.u_dot = (env_data_list[t + 1].u - data.u) / self.dt
                        # # Sphere center doesn't move, so its velocity is always zero
                        # if hasattr(data, 'sphere_center_idx'):
                        #     data.u_dot[data.sphere_center_idx] = 0.0
                    u_dot_stats.add(data.u_dot.detach().cpu().numpy())

                # Compute u_dot_dot (accelerations from consecutive u_dot states)
                for t, data in enumerate(env_data_list):
                    if t == len(env_data_list) - 1:
                        data.u_dot_dot = env_data_list[t - 1].u_dot_dot
                    else:
                        data.u_dot_dot = (env_data_list[t + 1].u_dot - data.u_dot) / self.dt
                        # # Sphere center doesn't accelerate, so its acceleration is always zero
                        # if hasattr(data, 'sphere_center_idx'):
                        #     data.u_dot_dot[data.sphere_center_idx] = 0.0
                    u_dot_dot_stats.add(data.u_dot_dot.detach().cpu().numpy())
                
                # Save each trajectory (environment) as a separate .pt file
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

    def _create_cloth_faces(self, cloth_positions):
        n_rows, n_cols = self._infer_cloth_grid_dimensions(cloth_positions)
        return create_cloth_grid(n_rows, n_cols, self._mesh_type)


@Env.register("SphereClothCouplingLongEnv")
class SphereClothCouplingLongEnv(SphereClothCouplingEnv):
    @property
    def trajectory_length(self) -> int:
        return 100
