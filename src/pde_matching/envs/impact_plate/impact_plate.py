import enum
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.data import BaseData, Data
from torch_geometric.transforms import Cartesian, Compose, Distance
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from pde_matching.data.preprocessing import MeanStdAccumulator, Standardizer
from pde_matching.data.state import StateKey
from pde_matching.data.transforms import EdgeCategorical
from pde_matching.envs.env import Env
from pde_matching.envs.trajectory_dataset import TrajectoryDatasetMixin
from pde_matching.envs.utils import EdgeFeatureScaler
from .impact_plate_visualizer import ImpactPlateVisualizer
from pde_matching.utils.common_utils import noise_like
from pde_matching.data.trajectory import DataTrajectory


class NodeType(enum.IntEnum):
    NORMAL = 0      # Normal nodes
    OBSTACLE = 1    # Obstacle/ball nodes
    AIRFOIL = 2     # Airfoil nodes
    HANDLE = 3      # Handle nodes
    INFLOW = 4      # Inflow boundary
    OUTFLOW = 5     # Outflow boundary
    WALL_BOUNDARY = 6  # Wall boundary
    SYMMETRIC = 7   # Symmetric boundary
    SIZE = 8


class EdgeType(enum.IntEnum):
    NORMAL_NORMAL = 0        # Normal-normal connections
    OBSTACLE_OBSTACLE = 1    # Obstacle-obstacle connections
    NORMAL_OBSTACLE = 2      # Normal-obstacle connections
    OBSTACLE_NORMAL = 3      # Obstacle-normal connections
    BOUNDARY_NORMAL = 4      # Boundary-normal connections
    NORMAL_BOUNDARY = 5      # Normal-boundary connections
    BOUNDARY_BOUNDARY = 6    # Boundary-boundary connections
    SIZE = 7


@Env.register("ImpactPlateEnv")
class ImpactPlateEnv(TrajectoryDatasetMixin, Env):
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
        **kwargs,
    ):
        # Transform similar to sphere_cloth_coupling
        transform = Compose([
            EdgeCategorical(EdgeType.SIZE),
            Cartesian(norm=False),
            Distance(norm=False),
            EdgeFeatureScaler(
                start_idx=EdgeType.SIZE, 
                end_idx=EdgeType.SIZE + 4, 
                scale_factor=1.0 / 10.0  # Max velocity scaling
            ),
        ])

        self.cache = {"idx": -1}
        self.stage = stage
        self.training_noise = training_noise
        self.training_noise_std = training_noise_std
        self._n_sequences = n_sequences
        self._n_total_sequences = None
        self._standardizer = None
        self._visualizer = None
        self._input_keys = input_keys
        self._output_keys = output_keys
        self._visualizer_kwargs = kwargs.pop("visualizer", {})

        super().__init__(
            root=root,
            window_size=window_size,
            window_shift=window_shift,
            transform=transform,
            log=log,
            **kwargs,
        )

    @property
    def raw_file_names(self) -> List[str]:
        return ["meta.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train/processed", "val/processed", "test/processed"]

    @property
    def stats_file(self) -> str:
        return Path(self.processed_dir) / "stats.npz"

    @property
    def dt(self) -> float:
        return 1.0  # Will be updated based on meta.json

    @property
    def trajectory_length(self) -> int:
        return 52  # From meta.json

    @property
    def n_total_sequences(self) -> int:
        if self._n_total_sequences is None:
            # Count processed files
            stage_dir = Path(self.processed_dir) / self.stage
            if stage_dir.exists():
                self._n_total_sequences = len([
                    f for f in stage_dir.iterdir()
                    if re.match(r"[0-9]+\.pt", f.name)
                ])
            else:
                self._n_total_sequences = 0
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
            self._visualizer = ImpactPlateVisualizer(
                output_mask_fn=self.output_mask, **self._visualizer_kwargs
            )
        return self._visualizer
    
    @property
    def pos_scale(self) -> float:
        return 1.0
    
    @property
    def stress_scale(self) -> float:
        return 1000.0

    @property
    def node_type_dim(self) -> int:
        return NodeType.SIZE

    def loss_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        node_types = torch.argmax(input, dim=-1)
        
        mask_x = torch.logical_or(
            torch.eq(node_types, NodeType.HANDLE),
            torch.eq(node_types, NodeType.SYMMETRIC)
        )
        mask_y = torch.eq(node_types, NodeType.HANDLE)
        mask = torch.stack([mask_x, mask_y], dim=1)
        mask = torch.logical_not(mask)
        return mask

    def output_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        node_types = torch.argmax(input, dim=-1)
        
        mask_x = torch.logical_or(
            torch.eq(node_types, NodeType.HANDLE),
            torch.eq(node_types, NodeType.SYMMETRIC)
        )
        mask_y = torch.eq(node_types, NodeType.HANDLE)
        mask = torch.stack([mask_x, mask_y], dim=1)
        mask = torch.logical_not(mask)
        return mask

    def control_params_mask(self, input: Tensor) -> Tensor:
        """No control parameters in this setup."""
        if input.ndim == 3:
            input = input[0]
        return torch.zeros(input.shape[0], dtype=torch.bool)

    def process(self):
        """Process TFRecord files into PyTorch format."""
        try:
            from tfrecord_lite import tf_record_iterator
        except ImportError:
            raise ImportError("tfrecord_lite is required. Install with: pip install tfrecord-lite")

        raw_root = Path(self.raw_dir)
        processed_dir = Path(self.processed_dir)

        # Load meta information
        meta_file = raw_root / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.json not found in {raw_root}")
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        print("Meta information:", meta)

        # Process each stage
        for stage in ["train", "val", "test"]:
            stage_dir = processed_dir / stage
            stage_dir.mkdir(exist_ok=True, parents=True)

            # Map stage names to file names
            tfrecord_file = {
                "train": "train.tfrecord",
                "val": "valid.tfrecord", 
                "test": "test.tfrecord"
            }[stage]

            tfrecord_path = raw_root / tfrecord_file
            if not tfrecord_path.exists():
                print(f"Warning: {tfrecord_file} not found, skipping {stage}")
                continue

            self._process_stage(tfrecord_path, stage_dir, meta, stage)

    def _process_stage(self, tfrecord_path: Path, stage_dir: Path, meta: dict, stage: str):
        """Process a single TFRecord file."""
        try:
            from tfrecord_lite import tf_record_iterator
        except ImportError:
            raise ImportError("tfrecord_lite is required. Install with: pip install tfrecord-lite")

        print(f"Processing {stage} from {tfrecord_path}")

        u_stats = MeanStdAccumulator()
        u_dot_stats = MeanStdAccumulator()
        u_dot_dot_stats = MeanStdAccumulator()
        trajectory_idx = 0

        for example_idx, example in enumerate(tqdm(tf_record_iterator(str(tfrecord_path)), desc=f"Processing {stage}")):
            # Extract features and create trajectory data
            trajectory_data = self._extract_features(example, meta, is_first=(example_idx == 0))
            
            # Accumulate statistics from all timesteps
            for data in trajectory_data:
                u_stats.add(data.u.detach().numpy())
                if hasattr(data, 'u_dot'):
                    u_dot_stats.add(data.u_dot.detach().numpy())
                if hasattr(data, 'u_dot_dot'):
                    u_dot_dot_stats.add(data.u_dot_dot.detach().numpy())
            
            # Save trajectory
            trajectory_file = stage_dir / f"{trajectory_idx}.pt"
            torch.save(trajectory_data, trajectory_file)
            trajectory_idx += 1

        # Mark stage as processed
        processed_file = stage_dir / "processed"
        processed_file.touch()

        # Save statistics for training stage
        if stage == "train" and trajectory_idx > 0:
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

        print(f"Processed {trajectory_idx} trajectories for {stage}")

    def triangles_to_edges(self, faces):
        """Computes mesh edges from triangles (adapted from TensorFlow version)."""
        # Collect edges from triangles
        edges = torch.cat([
            faces[:, 0:2],
            faces[:, 1:3], 
            torch.stack([faces[:, 2], faces[:, 0]], dim=1)
        ], dim=0)
        
        # Sort edges (smaller index first)
        receivers = torch.min(edges, dim=1)[0]
        senders = torch.max(edges, dim=1)[0]
        
        # Pack edges as single int64 for duplicate removal
        packed_edges = senders.long() * 1000000 + receivers.long()  # Assuming max node < 1M
        
        # Remove duplicates
        unique_packed = torch.unique(packed_edges)
        
        # Unpack edges
        unique_senders = unique_packed // 1000000
        unique_receivers = unique_packed % 1000000
        
        # Create bidirectional edges
        edge_index = torch.stack([
            torch.cat([unique_senders, unique_receivers]),
            torch.cat([unique_receivers, unique_senders])
        ])
        
        return edge_index

    def _add_obstacle_normal_connections(self, edge_index, edge_types, normal_nodes, obstacle_nodes, final_positions, distance_threshold=1.0):
        """Add connections between obstacle and normal nodes within distance threshold."""
        if len(obstacle_nodes) == 0 or len(normal_nodes) == 0:
            return edge_index, edge_types
        
        normal_positions = final_positions[normal_nodes]    # (n_normal, 3)
        obstacle_positions = final_positions[obstacle_nodes]  # (n_obstacle, 3)
        
        # Compute pairwise distances (vectorized)
        distances = torch.norm(normal_positions[:, None, :] - obstacle_positions[None, :, :], dim=2)
        
        # Find pairs within distance threshold
        close_pairs = torch.where(distances < distance_threshold)
        normal_indices = close_pairs[0]
        obstacle_indices = close_pairs[1]
        
        if len(normal_indices) == 0:
            return edge_index, edge_types
        
        # Convert to original node indices
        normal_node_indices = torch.tensor(normal_nodes)[normal_indices]
        obstacle_node_indices = torch.tensor(obstacle_nodes)[obstacle_indices]
        
        # Create bidirectional connections
        new_edges = torch.stack([
            torch.cat([obstacle_node_indices, normal_node_indices]),
            torch.cat([normal_node_indices, obstacle_node_indices])
        ])
        
        # Create edge types for new connections
        n_connections = len(normal_indices)
        new_edge_types = torch.cat([
            torch.full((n_connections,), EdgeType.OBSTACLE_NORMAL, dtype=torch.long),
            torch.full((n_connections,), EdgeType.NORMAL_OBSTACLE, dtype=torch.long)
        ])
        
        # Combine with existing edges
        combined_edge_index = torch.cat([edge_index, new_edges], dim=1)
        combined_edge_types = torch.cat([edge_types, new_edge_types])
        
        return combined_edge_index, combined_edge_types

    def _extract_features(self, example: dict, meta: dict, is_first: bool = False) -> dict:
        """Extract features from a TFRecord example and create graph structure like sphere_cloth."""
        
        def extract_feature_to_torch(name: str, expected_dtype: str):
            """Extract a feature from the example and convert to torch tensor."""
            if name not in example:
                raise KeyError(f"Feature {name} not found in example")
            
            # Get raw bytes
            raw_data = example[name][0] if isinstance(example[name], list) else example[name]
            
            # Convert based on expected dtype
            if expected_dtype == "int32":
                dtype = np.int32
            elif expected_dtype == "float32":
                dtype = np.float32
            else:
                raise ValueError(f"Unsupported dtype: {expected_dtype}")
            
            # Parse the data
            data = np.frombuffer(raw_data, dtype=dtype)
            
            # Get shape info from meta and reshape
            if name in meta["features"]:
                shape_info = meta["features"][name]["shape"]
                if len(shape_info) > 0:
                    data = data.reshape(shape_info)
            
            return torch.tensor(data, dtype=torch.float32 if dtype == np.float32 else torch.long)

        # Extract all features
        extracted = {}

        # Static features
        extracted['cells'] = extract_feature_to_torch('cells', 'int32')
        extracted['mesh_pos'] = extract_feature_to_torch('mesh_pos', 'float32')
        extracted['node_type'] = extract_feature_to_torch('node_type', 'int32')
        extracted['density'] = extract_feature_to_torch('density', 'float32') / 1000  # Normalize density to kg/m^3
        extracted['modulus'] = torch.log10(extract_feature_to_torch('modulus', 'float32')) # Log scale modulus for stability
        extracted['lap_pe'] = extract_feature_to_torch('lap_pe', 'float32')
        
        # Dynamic features
        extracted['world_pos'] = extract_feature_to_torch('world_pos', 'float32') / self.pos_scale  # Normalize positions to meters
        extracted['stress'] = extract_feature_to_torch('stress', 'float32') / self.stress_scale  # Normalize stress to kPa

        # Print shapes only for first trajectory
        if is_first:
            print(f"Extracted features shapes:")
            for key, value in extracted.items():
                print(f"  {key}: {value.shape}")
        
        # Process the data to create graph structure
        
        # Remove batch dimensions and get 2D positions
        mesh_pos = extracted['mesh_pos'].squeeze(0)  # (N, 2)
        node_type_raw = extracted['node_type'].squeeze()  # (N,)
        cells = extracted['cells'].squeeze(0)  # (M, 3) where M is number of triangles
        world_pos = extracted['world_pos']  # (T, N, 2)
        stress = extracted['stress']  # (T, N, 1)

        # Filter nodes within bounding box x=(0, 1), y=(0, 2)
        x_mask = (mesh_pos[:, 0] >= 0.0) & (mesh_pos[:, 0] <= 1 * self.pos_scale)
        y_mask = (mesh_pos[:, 1] >= 0.0) & (mesh_pos[:, 1] <= 2 * self.pos_scale)
        # node_mask = x_mask & y_mask
        node_mask = torch.ones(len(mesh_pos), dtype=torch.bool)  # Keep all nodes for now
        
        # Get indices of nodes to keep
        keep_indices = torch.where(node_mask)[0]
        n_keep = len(keep_indices)
        
        # Create mapping from old to new indices
        old_to_new = torch.full((len(mesh_pos),), -1, dtype=torch.long)
        old_to_new[keep_indices] = torch.arange(n_keep)
        
        # Filter mesh data
        mesh_pos = mesh_pos[keep_indices]
        node_type_raw = node_type_raw[keep_indices]
        
        # Filter cells to only include those with all vertices in the filtered set
        valid_cells = []
        for cell in cells:
            if torch.all(node_mask[cell]):  # All vertices are in the filtered set
                new_cell = old_to_new[cell]
                valid_cells.append(new_cell)
        
        if len(valid_cells) > 0:
            cells = torch.stack(valid_cells)
        else:
            cells = torch.empty((0, 3), dtype=torch.long)
        
        # Dynamic fields (T, N, ...)
        world_pos = world_pos[:, keep_indices]  # (T, N_filtered, 2)
        stress = stress[:, keep_indices]  # (T, N_filtered, 1)
        density = extracted['density'][:, keep_indices]  # (T, N_filtered, 1)
        modulus = extracted['modulus'][:, keep_indices]  # (T, N_filtered, 1)
        
        n_nodes = len(mesh_pos)
        n_timesteps = stress.shape[0]
        
        if is_first:
            print(f"Filtered to {n_nodes} nodes from {len(node_mask)} original nodes")
            print(f"Filtered to {len(cells)} cells from {len(extracted['cells'].squeeze(0))} original cells")
        
        # Convert 2D positions to 3D by adding z=0 for mesh_pos
        mesh_pos_3d = torch.cat([mesh_pos, torch.zeros(n_nodes, 1)], dim=1)
        
        # Convert world_pos to 3D by adding z coordinate from stress (deformation)
        world_pos_3d = torch.zeros(n_timesteps, n_nodes, 3)
        world_pos_3d[:, :, :2] = world_pos  # x, y coordinates
        
        # Get node lists based on node_type
        normal_nodes = (node_type_raw == NodeType.NORMAL).nonzero(as_tuple=True)[0].tolist()
        obstacle_nodes = (node_type_raw == NodeType.OBSTACLE).nonzero(as_tuple=True)[0].tolist()
        handle_nodes = (node_type_raw == NodeType.HANDLE).nonzero(as_tuple=True)[0].tolist()
        symmetric_nodes = (node_type_raw == NodeType.SYMMETRIC).nonzero(as_tuple=True)[0].tolist()
        
        if is_first:
            print(f"Node type distribution:")
            print(f"  Normal: {len(normal_nodes)}, Obstacle: {len(obstacle_nodes)}")
            print(f"  Handle: {len(handle_nodes)}, Symmetric: {len(symmetric_nodes)}")
        
        # Create node type one-hot encoding
        node_types_onehot = torch.zeros(n_nodes, NodeType.SIZE)
        for i in range(n_nodes):
            node_types_onehot[i, node_type_raw[i]] = 1
        
        # Combine node type, density, and modulus for properties
        density = density.squeeze(0)  # (N,)
        modulus = modulus.squeeze(0)  # (N,)
        properties = torch.cat([
            node_types_onehot,  # (N, NodeType.SIZE)
            density,  # (N, 1)
            modulus   # (N, 1)
        ], dim=1)  # (N, NodeType.SIZE + 2)
        
        # Extract edges from cells using triangles_to_edges
        edge_index = self.triangles_to_edges(cells)
        
        # Add obstacle-normal connections based on final timestep positions
        final_positions = world_pos_3d[-1]  # Use last timestep for proximity
        edge_index, edge_types = self._add_obstacle_normal_connections(
            edge_index, torch.zeros(edge_index.shape[1], dtype=torch.long),
            normal_nodes, obstacle_nodes, final_positions, distance_threshold=1.0 / self.pos_scale
        )
        
        # Set edge types based on node types for remaining edges
        for i in range(edge_index.shape[1]):
            if edge_types[i] != 0:  # Skip if already set by obstacle-normal connections
                continue
                
            src, dst = edge_index[0, i], edge_index[1, i]
            src_type = node_type_raw[src].item()
            dst_type = node_type_raw[dst].item()
            
            if src_type == NodeType.NORMAL and dst_type == NodeType.NORMAL:
                edge_types[i] = EdgeType.NORMAL_NORMAL
            elif src_type == NodeType.OBSTACLE and dst_type == NodeType.OBSTACLE:
                edge_types[i] = EdgeType.OBSTACLE_OBSTACLE
            elif src_type in [NodeType.HANDLE, NodeType.SYMMETRIC] and dst_type == NodeType.NORMAL:
                edge_types[i] = EdgeType.BOUNDARY_NORMAL
            elif src_type == NodeType.NORMAL and dst_type in [NodeType.HANDLE, NodeType.SYMMETRIC]:
                edge_types[i] = EdgeType.NORMAL_BOUNDARY
            else:
                edge_types[i] = EdgeType.BOUNDARY_BOUNDARY
        
        if is_first:
            print(f"Total edges: {edge_index.shape[1]}")
            print(f"Edge types distribution: {torch.bincount(edge_types)}")
        
        # Create trajectory data list (like sphere_cloth)
        trajectory_data = []
        
        for t in range(n_timesteps):
            # Concatenate positions and stress for the state
            # u_state = torch.cat([world_pos_3d[t], stress[t]], dim=1)  # (N, 4) = (x, y, z, stress)
            # u_state = stress[t]  # Use stress only for now
            u_state = torch.cat([
                world_pos_3d[t],
                torch.zeros(n_nodes, 1),
            ], dim=1)  # (N, 3) = (x, y, z)
            
            graph_data = Data(
                pos=world_pos_3d[t],  # Current 3D positions 
                u=u_state,           # State includes positions + stress
                properties=properties,  # Node features: type + density + modulus
                edge_index=edge_index,
                edge_type=edge_types,
                # Additional data for visualization and processing
                mesh_pos=mesh_pos_3d,  # Initial mesh positions
                stress=stress[t],      # Stress at this timestep
                cells=cells,           # Cell connectivity
                normal_nodes=torch.tensor(normal_nodes, dtype=torch.long),
                obstacle_nodes=torch.tensor(obstacle_nodes, dtype=torch.long),
                handle_nodes=torch.tensor(handle_nodes, dtype=torch.long),
                boundary_nodes=torch.tensor(symmetric_nodes, dtype=torch.long),
                density=extracted['density'].squeeze(0),
                modulus=extracted['modulus'].squeeze(0),
                lap_pe=extracted['lap_pe'].squeeze(0),
            )
            trajectory_data.append(graph_data)
        
        # Compute u_dot (velocities) from consecutive states
        for t in range(len(trajectory_data)):
            if t == len(trajectory_data) - 1:
                # For last timestep, copy velocity from previous
                trajectory_data[t].u_dot = trajectory_data[t-1].u_dot.clone()
            else:
                trajectory_data[t].u_dot = torch.cat([
                    (trajectory_data[t+1].u[..., :-1] - trajectory_data[t].u[..., :-1]) / self.dt,
                    stress[t]
                ], dim=1)
        
        # Compute u_dot_dot (accelerations) from consecutive velocities
        for t in range(len(trajectory_data)):
            if t == len(trajectory_data) - 1:
                # For last timestep, copy acceleration from previous
                trajectory_data[t].u_dot_dot = trajectory_data[t-1].u_dot_dot.clone()
            else:
                trajectory_data[t].u_dot_dot = (trajectory_data[t+1].u_dot - trajectory_data[t].u_dot) / self.dt
        
        return trajectory_data
    
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

                state[..., :mask.shape[1]] = torch.where(
                    mask, 
                    state[..., :mask.shape[1]], 
                    trajectory.u[0][..., :mask.shape[1]]
                )
                new_trajectory.state_list[0].u = state

                state_dot[..., :mask.shape[1]] = torch.where(
                    mask, 
                    state_dot[..., :mask.shape[1]], 
                    trajectory.u_dot[0][..., :mask.shape[1]]
                )
                new_trajectory.state_list[0].u_dot = state_dot
            else:
                # Clone the state to prevent in-place modification
                state = state.clone()
                state[..., :mask.shape[1]] = torch.where(
                    mask, 
                    state[..., :mask.shape[1]], 
                    trajectory.u[0][..., :mask.shape[1]]
                )
                new_trajectory.state_list[0].u = state

                if output is not None:
                    state_dot = output.clone()
                    state_dot[..., :mask.shape[1]] = torch.where(
                        mask, 
                        state_dot[..., :mask.shape[1]], 
                        trajectory.u_dot[0][..., :mask.shape[1]]
                    )
                    new_trajectory.state_list[0].u_dot = state_dot

        # Update the mesh with new positions
        graph = trajectory.data_list[0]
        graph.pos = self._extract_pos_from_state_or_graph(state, graph)
        new_trajectory.data_list[0] = self.post_update_data(graph)

        return new_trajectory
    
    def additional_metrics(self, predictions: Tensor, targets: DataTrajectory, 
                           mask: Optional[Tensor] = None) -> Dict[str, float]:
        """Compute additional metrics for the predictions."""
        # Assuming predictions is a tensor of [u, u_dot]
        # where u = [pos_x, pos_y, pos_z, 0]
        # and u_dot = [vel_x, vel_y, vel_z, stress]

        pos_pred = predictions[..., :2]
        pos_target = targets.u[..., :2] 

        stress_pred = predictions[..., -1].unsqueeze(-1)
        stress_target = targets.u_dot[..., -1].unsqueeze(-1)

        # Rescale positions and stress to original scale
        pos_pred = pos_pred * self.pos_scale
        pos_target = pos_target * self.pos_scale
        stress_pred = stress_pred * self.stress_scale
        stress_target = stress_target * self.stress_scale

        rmse_pos_sample_wise = (pos_pred - pos_target) ** 2
        rmse_stress_sample_wise = (stress_pred - stress_target) ** 2

        if mask is not None:
            rmse_pos_sample_wise[..., :mask.shape[1]] = torch.where(
                mask, 
                rmse_pos_sample_wise[..., :mask.shape[1]], 
                torch.zeros_like(rmse_pos_sample_wise[..., :mask.shape[1]])
            )

        rmse_pos = torch.sqrt(torch.mean(torch.sum(rmse_pos_sample_wise, dim=-1), -1))
        rmse_stress = torch.sqrt(torch.mean(torch.sum(rmse_stress_sample_wise, dim=-1), -1))

        scalars_pos = {f"pos_{horizon}": rmse_pos[horizon-1].item() * 1E3 for horizon in [1, 50, rmse_pos.shape[0]]}
        scalars_stress = {f"stress_{horizon}": rmse_stress[horizon-1].item() * 1E3 for horizon in [1, 50, rmse_stress.shape[0]]}

        # Compute metrics
        metrics = {
            **scalars_pos,
            **scalars_stress,
        }

        return metrics


@Env.register("ImpactPlateNormEnv")
class ImpactPlateNormEnv(ImpactPlateEnv):
    @property
    def pos_scale(self) -> float:
        return 10.0

    @property
    def stress_scale(self) -> float:
        return 1000.0
    
    @property
    def interesting_indices(self) -> List[int]:
        return [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # First 10 nodes