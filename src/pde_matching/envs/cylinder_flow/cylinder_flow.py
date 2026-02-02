import enum
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from pde_matching.data.trajectory import DataTrajectory
from pde_matching.data.preprocessing import MeanStdAccumulator, Standardizer
from pde_matching.data.state import StateKey
from pde_matching.data.transforms import EdgeCategorical
from pde_matching.envs.env import Env
from pde_matching.envs.trajectory_dataset import TrajectoryDatasetMixin
from pde_matching.envs.utils import EdgeFeatureScaler
from pde_matching.utils.common_utils import noise_like
from torch import Tensor
from torch_geometric.data.data import BaseData, Data
from torch_geometric.transforms import Cartesian, Compose, Distance
from torch_geometric.utils import to_undirected

from .cylinder_flow_visualizer import CylinderFlowVisualizer


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


class EdgeType(enum.IntEnum):
    NORMAL_NORMAL = 0        # Normal-normal connections
    NORMAL_OBSTACLE = 1      # Normal-obstacle connections
    OBSTACLE_NORMAL = 2      # Obstacle-normal connections
    OBSTACLE_OBSTACLE = 3    # Obstacle-obstacle connections
    BOUNDARY_NORMAL = 4      # Boundary-normal connections
    NORMAL_BOUNDARY = 5      # Normal-boundary connections
    BOUNDARY_BOUNDARY = 6    # Boundary-boundary connections
    SIZE = 7


@Env.register("CylinderFlowEnv")
class CylinderFlowEnv(TrajectoryDatasetMixin, Env):
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
        # Transform similar to airfoil
        transform = Compose([
            EdgeCategorical(EdgeType.SIZE),
            Cartesian(norm=False),
            Distance(norm=False),
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
        return 0.1

    @property
    def trajectory_length(self) -> int:
        # return 203
        return 102
    
    @property
    def node_type_dim(self) -> int:
        return NodeType.SIZE

    @property
    def interesting_indices(self) -> Tensor:
        return [17, 18, 19, 20, 21, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        # return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # return [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        # return [40, 17, 37, 38, 39, 40, 56, 80, 81, 97, 98, 100, 120, 122, 125, 127, 129, 131, 133]

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
            self._visualizer = CylinderFlowVisualizer(**self._visualizer_kwargs)
        return self._visualizer

    def loss_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.NORMAL
        mask |= torch.argmax(input, dim=-1) == NodeType.OUTFLOW
        return mask

    def output_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.NORMAL
        mask |= torch.argmax(input, dim=-1) == NodeType.OUTFLOW
        return mask

    def control_params_mask(self, input: Tensor) -> Tensor:
        """No control parameters in this setup."""
        if input.ndim == 3:
            input = input[0]
        return torch.zeros(input.shape[0], dtype=torch.bool)

    def _extract_pos_from_state_or_graph(self, state: torch.Tensor, graph: Data) -> torch.Tensor:
        # For static mesh, use fixed positions
        return graph.mesh_pos

    def post_update_data(self, data):
        # For static mesh, no position update needed
        return data
    
    def _add_noise_to_data(self, data_list):
        for i in range(len(data_list)):
            u_noise = noise_like(data_list[i].u, self.training_noise_std)
            u_dot_noise = noise_like(data_list[i].u_dot, self.training_noise_std)
            u_dot_dot_noise = noise_like(
                data_list[i].u_dot_dot, self.training_noise_std
            )

            # Noise is added to the state
            data_list[i].u_noise = u_noise
            data_list[i].u_dot_noise = u_dot_noise
            data_list[i].u_dot_dot_noise = u_dot_dot_noise

    def triangles_to_edges(self, faces):
        """Computes mesh edges from triangles."""
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
        packed_edges = senders.long() * 100000 + receivers.long()  # Assuming max node < 100k
        
        # Remove duplicates
        unique_packed = torch.unique(packed_edges)
        
        # Unpack edges
        unique_senders = unique_packed // 100000
        unique_receivers = unique_packed % 100000
        
        # Create bidirectional edges
        edge_index = torch.stack([
            torch.cat([unique_senders, unique_receivers]),
            torch.cat([unique_receivers, unique_senders])
        ])
        
        return edge_index

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

    def _extract_features(self, example: dict, meta: dict, is_first: bool = False) -> List[Data]:
        """Extract features from a TFRecord example and create graph structure."""
        
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
        
        # Dynamic features
        extracted['velocity'] = extract_feature_to_torch('velocity', 'float32')
        extracted['pressure'] = extract_feature_to_torch('pressure', 'float32')
        
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
        
        # Dynamic fields (T, N, ...)
        velocity = extracted['velocity']  # (T, N, 2)
        pressure = extracted['pressure']  # (T, N, 1)
        
        n_nodes = len(mesh_pos)
        n_timesteps = velocity.shape[0]
        
        if is_first:
            print(f"Number of nodes: {n_nodes}, timesteps: {n_timesteps}")
            print("Node types distribution:", torch.bincount(node_type_raw))

        # Subsample every 6th timestep to get 100 timesteps
        sub_sample_rate = 1
        timestep_indices = list(range(0, n_timesteps, sub_sample_rate))
        # timestep_indices = list(range(300, n_timesteps, 6))[:50]
        n_timesteps = len(timestep_indices)
        
        # Convert 2D positions to 3D by adding z=0
        mesh_pos_3d = torch.cat([mesh_pos, torch.zeros(n_nodes, 1)], dim=1)
        
        # Create node type one-hot encoding from actual node_type_raw
        node_types_onehot = torch.zeros(n_nodes, NodeType.SIZE)
        node_types_onehot[torch.arange(n_nodes), node_type_raw] = 1
        
        # Properties include node type
        properties = node_types_onehot
        # properties = torch.cat([
        #     node_types_onehot,
        #     mesh_pos_3d,  # Add positions as additional features
        # ], dim=1)
        
        # Extract edges from cells using triangles_to_edges
        edge_index = self.triangles_to_edges(cells)
        
        # Edge typing based on actual node types
        edge_types = torch.zeros(edge_index.shape[1], dtype=torch.long)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # src_type = torch.argmax(node_types_onehot[src]).item()
            # dst_type = torch.argmax(node_types_onehot[dst]).item()
            src_type = node_type_raw[src].item()
            dst_type = node_type_raw[dst].item()

            if src_type in [NodeType.NORMAL, NodeType.INFLOW, NodeType.OUTFLOW] and \
                dst_type in [NodeType.NORMAL, NodeType.INFLOW, NodeType.OUTFLOW]:
                edge_types[i] = EdgeType.NORMAL_NORMAL
            elif src_type in [NodeType.NORMAL, NodeType.INFLOW, NodeType.OUTFLOW] and \
                dst_type == NodeType.WALL_BOUNDARY:
                edge_types[i] = EdgeType.NORMAL_BOUNDARY
            elif dst_type in [NodeType.NORMAL, NodeType.INFLOW, NodeType.OUTFLOW] and \
                src_type ==  NodeType.WALL_BOUNDARY:
                edge_types[i] = EdgeType.BOUNDARY_NORMAL
            else:
                edge_types[i] = EdgeType.BOUNDARY_BOUNDARY
        
        if is_first:
            print(f"Total edges: {edge_index.shape[1]}")
            print(f"Edge types distribution: {torch.bincount(edge_types)}")
        
        # Create trajectory data list
        trajectory_data = []
        
        for t_idx, t in enumerate(timestep_indices):
            # Use velocity as the main state variable
            # u_state = velocity[t]  # (N, 2) - velocity components
            u_state = torch.cat([
                velocity[t], 
                pressure[t]
            ], dim=-1)
            
            graph_data = Data(
                pos=mesh_pos_3d,        # Fixed 3D positions 
                u=u_state,              # Velocity state
                properties=properties,  # Node features: type
                edge_index=edge_index,
                edge_type=edge_types,
                # Additional data for visualization and processing
                mesh_pos=mesh_pos_3d,  # Same as pos for fixed mesh
                velocity=velocity[t],  # Velocity at this timestep
                pressure=pressure[t],  # Pressure at this timestep  
                cells=cells,           # Cell connectivity
                timestep=t_idx,        # Current timestep index
            )
            trajectory_data.append(graph_data)
        
        # Compute u_dot (time derivatives) from consecutive states
        for t in range(len(trajectory_data)):
            if t == len(trajectory_data) - 1:
                # For last timestep, copy derivative from previous
                trajectory_data[t].u_dot = trajectory_data[t-1].u_dot.clone()
            else:
                trajectory_data[t].u_dot = (trajectory_data[t+1].u - trajectory_data[t].u) / self.dt

        # Compute u_dot_dot (accelerations) from consecutive derivatives
        for t in range(len(trajectory_data)):
            if t == len(trajectory_data) - 1:
                # For last timestep, copy acceleration from previous
                trajectory_data[t].u_dot_dot = trajectory_data[t-1].u_dot_dot.clone()
            else:
                trajectory_data[t].u_dot_dot = (trajectory_data[t+1].u_dot - trajectory_data[t].u_dot) / self.dt

        return trajectory_data

    def additional_metrics(self, predictions: Tensor, targets: DataTrajectory, 
                           mask: Optional[Tensor] = None) -> Dict[str, float]:
        """Compute additional metrics for the predictions."""
        mae_sample_wise = (predictions - targets.state_vector).abs()
        if mask is not None:
            mae_sample_wise = mae_sample_wise[:, mask]

        metrics = {}

        if targets.is_output_a_state_pair():
            mae_pos, mae_vel = targets.split(mae_sample_wise)
            metrics["mae_pos"] = torch.mean(mae_pos).item()
            metrics["mae_vel"] = torch.mean(mae_vel).item()
        else:
            metrics["mae_pos"] = torch.mean(mae_sample_wise).item()
            metrics["mae_vel"] = 0.0

        return metrics
    
@Env.register("CylinderFlowSkipEnv")
class CylinderFlowSkipEnv(CylinderFlowEnv):
    @property
    def trajectory_length(self) -> int:
        return 150

    @property
    def interesting_indices(self) -> Tensor:
        # return [17, 18, 19, 20, 21, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        return [54, 56, 132, 80, 81, 97, 17, 132, 133, 134, 122, 125, 127, 129, 131, 133, 64, 80, 81, 97, 98, 99, 17, 38, 39, 40]
        # return [54, 55, 56, 120, 122, 132, 133, 134, 122, 125, 127, 129, 131, 133, 64, 80, 81, 97, 98, 99, 17, 38, 39, 40]
        # return [56, 57, 37, 120, 134, 122, 125, 127, 129, 131, 133, 64, 80, 81, 97, 98, 99, 17, 38, 39, 40]
        # return [17, 37, 38, 39, 40, 56, 80, 81, 97, 98, 100, 120, 122, 125, 127, 129, 131, 133]
