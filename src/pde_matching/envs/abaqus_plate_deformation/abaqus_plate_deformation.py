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
from pde_matching.envs.abaqus_plate_deformation.abaqus_plate_deformation_visualizer import (
    AbaqusPlateDeformationVisualizer,
)
from torch import Tensor
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import Cartesian, Compose, Distance
from torch_geometric.nn import knn
from tqdm import tqdm

from ..env import Env
from ..trajectory_dataset import TrajectoryDatasetMixin
from ..utils import connect_two_graphs, EdgeFeatureScaler
from .utils import (
    extract_cells_fields,
    extract_faces_edges,
    extract_geometry,
    parse_yaml_to_numpy,
    read_vtk_file,
)


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    BOUNDARY = 2
    SIZE = 3


class EdgeType(enum.IntEnum):
    NORMAL_NORMAL = 0
    OBSTACLE_OBSTACLE = 1
    NORMAL_OBSTACLE = 2
    OBSTACLE_NORMAL = 3
    BOUNDARY_NORMAL = 4
    SIZE = 5
    

@Env.register("AbaqusPlateDeformationEnv")
class AbaqusPlateDeformationEnv(TrajectoryDatasetMixin, Env):
    def __init__(
        self,
        root: str,
        window_size: int,
        window_shift: int,
        n_sequences: int = None,
        stage: str = "train",
        log: bool = True,
        connection_radius: float = 0.3,
        input_keys: List[str] = None,
        output_keys: List[str] = None,
        training_noise: bool = False,
        training_noise_std: float = 0.0,
        standardize_by_trajectory: bool = False,
        boundary_k: int = 0,
        single_step: bool = False,
        **kwargs,
    ):
        # Env specific parameters
        self._max_force = 400.0  # Maximum force applied to the plate
        self._max_plate_size = 140.0  # Maximum size of the plate

        transform = Compose(
            [
                EdgeCategorical(EdgeType.SIZE),
                Cartesian(norm=False),
                Distance(norm=False),
                EdgeFeatureScaler(
                    start_idx=EdgeType.SIZE, end_idx=EdgeType.SIZE + 4, scale_factor=1.0 / self._max_force  # Scale both the cartesian and distance features
                ),
            ]
        )

        self.cache = {"idx": -1}
        self.stage = stage
        self.single_step = single_step
        self.training_noise = training_noise
        self.training_noise_std = training_noise_std
        self._standardize_by_trajectory = standardize_by_trajectory
        self._n_sequences = n_sequences
        self._n_total_sequences = None
        self._standardizer = None
        self._visualizer = None
        self._connection_radius = connection_radius
        self._input_keys = input_keys
        self._output_keys = output_keys
        self._boundary_k = boundary_k
        self._visualizer_kwargs = kwargs.pop("visualizer", {})

        self._train_size = 800
        self._val_size = 100
        self._test_size = 100

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
        return 49

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def n_total_sequences(self) -> int:
        if self._n_total_sequences is None:
            if self.stage == "train":
                self._n_total_sequences = self._train_size
            elif self.stage == "val":
                self._n_total_sequences = self._val_size
            elif self.stage == "test":
                self._n_total_sequences = self._test_size

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
            self._visualizer = AbaqusPlateDeformationVisualizer(
                self.trajectory_length, **self._visualizer_kwargs
            )
        return self._visualizer

    @property
    def connection_radius(self) -> float:
        return self._connection_radius

    @property
    def boundary_k(self) -> int:
        return self._boundary_k
    
    @property
    def node_type_dim(self) -> int:
        return NodeType.SIZE
    
    @property
    def interesting_indices(self) -> Tensor:
        return [
            5, 20, 16, 15, 5, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
            35, 36, 37, 38, 39, 40
        ]

    def loss_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        mask = torch.argmax(input, dim=-1) == NodeType.NORMAL
        mask |= torch.argmax(input, dim=-1) == NodeType.BOUNDARY
        return mask

    def output_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        return torch.argmax(input, dim=-1) == NodeType.NORMAL
    
    def control_params_mask(self, input: Tensor) -> Tensor:
        if input.ndim == 3:
            input = input[0]
        input = input[..., :NodeType.SIZE]
        return torch.argmax(input, dim=-1) == NodeType.NORMAL

    def process(self):
        for stage in ["train", "val", "test"]:
            self._process_by_stage(stage)

    def _process_by_stage(self, stage):
        stage_dir = Path(self.processed_dir) / stage
        stage_dir.mkdir(exist_ok=True)

        raw_dirs = self._generate_raw_dirs_by_stage(stage)

        for traj_idx, raw_dir in enumerate(
            tqdm(raw_dirs, desc=f"Processing {stage}", unit=" trajectories")
        ):

            data_list = []
            fields_list = []

            u_stats = MeanStdAccumulator()
            u_dot_stats = MeanStdAccumulator()
            u_dot_dot_stats = MeanStdAccumulator()

            # +2 for u_dot and u_dot_dot
            for t in range(self.trajectory_length + 2):
                file_path = os.path.join(raw_dir, "PLY-1", f"PLY-1-{t}.vtk")
                poly_data = read_vtk_file(file_path)

                if t == 0:
                    edges, faces = extract_faces_edges(poly_data)
                    force_array = parse_yaml_to_numpy(
                        os.path.join(raw_dir, "Parameter.yaml")
                    )

                num_points, fields = extract_cells_fields(poly_data)
                fields_list.append(fields)

                positions = extract_geometry(poly_data)

                # Create the data object for the mesh and the collider
                mesh = Data(
                    face=torch.tensor(faces, dtype=torch.long).T,
                    pos=torch.from_numpy(positions).float(),
                    edge_index=to_undirected(torch.tensor(edges, dtype=torch.long).T),
                )

                applied_forces = Data(
                    pos=torch.from_numpy(force_array["position"]).float(),
                    edge_index=torch.tensor(
                        [[i, i] for i in range(len(force_array))],
                        dtype=torch.long,
                    ).T,
                )

                if t == 0:
                    pos_tensor = mesh.pos
                    condition = ~(
                        (pos_tensor[:, 0] > 1)
                        & (pos_tensor[:, 0] < self._max_plate_size - 1)
                        & (pos_tensor[:, 1] > 1)
                        & (pos_tensor[:, 1] < self._max_plate_size - 1)
                    )
                    condition = torch.cat(
                        [condition, torch.zeros(applied_forces.num_nodes, dtype=torch.bool)], dim=0
                    )

                data = connect_two_graphs(
                    mesh,
                    applied_forces,
                    node_type=NodeType,
                    edge_type=EdgeType,
                    radius=force_array["length"][0] / 2,
                    # knn_subsample=2,
                    max_num_neighbors=mesh.pos.shape[0],
                    collider_to_collider_edges=False,
                    mesh_to_collider_edges=False,
                    collider_to_mesh_edges=True,
                )

                # Extract displacement field from VTK data (only for mesh nodes)
                field = fields_list[t]["U"]
                
                # Pad displacement field to match total number of nodes (mesh + applied forces)
                displacement_field = torch.cat([
                    torch.from_numpy(field),  # Displacement for mesh nodes
                    torch.zeros((len(data.pos) - len(field), 3), dtype=torch.float)  # Zero displacement for applied force nodes
                ], dim=0)

                # 6D node features: positions [3D] + zero displacement [3D]
                data.u = torch.cat(
                    [
                        data.pos, # Position of mesh nodes
                        displacement_field,  # Displacement field for mesh nodes
                    ],
                    dim=1,
                ).float()

                # Node properties
                node_type_onehot = torch.eye(NodeType.SIZE)[data.node_type]
                
                # Create force direction features: zero for mesh nodes, actual values for applied force nodes
                mesh_force_features = torch.zeros((len(mesh.pos), 3), dtype=torch.float)
                applied_force_features = torch.from_numpy(force_array["direction"])
                force_features = torch.cat([mesh_force_features, applied_force_features], dim=0)
                
                # Combine node type and force features
                data.properties = torch.cat([node_type_onehot, force_features], dim=1).float()

                # Set boundary node properties: boundary type one-hot + zero force features
                boundary_type_onehot = torch.nn.functional.one_hot(
                    torch.tensor(NodeType.BOUNDARY), num_classes=NodeType.SIZE
                )
                boundary_force_features = torch.zeros(3, dtype=torch.float)
                boundary_properties = torch.cat([boundary_type_onehot, boundary_force_features])
                data.properties[condition] = boundary_properties

                # Set node types for boundary nodes
                data.node_type[condition] = NodeType.BOUNDARY

                # Add KNN connections from boundary nodes to mesh nodes
                if t == 0 and self.boundary_k > 0:
                    # Find boundary nodes (only mesh nodes, not applied forces)
                    boundary_nodes = torch.where(condition[:len(mesh.pos)])[0]
                    mesh_nodes = torch.where(~condition[:len(mesh.pos)])[0]
                    
                    if len(boundary_nodes) > 0 and len(mesh_nodes) > 0:
                        # Get positions
                        boundary_pos = mesh.pos[boundary_nodes]
                        mesh_node_pos = mesh.pos[mesh_nodes]
                        
                        # Find KNN connections from boundary to mesh nodes
                        k_val = min(self.boundary_k, len(mesh_nodes))
                        knn_edges = knn(mesh_node_pos, boundary_pos, k=k_val)
                        
                        # Convert to global indices
                        knn_edges_adjusted = torch.stack([
                            boundary_nodes[knn_edges[0]],  # source: boundary nodes
                            mesh_nodes[knn_edges[1]]       # target: mesh nodes
                        ])
                        
                        # Add edges to the graph
                        data.edge_index = torch.cat([data.edge_index, knn_edges_adjusted], dim=1)
                        data.edge_type = torch.cat([
                            data.edge_type,
                            torch.ones(knn_edges_adjusted.shape[1], dtype=torch.long) * EdgeType.BOUNDARY_NORMAL
                        ])

                # We need mesh and collider faces for visualization
                data.mesh_face = mesh.face

                data_list.append(data)

                # Compute neccessary quantities for standardization
                u_stats.add(data.u.detach().cpu().numpy())

            # Compute velocities
            for t, data in enumerate(data_list):
                if t == len(data_list) - 1:
                    # For the last timestep, copy velocity from previous timestep
                    data.u_dot = data_list[t - 1].u_dot
                else:
                    data.u_dot = (data_list[t + 1].u - data.u) / self.dt

                u_dot_stats.add(data.u_dot.detach().cpu().numpy())

            # Compute accelerations
            for t, data in enumerate(data_list):
                if t == len(data_list) - 1:
                    data.u_dot_dot = data_list[t - 1].u_dot_dot
                else:
                    data.u_dot_dot = (data_list[t + 1].u_dot - data.u_dot) / self.dt

                u_dot_dot_stats.add(data.u_dot_dot.detach().cpu().numpy())

            torch.save(data_list, stage_dir / f"{traj_idx}.pt")

        processed_file = Path(stage_dir / "processed")
        processed_file.touch()

        u_mean, u_std = u_stats.mean_and_std()
        u_dot_mean, u_dot_std = u_dot_stats.mean_and_std()
        u_dot_dot_mean, u_dot_dot_std = u_dot_dot_stats.mean_and_std()

        if stage == "train":
            np.savez(
                self.stats_file,
                u_mean=u_mean,
                u_std=u_std,
                u_dot_mean=u_dot_mean,
                u_dot_std=u_dot_std,
                u_dot_dot_mean=u_dot_dot_mean,
                u_dot_dot_std=u_dot_dot_std,
            )

    def _generate_raw_dirs_by_stage(self, stage):
        raw_dirs = []
        if stage == "train":
            for i in range(*self.raw_dirs_range_tuple[0]):
                raw_dirs.append(Path(self.raw_dir) / f"Sim{i:03d}")
        elif stage == "val":
            for i in range(*self.raw_dirs_range_tuple[1]):
                raw_dirs.append(Path(self.raw_dir) / f"Sim{i:03d}")
        elif stage == "test":
            for i in range(*self.raw_dirs_range_tuple[2]):
                raw_dirs.append(Path(self.raw_dir) / f"Sim{i:03d}")

        return raw_dirs
