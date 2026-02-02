from pathlib import Path
from typing import List, Dict

import torch
import torch_cluster
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform


@functional_transform('scale_edge_attr')
class EdgeFeatureScaler(BaseTransform):
    def __init__(self, start_idx: int, end_idx: int, scale_factor: float):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.scale_factor = scale_factor

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_attr'):
            data.edge_attr[:, self.start_idx:self.end_idx] *= self.scale_factor
        return data


def plot_graph(pos, edge_index, ax=None, c="blue", width=1):
    """
    Quick visualization of incoming (edge_index_in) and outgoing (edge_index_out) graphs.

    Args:
        pos: (N, 2) numpy array of node positions.
        edge_index: (2, E_in) numpy array of edges for the 'incoming' graph.
        ax: optional matplotlib axis.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # incoming edges
    for i, j in edge_index.T:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=c, alpha=0.5, linewidth=width)

    # draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], c="black", s=20, zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def download_file(url: str, to: Path):
    import requests

    response = requests.get(url, stream=True)
    if "Content-Length" in response.headers:
        total = int(response.headers["Content-Length"])
    else:
        total = 0
    with to.open("wb") as f:
        pbar = tqdm(
            response.iter_content(chunk_size=10**5),
            desc=to.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        for data in pbar:
            f.write(data)
            pbar.update(len(data))


def connect_two_graphs(
    mesh: Data,
    collider: Data,
    node_type,
    edge_type,
    radius: float = None,
    max_num_neighbors: int = 100,
    knn_subsample: int = 0,
    collider_to_collider_edges: bool = True,
    mesh_to_collider_edges: bool = True,
    collider_to_mesh_edges: bool = True,
    return_connected_edges: bool = False,
    device: torch.device = torch.device("cpu"),
):
    # Currently, torch_cluster.radius only supports CPU tensors
    mesh = mesh.to(device)
    collider = collider.to(device)

    # Compute the edge indices between mesh and collider
    if radius is None:
        knn = max_num_neighbors if knn_subsample == 0 else knn_subsample * max_num_neighbors
        connected_edge_index = torch_cluster.knn(
            x=mesh.pos, y=collider.pos, k=knn
        )
        connected_edge_index = connected_edge_index[:, ::knn_subsample]
    else:
        connected_edge_index = torch_cluster.radius(
            x=mesh.pos, y=collider.pos, r=radius, max_num_neighbors=max_num_neighbors
        )

    # Adjust the indices to match the combined graph
    connected_edge_index[0] = connected_edge_index[0] + mesh.num_nodes

    # Adjust the indices of collider.edge_index to match the combined graph
    mesh_edge_index = mesh.edge_index
    collider_edge_index = collider.edge_index + mesh.num_nodes

    # Merge the edge indices while avoiding duplications
    num_connected_edges = connected_edge_index.shape[1]
    reversed_edge_index = torch.stack(
        [connected_edge_index[1], connected_edge_index[0]]
    )

    edge_index = mesh_edge_index
    if collider_to_collider_edges:
        edge_index = torch.cat([edge_index, collider_edge_index], dim=1)
    if mesh_to_collider_edges:
        edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    if collider_to_mesh_edges:
        edge_index = torch.cat([edge_index, connected_edge_index], dim=1)

    # Concatenate the node positions and create the connected graph
    combined_pos = torch.cat([mesh.pos, collider.pos])
    connected_graph = Data(pos=combined_pos, edge_index=edge_index)

    connected_graph.node_type = torch.cat(
        [
            torch.ones(mesh.num_nodes, dtype=torch.long, device=device)
            * node_type.NORMAL,
            torch.ones(collider.num_nodes, dtype=torch.long, device=device)
            * node_type.OBSTACLE,
        ]
    )

    connected_graph.edge_type = torch.ones(mesh.num_edges, dtype=torch.long, device=device) * edge_type.NORMAL_NORMAL
    if collider_to_collider_edges:
        connected_graph.edge_type = torch.cat(
            [connected_graph.edge_type, torch.ones(collider.num_edges, dtype=torch.long, device=device) * edge_type.OBSTACLE_OBSTACLE]
        )
    
    if mesh_to_collider_edges:
        connected_graph.edge_type = torch.cat(
            [connected_graph.edge_type, torch.ones(num_connected_edges, dtype=torch.long, device=device) * edge_type.NORMAL_OBSTACLE]
        )

    if collider_to_mesh_edges:
        connected_graph.edge_type = torch.cat(
            [connected_graph.edge_type, torch.ones(num_connected_edges, dtype=torch.long, device=device) * edge_type.OBSTACLE_NORMAL]
        )

    if return_connected_edges:
        return connected_graph, connected_edge_index
    else:
        return connected_graph


def extract_mesh_and_collider_from_connected_graph(
    graph: Data, node_type: torch.Tensor, NodeType: type
) -> tuple[Data, Data]:
    # Extract the collider and mesh nodes
    collider_nodes = graph.pos[node_type == NodeType.OBSTACLE]
    mesh_nodes = graph.pos[node_type == NodeType.NORMAL]

    # Extract the collider and mesh edges
    collider_edge_index = graph.edge_index[
        :,
        (node_type[graph.edge_index[0]] == NodeType.OBSTACLE)
        & (node_type[graph.edge_index[1]] == NodeType.OBSTACLE),
    ]
    collider_edge_index -= mesh_nodes.shape[0]
    mesh_edge_index = graph.edge_index[
        :,
        (node_type[graph.edge_index[0]] == NodeType.NORMAL)
        & (node_type[graph.edge_index[1]] == NodeType.NORMAL),
    ]

    # Create the collider and mesh graphs
    collider = Data(pos=collider_nodes, edge_index=collider_edge_index)
    mesh = Data(pos=mesh_nodes, edge_index=mesh_edge_index)

    return mesh, collider


def extract_edges_from_tetrahedra(flattened_indices):
    # Extract unique edges from tetrahedral indices
    edges = set()

    for i in range(0, len(flattened_indices), 4):
        tetra = sorted(flattened_indices[i : i + 4])
        tetra_edges = [
            (tetra[0], tetra[1]),
            (tetra[1], tetra[0]),
            (tetra[0], tetra[2]),
            (tetra[2], tetra[0]),
            (tetra[0], tetra[3]),
            (tetra[3], tetra[0]),
            (tetra[1], tetra[2]),
            (tetra[2], tetra[1]),
            (tetra[1], tetra[3]),
            (tetra[3], tetra[1]),
            (tetra[2], tetra[3]),
            (tetra[3], tetra[2]),
        ]
        edges.update(tetra_edges)

    return torch.tensor(list(edges)).t()


def extract_boundary_faces(indices):
    # Define tetrahedron faces
    tetra_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    all_faces = []
    for i in range(0, len(indices), 4):
        tetra = indices[i : i + 4]
        for face in tetra_faces:
            all_faces.append(sorted([tetra[face[0]], tetra[face[1]], tetra[face[2]]]))

    # Find boundary faces
    idx_boundary_faces = []
    boundary_faces = []
    for i, face in enumerate(all_faces):
        if (
            all_faces.count(face) == 1
        ):  # if a face appears only once, it's a boundary face
            boundary_faces.append(face)
            idx_boundary_faces.append(i)

    all_faces = torch.tensor(all_faces).t()
    boundary_faces = torch.tensor(boundary_faces).t()
    idx_boundary_faces = torch.tensor(idx_boundary_faces)

    return all_faces, boundary_faces, idx_boundary_faces


def extract_triangular_faces(indices):
    """Extract triangular faces from triangular face indices.
    
    Args:
        indices: Flattened array of triangular face indices
        
    Returns:
        all_faces: Tensor of shape (3, num_faces) containing all triangular faces
    """
    all_faces = []
    for i in range(0, len(indices), 3):
        triangle = indices[i:i + 3]
        all_faces.append([triangle[0], triangle[1], triangle[2]])

    return torch.tensor(all_faces)
