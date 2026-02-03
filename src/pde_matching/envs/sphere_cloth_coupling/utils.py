import numpy as np
import torch


def faces_to_edges(faces):
    """Convert face indices to edge indices."""
    edges = set()
    
    for face in faces:
        n_vertices = len(face)
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                edge = tuple(sorted([face[i], face[j]]))
                edges.add(edge)
    
    return np.array(list(edges))


def create_triangular_cloth_grid(n_rows, n_cols):
    """Create triangular mesh connectivity for a regular grid."""
    faces = []
    
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            # Bottom-left vertex of the quad
            v0 = i * n_cols + j
            v1 = i * n_cols + (j + 1)
            v2 = (i + 1) * n_cols + j
            v3 = (i + 1) * n_cols + (j + 1)
            
            # Two triangular faces per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return np.array(faces)


def create_triangular_cloth_edges(n_rows, n_cols):
    """Create edge connectivity for triangular cloth mesh."""
    edges = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            
            # Right neighbor
            if j < n_cols - 1:
                edges.append([idx, idx + 1])
            
            # Bottom neighbor
            if i < n_rows - 1:
                edges.append([idx, idx + n_cols])
            
            # Diagonal neighbors for triangular mesh
            if i < n_rows - 1 and j < n_cols - 1:
                edges.append([idx, idx + n_cols + 1])
            if i < n_rows - 1 and j > 0:
                edges.append([idx, idx + n_cols - 1])
    
    return np.array(edges)


def create_quad_cloth_grid(n_rows, n_cols):
    """Create quad mesh connectivity for a regular grid."""
    faces = []
    
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            # Bottom-left vertex of the quad
            v0 = i * n_cols + j
            v1 = i * n_cols + (j + 1)
            v2 = (i + 1) * n_cols + j
            v3 = (i + 1) * n_cols + (j + 1)
            
            # Single quad face
            faces.append([v0, v1, v3, v2])
    
    return np.array(faces)


def create_quad_cloth_edges(n_rows, n_cols):
    """Create edge connectivity for quad cloth mesh."""
    edges = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            
            # Right neighbor
            if j < n_cols - 1:
                edges.append([idx, idx + 1])
            
            # Bottom neighbor
            if i < n_rows - 1:
                edges.append([idx, idx + n_cols])
    
    return np.array(edges)


def compute_sphere_center_connections(sphere_center_pos, cloth_positions, radius):
    """Compute connections between sphere center and cloth nodes within radius."""
    connections = []
    sphere_center_xy = sphere_center_pos[:2]
    
    for i, cloth_pos in enumerate(cloth_positions):
        cloth_xy = cloth_pos[:2]
        dist_xy = np.linalg.norm(sphere_center_xy - cloth_xy)
        if dist_xy <= radius:
            connections.append(i)
    
    return connections


def create_cloth_grid(n_rows, n_cols, mesh_type="triangular"):
    """Create cloth mesh connectivity for a regular grid.
    
    Args:
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        mesh_type: "triangular" or "quad"
    """
    if mesh_type == "triangular":
        return create_triangular_cloth_grid(n_rows, n_cols)
    elif mesh_type == "quad":
        return create_quad_cloth_grid(n_rows, n_cols)
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}")


def create_cloth_edges(n_rows, n_cols, mesh_type="triangular"):
    """Create edge connectivity for cloth mesh.
    
    Args:
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        mesh_type: "triangular" or "quad"
    """
    if mesh_type == "triangular":
        return create_triangular_cloth_edges(n_rows, n_cols)
    elif mesh_type == "quad":
        return create_quad_cloth_edges(n_rows, n_cols)
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}")


# ...existing code...
