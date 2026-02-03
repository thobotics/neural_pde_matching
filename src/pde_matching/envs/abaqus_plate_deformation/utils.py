import numpy as np
import meshio
import yaml


def read_vtk_file(file_path):
    return meshio.read(file_path)

def extract_faces_edges(mesh):
    edges = set()
    faces = set()

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            for cell in cell_block.data:
                face = tuple(sorted(cell))
                faces.add(face)
                for i in range(3):
                    edge = tuple(sorted([cell[i], cell[(i + 1) % 3]]))
                    edges.add(edge)
        elif cell_block.type == "tetra":
            for cell in cell_block.data:
                for i in range(4):
                    face = tuple(sorted([cell[j] for j in range(4) if j != i]))
                    faces.add(face)
                    for j in range(3):
                        edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
                        edges.add(edge)
        elif cell_block.type == "quad":
            for cell in cell_block.data:
                face = tuple(sorted(cell))
                faces.add(face)
                for i in range(4):
                    edge = tuple(sorted([cell[i], cell[(i + 1) % 4]]))
                    edges.add(edge)

    edges_list = [list(edge) for edge in edges]
    faces_list = [list(face) for face in faces]

    return edges_list, faces_list

def extract_cells_fields(mesh):
    fields = {}

    # Extract point data fields
    for name, data in mesh.point_data.items():
        fields[name] = data

    # Extract cell data fields
    for name, data in mesh.cell_data.items():
        for cell_type, cell_data in data.items():
            fields[f"{name}_{cell_type}"] = cell_data

    return len(mesh.cells), fields

def extract_geometry(mesh):
    return mesh.points

def parse_yaml_to_numpy(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Extract forces
    forces = data["params"]["forces"]

    # Preparing a structured NumPy array to hold force data
    force_dtype = np.dtype(
        [
            ("name", "U10"),  # Assuming force names are strings of up to 10 characters
            ("direction", "f8", (3,)),
            ("position", "f8", (3,)),
            ("length", "f8"),
            (
                "search_mode",
                "U10",
            ),  # Assuming search_mode strings are up to 10 characters
        ]
    )

    force_array = np.zeros(len(forces), dtype=force_dtype)

    for i, (force_name, force_info) in enumerate(forces.items()):
        force_array[i]["name"] = force_name
        force_array[i]["direction"] = force_info["direction"]
        force_array[i]["position"] = force_info["position"]
        force_array[i]["length"] = force_info["force_application"]["length"]
        force_array[i]["search_mode"] = force_info["force_application"]["search_mode"]

    return force_array
