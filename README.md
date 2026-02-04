# PDE Trajectory Matching via port-Hamiltonian Dynamics

---

## Quick Start

### Prerequisites
- Python 3.10.13
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
uv venv --python 3.10.13
source .venv/bin/activate
uv sync
```

---

## Dataset Setup

Download and extract datasets to `./dataset_public/`:

| Dataset | Source | Path |
|---------|--------|------|
| **Main** (Plate, Sphere Cloth, Wave, Kuramoto) | [Zenodo](https://zenodo.org/records/18480131) | `./dataset_public/` |
| **Impact Plate** (optional) | [HCMT](https://github.com/yuyudeep/hcmt) | `./dataset_public/impact_plate/raw/` |
| **Cylinder Flow** (optional) | [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | `./dataset_public/cylinder_flow/raw/` |

**Verify:**
```bash
python main.py -cn experiment/abaqus_plate_deformation_matching trainer.max_epochs=1 dataset.train_n_sequences=5
```

---

## Usage

```bash
# Default config
python main.py

# Specific environments
python main.py -cn experiment/abaqus_plate_deformation_matching
python main.py -cn experiment/cylinder_flow_matching
python main.py -cn experiment/wave_fluid_matching
python main.py -cn experiment/sphere_cloth_matching
```
