# PDE Trajectory Matching via port-Hamiltonian Dynamics

## Quick Start

### Prerequisites
- Python 3.10.13
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
# Create virtual environment and install dependencies
uv venv --python 3.10.13
source .venv/bin/activate
uv sync
```

### Dataset Setup

Download and extract datasets to `./dataset_public/`:

**Main dataset:** [Zenodo](https://zenodo.org/records/18480131) - Plate Deformation, Sphere Cloth, Wave Balls, and Kuramoto-Sivashinsky  
- Extract to: `./dataset_public/`

**Impact plate (optional):** [HCMT](https://github.com/yuyudeep/hcmt)
- Extract raw data to: `./dataset_public/impact_plate/raw/`

**Cylinder flow (optional):** [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- Extract raw data to: `./dataset_public/cylinder_flow/raw/`

### Verify Setup

```bash
# Quick training test
python main.py -cn experiment/abaqus_plate_deformation_matching trainer.max_epochs=1 dataset.train_n_sequences=5
```

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
