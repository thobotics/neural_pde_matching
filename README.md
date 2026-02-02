# PDE Trajectory Matching via port-Hamiltonian Dynamics

## Quick Start

### Prerequisites
- Python 3.10
- [uv](https://docs.astral.sh/uv/) (recommended)

### Installation

```bash
# Create virtual environment and install dependencies
uv venv --python 3.10.13

# Activate environment
source .venv/bin/activate

# Install
uv sync
```

### Verify Setup
After installation, test that everything works:
```bash

# Quick training test (optional)
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
