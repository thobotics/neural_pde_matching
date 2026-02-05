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

| Dataset | Source |
|---------|--------|
| **PlateDeformation**      | [Zenodo](https://zenodo.org/records/18480131) | 
| **SphereCloth**           | [Zenodo](https://zenodo.org/records/18480131) | 
| **WaveBalls**            | [Zenodo](https://zenodo.org/records/18480131) | 
| **Kuramoto-Sivashinsky**  | [Zenodo](https://zenodo.org/records/18480131) | 
| **Impact Plate**          | [HCMT](https://github.com/yuyudeep/hcmt) | 
| **Cylinder Flow**         | [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | 

*Note: These are raw datasets and should be stored under `./dataset_public/DATASET_NAME/raw`*

**Verify:**
```bash
python main.py -cn experiment/abaqus_plate_deformation_matching trainer.max_epochs=1 dataset.train_n_sequences=5
```

---

## Usage

```bash
# Specific environments
python main.py -cn experiment/abaqus_plate_deformation_matching
python main.py -cn experiment/sphere_cloth_matching
python main.py -cn experiment/wave_balls_matching
python main.py -cn experiment/kuramoto_matching
python main.py -cn experiment/impact_plate_matching
python main.py -cn experiment/cylinder_flow_matching
```
