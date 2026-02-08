<div align="center">

# Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/Conferences/2026)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-red.svg)](https://openreview.net/forum?id=x66u6TEDUw)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange.svg)](https://zenodo.org/records/18480131)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-green.svg)](https://thobotics.github.io/neural_pde_matching/)

**[Tai Hoang](https://thobotics.github.io/)<sup>1</sup>, [Alessandro Trenta](https://alexthirty.github.io/)<sup>\*,2</sup>, [Alessio Gravina](https://pages.di.unipi.it/gravina/)<sup>\*,2</sup>, [Niklas Freymuth](https://nfreymuth.com/)<sup>1</sup>, [Philipp Becker](https://pbecker93.github.io/)<sup>1</sup>, [Davide Bacciu](https://pages.di.unipi.it/bacciu/)<sup>2</sup>, [Gerhard Neumann](https://alr.iar.kit.edu/21_65.php)<sup>1</sup>**

<sup>1</sup>Karlsruhe Institute of Technology (KIT) | <sup>2</sup>University of Pisa | <sup>*</sup>Equal contribution

*Published at the International Conference on Learning Representations (ICLR) 2026*

</div>

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
| **WaveBalls**             | [Zenodo](https://zenodo.org/records/18480131) | 
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

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
  hoang2026igns,
  title={Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics},
  author={Hoang, Tai and Trenta, Alessandro and Gravina, Alessio and 
          Freymuth, Niklas and Becker, Philipp and Bacciu, Davide and Neumann, Gerhard},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=x66u6TEDUw}
}
```

---

<div align="center">

**[Website](https://thobotics.github.io/neural_pde_matching/) | [Paper](https://openreview.net/forum?id=x66u6TEDUw) | [Dataset](https://zenodo.org/records/18480131)**

</div>
