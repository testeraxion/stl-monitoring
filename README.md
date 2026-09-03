# 🛡️ Uncertainty-Aware Predictive STL Monitoring of RL Locomotion Policies Under Distribution Shift

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026%20Workshop-red.svg)](https://neurips.cc/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.0-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-v5-purple.svg)](https://mujoco.org/)

---

## 📖 TL;DR

We show that **ensemble uncertainty** in learned world models is more useful for **diagnosing predictive model error** than for **directly predicting future safety violations** in RL locomotion policies under distribution shift.

---

## 🎯 Overview

This repository provides the official implementation of the uncertainty-aware predictive STL monitoring framework evaluated on MuJoCo Ant-v5 and HalfCheetah-v5 under controlled distribution shifts (external forces, observation noise, mass variation).

### Key Components

| Component | Description |
|-----------|-------------|
| 🌍 **Learned World Model** | Feedforward encoder-decoder with latent dynamics for short-horizon future prediction |
| 📊 **STL Monitoring** | Signal Temporal Logic robustness evaluation on predicted trajectories |
| 🔮 **Ensemble Uncertainty** | Deep ensemble disagreement quantified in STL-robustness space |
| ⚠️ **Uncertainty-Penalized Score** | Conservative safety estimate combining predictive robustness and uncertainty |

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/RL_policy.git
cd RL_policy

# Install dependencies
pip install -e .

# Verify installation
python -c "import stl_safety_monitor; print('✅ Installation successful')"
```

---

## 🚀 Quick Start

### 1. Train World Model Ensemble

```bash
python scripts/train_ensemble.py --env Ant-v5 --ensemble-size 3 --episodes 50
```

### 2. Run Predictive Monitoring Evaluation

```bash
python scripts/predictive_ensemble_eval.py \
    --env Ant-v5 \
    --horizon 10 \
    --conditions baseline force_15N noise_0.15
```

### 3. Generate Paper Results

```bash
# Generate tables
python scripts/generate_tables.py

# Generate figures
python scripts/generate_figures.py
```

---

## 📁 Repository Structure

```
stl-monitoring/
├── src/
│   ├── monitors/
│   │   ├── stl_monitor.py          # Core STL monitoring
│   │   └── baselines.py            # Threshold & barrier baselines
│   ├── environments/
│   │   └── locomotion_wrapper.py   # MuJoCo environment wrapper
│   └── evaluation/
│       ├── distribution_shift.py   # Shift evaluation utilities
│       └── evaluate_monitors.py    # Monitor evaluation
├── scripts/
│   ├── train_ensemble.py           # Train world model ensemble
│   ├── world_model_train.py        # Single world model training
│   ├── predictive_ensemble_eval.py # Main evaluation script
│   ├── predictive_stl_monitor.py   # Predictive STL monitoring
│   ├── generate_figures.py         # Paper figures
│   └── generate_tables.py          # Paper tables
├── configs/
│   ├── stl_properties.yaml         # Ant-v5 STL specifications
│   ├── stl_properties_halfcheetah.yaml  # HalfCheetah-v5 STL specs
│   ├── perturbation.yaml           # Distribution shift conditions
│   └── training.yaml               # Training configuration
└── Readme.md               # This file
```

---

## 📊 Experimental Results

### Distribution Shift Conditions

| Environment | Shift Type | Conditions |
|-------------|------------|------------|
| Ant-v5 | External Force | 5N, 15N, 30N |
| Ant-v5 | Observation Noise | σ = 0.01, 0.05, 0.15 |
| Ant-v5 | Mass Variation | 0.8×, 1.3×, 1.6× |
| HalfCheetah-v5 | External Force | 5N, 15N, 30N |
| HalfCheetah-v5 | Observation Noise | σ = 0.01, 0.05, 0.15 |
| HalfCheetah-v5 | Mass Variation | 0.8×, 1.3×, 1.6× |

### Key Findings

1. **Predictive STL exhibits environment-dependent bias** — mildly pessimistic for Ant-v5, systematically optimistic for HalfCheetah-v5
2. **Ensemble uncertainty does not consistently track prediction error** — direction varies across shifts and environments
3. **Uncertainty does not reliably predict future STL violations** — despite substantial violation rates in HalfCheetah-v5 (up to 66.7%)

---

## ⚙️ Configuration

### STL Properties (`configs/stl_properties.yaml`)

```yaml
properties:
  - name: "roll"
    formula: "G[0,500](|roll| < 1.0)"
    weight: 1.0
  - name: "pitch"
    formula: "G[0,500](|pitch| < 1.0)"
    weight: 1.0
  - name: "height"
    formula: "G[0,500](height > 0.4)"
    weight: 0.8
  - name: "airborne"
    formula: "G[0,500](airborne <= 3)"
    weight: 0.5
  - name: "velocity"
    formula: "G[0,500](|v_fwd| < 4.0)"
    weight: 1.0
```

### Training (`configs/training.yaml`)

```yaml
ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2

world_model:
  latent_dim: 128
  hidden_dim: 256
  ensemble_size: 3
  training_epochs: 30
  learning_rate: 1e-3
  batch_size: 256
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{uncertainty_stl_2026,
  title={Uncertainty-Aware Predictive STL Monitoring of RL Locomotion Policies Under Distribution Shift},
  author={Anonymous Authors},
  booktitle={NeurIPS 2026 Workshop on Robot Learning with World Models},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL implementations
- [MuJoCo](https://mujoco.org/) for physics simulation
- [RTAMT](https://github.com/yalctse/rtamt) for STL monitoring

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

<p align="center">
  <i>Built with ❤️ for safe robot learning</i>
</p>
