# PURD-Net

## Overview

Underwater depth estimation in turbid environments is fundamentally limited by strong backscatter and absorption. **PURD-Net** is a unified polarization-guided framework that integrates polarization imaging with monocular depth learning to enable robust 3D reconstruction through scattering media. It takes four polarimetric images (I₀, I₄₅, I₉₀, I₁₃₅) as input and jointly estimates the scene depth map and a backscatter-free recovered image.

## Project Structure

```
.
├── data/
│   └── example_dataset/
│       ├── I0/                    # Polarimetric image at 0°
│       ├── I45/                   # Polarimetric image at 45°
│       ├── I90/                   # Polarimetric image at 90°
│       ├── I135/                  # Polarimetric image at 135°
│       ├── A/                     # Angle of polarization (AoP)
│       ├── J/                     # Object radiance
│       ├── depth/                 # Ground-truth depth maps
│       ├── t/                     # Transmission maps
│       ├── dataset.csv
│       └── test_dataset.csv
├── saved_models/                  # Pre-trained weights (*.pth)
├── depth_estimation/              # Model & training utilities
├── train.py
├── inference.py
├── all.py
├── dependencies.txt
└── LICENSE
```

## Environment

```bash
git clone <repo-url>
cd purd-net

# Create conda environment
conda create -n uwdepth python=3.13 -y
conda activate uwdepth

# Install PyTorch
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# Install remaining dependencies
pip install -r dependencies.txt
```

## Dataset & Pre-trained Weights

| Resource | Link |
|:---|---|
| Simulated polarization dataset | [Baidu Pan](https://pan.baidu.com/s/1XSKCfwOvedFstFa4a3Bm7Q) (code: `qae8`) |
| Pre-trained weights | [Baidu Pan](https://pan.baidu.com/s/1Pw7Ke3lJurCGSoyhgVVrcw) (code: `yzzf`) |

The dataset provides paired I₀, I₄₅, I₉₀, I₁₃₅ frames along with ground-truth depth maps and clean reference images, generated under physically consistent scattering models.

## Train

```bash
python train.py
```

Key hyperparameters (batch size, learning rate, epochs, etc.) are configured at the top of `train.py`. Training logs can be monitored via TensorBoard:

```bash
tensorboard --logdir lr0.0001_bs4_lrd0.9/
```

## Inference

```bash
python inference.py
```

Processes polarimetric image sets (I₀, I₄₅, I₉₀, I₁₃₅) and outputs depth maps, magma visualizations, and recovered images.
