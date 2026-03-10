# Breast cancer attention training

A PyTorch-based project for binary classification of breast cancer pathological images.

The project implements a unified training and comparative evaluation workflow for `CNN`, `ResNet`, and their attention-enhanced versions (`CNN_Attention`, `ResNet_Attention`).

## Features

- Supports one-click training and comparison of 4 models
- Custom attention module (1x1 Conv + BN + Sigmoid)
- Automatically saves model weights (`.pth`)
- Automatically plots training/validation loss and accuracy curves
- Automatically summarizes best metrics to `best_metrics.csv`

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── datasets/
    ├── benign/
    └── malignant/
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Prepare the data directory:

```text
datasets/
  benign/
  malignant/
```

2. Run training:

```bash
python main.py
```

3. After training completes, the following will be generated:
- Model weights for each model: `*.pth`
- Training curves for each model: `*_training.png`
- Summary of best metrics: `best_metrics.csv`

## Current Experiment Snapshot

| Model | Epoch | Train Acc | Val Acc | Val Loss |
|---|---:|---:|---:|---:|
| CNN | 28 | 89.29% | 100.00% | 0.1611 |
| CNN_Attention | 27 | 96.43% | 92.86% | 0.1345 |
| ResNet | 34 | 94.64% | 100.00% | 0.0606 |
| ResNet_Attention | 49 | 100.00% | 100.00% | 0.1040 |

## Notes

- This project currently uses a simple train/val split (`test_size=0.2`); stratified sampling and cross-validation are not enabled.
- For medical image tasks, it is recommended to perform external validation on larger-scale datasets to avoid inflated metrics caused by overfitting.
