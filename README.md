# Breast Pathology Attention Benchmark

基于 PyTorch 的乳腺癌病理图像二分类项目。  
项目实现了 `CNN`、`ResNet` 及其注意力增强版本（`CNN_Attention`、`ResNet_Attention`）的统一训练与对比评估流程。

## Features

- 支持 4 种模型的一键训练与对比
- 自定义注意力模块（1x1 Conv + BN + Sigmoid）
- 自动保存模型权重（`.pth`）
- 自动绘制训练/验证损失与准确率曲线
- 自动汇总最优指标到 `best_metrics.csv`

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── datasets/              # 本地数据目录，默认不纳入 Git
    ├── benign/
    └── malignant/
```

## Environment

安装依赖：

```bash
pip install -r requirements.txt
```

## Quick Start

1. 准备数据目录：

```text
datasets/
  benign/
  malignant/
```

2. 运行训练：

```bash
python main.py
```

3. 训练完成后将生成：
- 各模型权重：`*.pth`
- 各模型训练曲线：`*_training.png`
- 最优指标汇总：`best_metrics.csv`

## Current Experiment Snapshot

| Model | Epoch | Train Acc | Val Acc | Val Loss |
|---|---:|---:|---:|---:|
| CNN | 28 | 89.29% | 100.00% | 0.1611 |
| CNN_Attention | 27 | 96.43% | 92.86% | 0.1345 |
| ResNet | 34 | 94.64% | 100.00% | 0.0606 |
| ResNet_Attention | 49 | 100.00% | 100.00% | 0.1040 |

## Notes

- 该项目当前使用简单 train/val 切分（`test_size=0.2`），未启用分层抽样与交叉验证。
- 医学图像任务建议在更大规模数据集上进行外部验证，以避免过拟合导致的指标偏高。
