# Breast Pathology Attention Benchmark

推荐 GitHub 仓库名：`breast-pathology-attention-benchmark`

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

- Python 3.9+
- 建议使用 GPU（自动检测 CUDA，不可用时回退到 CPU）

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

上述数据集、权重和训练产物默认已被 `.gitignore` 排除，便于直接发布到 GitHub。

## Current Experiment Snapshot

当前本地实验结果（来自 `best_metrics.csv`，默认不提交 Git）：

| Model | Epoch | Train Acc | Val Acc | Val Loss |
|---|---:|---:|---:|---:|
| CNN | 28 | 89.29% | 100.00% | 0.1611 |
| CNN_Attention | 27 | 96.43% | 92.86% | 0.1345 |
| ResNet | 34 | 94.64% | 100.00% | 0.0606 |
| ResNet_Attention | 49 | 100.00% | 100.00% | 0.1040 |

数据样本数（当前仓库）：`70`（benign: `22`, malignant: `48`）。

## Notes

- 该项目当前使用简单 train/val 切分（`test_size=0.2`），未启用分层抽样与交叉验证。
- 医学图像任务建议在更大规模数据集上进行外部验证，以避免过拟合导致的指标偏高。
- 发布开源时，请确保数据集使用与再分发符合原始数据许可。
- 若需要在 GitHub 展示结果，建议单独挑选 1 份示例图表放入 `assets/` 目录，而不是直接提交全部训练产物。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
