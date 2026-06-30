# I-JEPA for Contact Network Anomaly Detection

基于 I-JEPA (Image-based Joint-Embedding Predictive Architecture) 自监督学习的接触网零部件图像异常检测系统。

## 方法概述

1. **I-JEPA 预训练**：在全部类别的正常样本上进行自监督预测任务学习
2. **特征提取**：用预训练的 context encoder 提取多尺度 patch features
3. **异常检测**：对正常特征的 patch 级别高斯分布建模（PaDiM 风格），用马氏距离作为异常分数

## 项目结构

```
ijepa/
├── config/
│   ├── ijepa_pretrain.yaml      # 预训练配置
│   └── anomaly_detection.yaml   # 异常检测配置
├── src/
│   ├── data/                    # 数据加载、增强、mask 生成
│   ├── models/                  # ViT backbone、predictor、I-JEPA 模型
│   ├── training/                # 训练循环、EMA、学习率调度
│   ├── features/                # 特征提取 + 特征库构建
│   ├── anomaly/                 # PaDiM 检测器 + 评估器
│   └── utils/                   # 配置加载、评估指标
├── scripts/
│   ├── pretrain.py              # 预训练入口
│   ├── extract_features.py      # 特征提取入口
│   └── anomaly_detect.py        # 异常检测入口
├── requirements.txt
├── .gitignore
├── CLAUDE.md
└── README.md
```

## 数据格式

数据集遵循 MVTec-AD 结构，位于远程服务器 `~/UCAD/mvtec2d/`：

```
<root>/
  <category>/
    train/good/          # 正常样本（用于预训练）
    test/good/           # 正常测试样本
    test/<defect_type>/  # 异常测试样本
    ground_truth/<defect_type>/  # 像素级异常掩码
```

共 12 个接触网零部件类别，所有类别 `train/good` 图像混合用于预训练通用模型。

## 快速开始

### 1. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 2. 修改配置

编辑 `config/ijepa_pretrain.yaml`，设置 `data.root` 为数据集路径（默认 `~/UCAD/mvtec2d`）。

### 3. 运行预训练

```bash
python scripts/pretrain.py --config config/ijepa_pretrain.yaml
```

CLI 覆盖配置：

```bash
python scripts/pretrain.py --config config/ijepa_pretrain.yaml \
    -o data.root=/path/to/dataset data.batch_size=64
```

Checkpoint 保存在 `checkpoints/` 目录。

### 4. 运行异常检测

```bash
python scripts/anomaly_detect.py --config config/anomaly_detection.yaml \
    -o features.checkpoint=checkpoints/ijepa_best.pth
```

此步骤自动进行特征提取（如果尚未建立特征库），然后逐类别评估异常检测性能。

输出：
- 每个类别的 image-level AUROC 和 pixel-level AUROC
- 结果 JSON 保存在 `results/` 目录

### 5. 单独提取特征（可选）

```bash
python scripts/extract_features.py --config config/anomaly_detection.yaml \
    -o features.checkpoint=checkpoints/ijepa_best.pth
```

特征库保存在 `feature_banks/` 目录，供后续检测复用。

### 6. 生成可视化图表

在完成预训练和异常检测后，用于生成报告/论文所需的各种图表：

```bash
# 训练曲线（loss、学习率、EMA 动量）
python scripts/visualize.py --mode training --log_dir logs/ --checkpoint_dir checkpoints/

# I-JEPA mask 可视化（context/target block 划分示意）
python scripts/visualize.py --mode masks --config config/ijepa_pretrain.yaml

# 异常检测 AUROC 柱状图
python scripts/visualize.py --mode anomaly --results results/anomaly_results.json

# 异常热力图（anomaly score map 叠加在测试图像上）
python scripts/visualize.py --mode heatmaps --config config/anomaly_detection.yaml \
    -o features.checkpoint=checkpoints/ijepa_best.pth

# 一键生成全部图表
python scripts/visualize.py --mode all \
    --config config/anomaly_detection.yaml \
    --log_dir logs/ --results results/anomaly_results.json \
    -o features.checkpoint=checkpoints/ijepa_best.pth
```

图表输出到 `figures/` 目录：
- `training_curves.png` — loss、LR schedule、EMA momentum
- `ijepa_masks.png` — 训练时 context/target block 划分可视化
- `anomaly_auroc.png` — 各类别异常检测 AUROC 柱状图
- `heatmaps_<category>.png` — 每个类别的异常热力图

## 关键参考

- **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", arXiv:2301.08243
- **PaDiM**: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", arXiv:2011.08785