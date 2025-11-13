# ADNI 数据集训练指南

本指南说明如何使用 ADNI 数据集进行三分类任务（CN / AD / MCI）的训练。

## 数据集信息

- **数据路径**: `/home/chenx/NeuroSTORM/data/ADNI`
- **数据格式**: `.nii.gz` 文件
- **数据维度**: `(96, 96, 96, 197)`
  - 空间: 96 × 96 × 96 (已配准到 MNI152 空间)
  - 时间: 197 个时间点
- **任务**: 三分类
  - CN (Cognitive Normal): 0
  - AD (Alzheimer's Disease): 1
  - MCI (Mild Cognitive Impairment): 2
- **数据分割**: 已提供
  - `train.txt`: 训练集文件列表
  - `val.txt`: 验证集文件列表
  - `test.txt`: 测试集文件列表

## 快速开始

### 步骤 1: 测试数据加载

首先测试数据是否能正确加载：

```bash
# 从项目根目录运行
cd /home/user/NeuroSTORM

# 运行测试脚本
python scripts/adni_downstream/test_data_loading.py
```

如果一切正常，你会看到：
- ✓ 数据文件检查通过
- ✓ 数据集创建成功
- ✓ 样本加载成功
- ✓ DataLoader 测试通过
- 类别分布统计

### 步骤 2: 配置训练脚本（可选）

编辑训练脚本以调整参数：

```bash
vim scripts/adni_downstream/train_adni_classification.sh
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBED_DIM` | 72 | 嵌入维度（可选: 36, 72, 144） |
| `SEQUENCE_LENGTH` | 40 | 时间序列长度 |
| `STRIDE` | 20 | 滑动窗口步长 |
| `BATCH_SIZE` | 8 | 批大小 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `MAX_EPOCHS` | 100 | 最大训练轮数 |

### 步骤 3: 运行训练

```bash
# 添加执行权限
chmod +x scripts/adni_downstream/train_adni_classification.sh

# 运行训练
./scripts/adni_downstream/train_adni_classification.sh
```

### 步骤 4: 监控训练

使用 TensorBoard 监控训练过程：

```bash
# 在新终端中运行
tensorboard --logdir ./output/adni_downstream/classification/lightning_logs

# 在浏览器中打开
# http://localhost:6006
```

## 文件说明

```
scripts/adni_downstream/
├── README_ADNI.md                    # 本文件
├── test_data_loading.py              # 数据加载测试脚本
└── train_adni_classification.sh      # 训练脚本
```

## 数据加载详情

### ADNI 数据集类

实现位置: `datasets/adni_dataset.py`

**特点:**
- 直接从 `.nii.gz` 文件读取，无需预处理为 `.pt` 格式
- 支持滑动窗口采样（从 197 帧中提取 40 帧的片段）
- 自动从文件路径提取标签（cn/ad/mci）
- 支持数据增强（可选）

**工作流程:**
1. 读取 `train.txt` / `val.txt` / `test.txt` 文件列表
2. 从文件路径中提取标签（例如: `.../mnispace/cn/xxx.nii.gz` → label=0）
3. 使用滑动窗口从每个文件生成多个样本
4. 运行时加载 `.nii.gz` 文件并提取时间窗口

### 滑动窗口策略

由于原始数据有 197 个时间点，而模型输入需要 40 个时间点，我们使用滑动窗口来增加训练样本：

```python
# 训练集: stride=20 (有重叠)
# 从 197 帧中，每隔 20 帧提取一个 40 帧的片段
# 生成样本数: (197 - 40) / 20 + 1 ≈ 8 个样本/文件

# 验证/测试集: stride=40 (无重叠)
# 生成样本数: (197 - 40) / 40 + 1 ≈ 4 个样本/文件
```

## 训练输出

训练完成后，输出目录结构：

```
output/adni_downstream/classification/
├── checkpoints/
│   └── best_model.ckpt          # 最佳模型
├── lightning_logs/
│   └── version_0/
│       ├── events.out.tfevents  # TensorBoard 日志
│       └── metrics.csv          # 指标 CSV
└── hparams.yaml                 # 超参数配置
```

## 评估指标

对于三分类任务，主要关注以下指标：

1. **Accuracy**: 整体准确率
2. **Per-class Accuracy**: 每个类别的准确率
   - CN accuracy
   - AD accuracy
   - MCI accuracy
3. **Confusion Matrix**: 混淆矩阵
4. **F1 Score**: 加权 F1 分数

## 常见问题

### Q1: 显存不足（OOM）

**解决方法:**

```bash
# 在训练脚本中修改:
BATCH_SIZE=4              # 减小批大小
SEQUENCE_LENGTH=20        # 减小序列长度
EMBED_DIM=36              # 减小模型大小
NUM_WORKERS=2             # 减少 worker 数量
```

### Q2: 数据加载很慢

**原因:** 每次都从 `.nii.gz` 文件读取数据

**解决方法:**
1. 减少 `NUM_WORKERS` (避免内存占用过多)
2. 如果有足够磁盘空间，可以预处理数据为 `.pt` 格式
3. 使用 SSD 存储数据

### Q3: 类别不平衡

如果三个类别的样本数量差异很大，考虑：

1. **类别权重**: 在损失函数中使用类别权重
2. **重采样**: 对少数类进行过采样
3. **数据增强**: 对少数类使用更强的数据增强

### Q4: 验证集准确率不提升

**可能原因:**
1. 学习率过大或过小
2. 模型过拟合
3. 数据增强不足

**解决方法:**
```bash
# 调整学习率
LEARNING_RATE=5e-5  # 或 1e-4, 5e-4

# 增大 dropout
DROPOUT=0.2

# 开启数据增强
USE_AUGMENTATION=true

# 增大权重衰减
WEIGHT_DECAY=0.05
```

### Q5: 如何使用预训练模型？

如果有在其他数据集（如 HCP）上预训练的模型：

```bash
# 在训练脚本中设置:
PRETRAINED_MODEL="./pretrained/neurostorm_hcp_pretrained.ckpt"

# 选择是否冻结编码器
FREEZE_ENCODER=false  # 端到端微调
# 或
FREEZE_ENCODER=true   # 只训练分类头
```

## 实验建议

### Baseline 实验

1. **从头训练**
   ```bash
   # 不使用预训练模型
   ./scripts/adni_downstream/train_adni_classification.sh
   ```

2. **使用预训练模型（如果有）**
   ```bash
   # 修改脚本，添加预训练模型路径
   PRETRAINED_MODEL="./pretrained/neurostorm_pretrained.ckpt"
   ```

3. **对比不同模型大小**
   ```bash
   # 小模型
   EMBED_DIM=36

   # 中模型
   EMBED_DIM=72

   # 大模型
   EMBED_DIM=144
   ```

### 消融实验

1. **序列长度影响**
   ```bash
   for seq_len in 20 40 60 80; do
       SEQUENCE_LENGTH=$seq_len ./train_adni_classification.sh
   done
   ```

2. **数据增强影响**
   ```bash
   # 不使用数据增强
   USE_AUGMENTATION=false

   # 使用数据增强
   USE_AUGMENTATION=true
   ```

## 下一步

完成 ADNI 数据集的 baseline 后，可以：

1. **改进模型架构**
   - 尝试不同的分类头（v1, v2, v3）
   - 调整模型深度和宽度

2. **优化训练策略**
   - 学习率调度（cosine, step）
   - 早停策略
   - 梯度裁剪

3. **高级技术**
   - 自监督预训练
   - 集成学习
   - 跨数据集泛化

4. **可解释性分析**
   - 注意力可视化
   - 显著性图
   - 错误分析

## 参考资料

- NeuroSTORM 论文: [待添加]
- ADNI 数据集: http://adni.loni.usc.edu/
- 项目主页: https://github.com/Xiang-Chen1207/NeuroSTORM

---

如有问题，请提交 Issue 或联系项目维护者。
