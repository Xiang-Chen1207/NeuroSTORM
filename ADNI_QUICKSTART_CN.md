# ADNI 数据集快速开始指南

本指南说明如何使用你的 ADNI 数据集（已配准到 MNI152 空间的 `.nii.gz` 文件）来训练 NeuroSTORM 模型进行三分类任务。

## 📋 你的数据集信息

- **数据路径**: `/home/chenx/NeuroSTORM/data/ADNI`
- **数据格式**: `.nii.gz` (无需转换为 `.pt` 格式)
- **数据维度**: `(96, 96, 96, 197)`
- **任务类型**: 三分类 (CN / AD / MCI)
- **数据分割**: 已提供
  - `train.txt` - 训练集文件列表
  - `val.txt` - 验证集文件列表
  - `test.txt` - 测试集文件列表

## 🚀 三步快速开始

### 步骤 1: 测试数据加载

首先验证数据能否正确加载：

```bash
cd /home/chenx/NeuroSTORM

# 运行数据加载测试
python scripts/adni_downstream/test_data_loading.py
```

**预期输出:**
```
============================================================
Testing ADNI Dataset Loading
============================================================

1. Checking data files...
   ✓ train.txt found
   ✓ val.txt found
   ✓ test.txt found

2. Creating train dataset...
   ✓ Train dataset created successfully

3. Creating validation dataset...
   ✓ Validation dataset created successfully

...

All tests passed! ✓
```

### 步骤 2: 运行训练

```bash
# 添加执行权限
chmod +x scripts/adni_downstream/train_adni_classification.sh

# 开始训练
./scripts/adni_downstream/train_adni_classification.sh
```

训练脚本会：
1. ✓ 检查数据文件是否存在
2. ✓ 加载训练/验证/测试数据
3. ✓ 创建 NeuroSTORM 模型
4. ✓ 开始训练并保存最佳模型

### 步骤 3: 监控训练

在新终端中运行：

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/adni_downstream/classification/lightning_logs

# 浏览器访问: http://localhost:6006
```

你可以看到：
- 训练/验证损失曲线
- 准确率变化
- 每个类别的准确率

## 📁 代码结构

### 核心文件

```
NeuroSTORM/
├── datasets/
│   └── adni_dataset.py              # ADNI 数据集类（直接读取 .nii.gz）
├── utils/
│   └── data_module.py               # 已添加 ADNI 支持
├── scripts/adni_downstream/
│   ├── README_ADNI.md               # 详细文档
│   ├── test_data_loading.py        # 数据加载测试
│   └── train_adni_classification.sh # 训练脚本
└── data/ADNI/                       # 你的数据目录
    ├── train.txt                     # 训练集文件列表
    ├── val.txt                       # 验证集文件列表
    └── test.txt                      # 测试集文件列表
```

### 数据加载流程

```
train.txt (文件路径列表)
    ↓
ADNIDataset 类
    ↓
从路径提取标签 (cn/ad/mci → 0/1/2)
    ↓
加载 .nii.gz 文件 (96, 96, 96, 197)
    ↓
滑动窗口采样 (提取 40 帧片段)
    ↓
返回样本: (1, 96, 96, 96, 40) + label
```

## ⚙️ 关键参数

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEQUENCE_LENGTH` | 40 | 时间序列长度（从197帧中采样） |
| `STRIDE` | 20 | 滑动窗口步长（训练集） |

**滑动窗口说明:**
- 原始数据: 197 个时间点
- 模型输入: 40 个时间点
- 训练集 stride=20: 每个文件生成 ~8 个样本（有重叠）
- 验证/测试集 stride=40: 每个文件生成 ~4 个样本（无重叠）

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | neurostorm | 模型类型 |
| `EMBED_DIM` | 72 | 嵌入维度 (36/72/144) |
| `NUM_CLASSES` | 3 | 类别数 (CN/AD/MCI) |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 8 | 批大小 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `MAX_EPOCHS` | 100 | 最大训练轮数 |
| `USE_AUGMENTATION` | true | 数据增强 |

## 🔧 常见问题

### Q1: 显存不足怎么办?

编辑 `scripts/adni_downstream/train_adni_classification.sh`:

```bash
BATCH_SIZE=4              # 减小批大小
SEQUENCE_LENGTH=20        # 减小序列长度
EMBED_DIM=36              # 使用更小的模型
```

### Q2: 训练太慢?

```bash
NUM_WORKERS=2             # 减少数据加载的worker数
BATCH_SIZE=16             # 增大批大小（如果显存允许）
```

### Q3: 如何调整超参数?

主要参数优先级:
1. **学习率** - 最重要，建议尝试: [5e-5, 1e-4, 5e-4]
2. **批大小** - 影响训练稳定性
3. **Dropout** - 防止过拟合，建议: [0.1, 0.2, 0.3]
4. **模型大小** - EMBED_DIM 越大模型容量越大

### Q4: 类别不平衡怎么处理?

查看测试输出中的类别分布，如果不平衡严重，可以：
1. 使用类别权重（需要修改代码）
2. 增大少数类的 stride 以生成更多样本
3. 使用数据增强

## 📊 训练输出

训练完成后，查看结果：

```bash
# 最佳模型
ls output/adni_downstream/classification/checkpoints/

# 训练指标
cat output/adni_downstream/classification/lightning_logs/version_0/metrics.csv

# TensorBoard 可视化
tensorboard --logdir output/adni_downstream/classification/lightning_logs
```

## 📈 评估结果

主要关注的指标：

1. **Overall Accuracy** - 整体准确率
2. **Per-class Accuracy** - 每个类别的准确率
   - CN (正常) 准确率
   - AD (阿尔茨海默病) 准确率
   - MCI (轻度认知障碍) 准确率
3. **Confusion Matrix** - 混淆矩阵（分析错误分类模式）

## 🎯 下一步

### Baseline 对比

1. **从头训练** (当前设置)
   ```bash
   ./scripts/adni_downstream/train_adni_classification.sh
   ```

2. **使用预训练模型** (如果有)
   ```bash
   # 在脚本中取消注释:
   PRETRAINED_MODEL="./pretrained/neurostorm_pretrained.ckpt"
   ```

3. **不同模型大小对比**
   ```bash
   # 分别设置 EMBED_DIM=36, 72, 144
   ```

### 改进方向

1. **超参数调优**
   - 网格搜索最优学习率
   - 尝试不同的序列长度
   - 调整数据增强策略

2. **模型改进**
   - 尝试不同的分类头 (v1, v2, v3)
   - 使用集成学习
   - 添加注意力机制

3. **数据分析**
   - 分析错误分类样本
   - 可视化特征表示
   - 分析不同类别的区分性特征

## 📚 详细文档

- **完整文档**: `scripts/adni_downstream/README_ADNI.md`
- **数据集代码**: `datasets/adni_dataset.py`
- **通用指南**: `CUSTOM_DATASET_GUIDE.md`

## 💬 获取帮助

- 查看日志文件了解错误信息
- 运行 `python scripts/adni_downstream/test_data_loading.py` 诊断数据问题
- 参考 `scripts/adni_downstream/README_ADNI.md` 获取详细说明

---

祝你实验顺利！如有问题随时查阅文档或提交 Issue。🎉
