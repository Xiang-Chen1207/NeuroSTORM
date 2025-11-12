# 快速开始 - 使用自定义数据集

本指南将帮助你快速上手，使用自己的数据集运行 brain foundation model 下游任务。

## 📋 前置要求

- ✅ 已克隆项目到本地
- ✅ 已配置环境（运行 `source set_env.sh`）
- ✅ 有一个 fMRI 数据集（NIfTI 或已预处理的格式）
- ✅ 有对应的标签文件（CSV 格式）

## 🚀 五步快速开始

### 步骤 1: 准备数据

#### 方法 A: 如果你的数据已经是预处理好的 `.pt` 格式

确保数据组织如下：

```
./data/YOUR_DATASET_MNI_to_TRs_minmax/
├── img/
│   ├── sub001/
│   │   ├── frame_0.pt
│   │   ├── frame_1.pt
│   │   └── ...
│   └── sub002/
│       └── ...
└── metadata/
    └── metadata.csv
```

#### 方法 B: 如果你的数据是原始 NIfTI 格式

运行预处理脚本：

```bash
python datasets/preprocessing_volume.py \
  --input_dir ./raw_data/your_dataset \
  --output_dir ./data/YOUR_DATASET_MNI_to_TRs_minmax \
  --img_size 96 96 96 \
  --normalize minmax
```

### 步骤 2: 准备元数据文件

创建 `./data/YOUR_DATASET_MNI_to_TRs_minmax/metadata/metadata.csv`:

```csv
subject_id,label,age,sex
sub001,0,25,M
sub002,1,30,F
sub003,0,28,M
...
```

**字段说明：**
- `subject_id`: 被试 ID（必需，需与 img/ 目录下的文件夹名匹配）
- 其他列根据你的任务添加（如 `label`, `age`, `sex` 等）

### 步骤 3: 修改数据集加载代码

#### 3.1 在 `datasets/fmri_datasets.py` 中注册你的数据集

在文件末尾的 `get_dataset` 函数中添加：

```python
def get_dataset(dataset_name, **kwargs):
    datasets = {
        'HCP1200': HCP1200Dataset,
        'YOUR_DATASET': HCP1200Dataset,  # 先复用现有的数据集类
        # ... 其他数据集
    }
```

> 💡 **提示**: 如果现有的 `HCP1200Dataset` 类能满足你的需求，可以直接复用。如果需要自定义，参考 `datasets/custom_dataset_template.py`。

#### 3.2 在 `utils/data_module.py` 中添加标签加载逻辑

找到 `make_subject_dict` 方法，添加你的数据集：

```python
def make_subject_dict(self):
    # ... 现有代码 ...

    # 添加你的数据集
    elif self.dataset_name == 'YOUR_DATASET':
        meta_path = os.path.join(self.image_path, 'metadata', 'metadata.csv')
        meta_df = pd.read_csv(meta_path)

        if self.task_name == 'diagnosis':  # 分类任务
            for _, row in meta_df.iterrows():
                subject_id = str(row['subject_id'])
                label = int(row['label'])
                subject_dict[subject_id] = {
                    'subject': subject_id,
                    'target': label
                }

        elif self.task_name == 'age':  # 回归任务
            for _, row in meta_df.iterrows():
                subject_id = str(row['subject_id'])
                age = float(row['age'])
                subject_dict[subject_id] = {
                    'subject': subject_id,
                    'target': age
                }

        return subject_dict
```

### 步骤 4: 配置训练脚本

选择一个训练脚本并修改参数：

#### 选项 A: 分类任务（如疾病诊断）

```bash
# 编辑脚本
vim scripts/custom_downstream/train_classification.sh

# 修改以下参数:
DATASET_NAME="YOUR_DATASET"
DATA_PATH="./data/YOUR_DATASET_MNI_to_TRs_minmax"
TASK_NAME="diagnosis"
NUM_CLASSES=2
```

#### 选项 B: 回归任务（如年龄预测）

```bash
# 编辑脚本
vim scripts/custom_downstream/train_regression.sh

# 修改以下参数:
DATASET_NAME="YOUR_DATASET"
DATA_PATH="./data/YOUR_DATASET_MNI_to_TRs_minmax"
TASK_NAME="age"
```

#### 选项 C: 从头训练（不使用预训练模型）

```bash
# 编辑脚本
vim scripts/custom_downstream/train_from_scratch.sh

# 修改以下参数:
DATASET_NAME="YOUR_DATASET"
DATA_PATH="./data/YOUR_DATASET_MNI_to_TRs_minmax"
TASK_NAME="diagnosis"
TASK_TYPE="classification"  # 或 "regression"
```

### 步骤 5: 运行训练

```bash
# 给脚本添加执行权限
chmod +x scripts/custom_downstream/train_classification.sh

# 运行训练
./scripts/custom_downstream/train_classification.sh
```

## 📊 监控训练进度

### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/custom_downstream/

# 在浏览器中打开
# http://localhost:6006
```

### 查看训练日志

```bash
# 查看最新日志
tail -f ./output/custom_downstream/*/lightning_logs/version_*/metrics.csv
```

## 🎯 评估结果

训练完成后：

```bash
# 查看最佳模型
ls ./output/custom_downstream/*/checkpoints/

# 查看训练指标
cat ./output/custom_downstream/*/lightning_logs/version_0/metrics.csv
```

## 🔧 常见问题

### Q1: 显存不足（OOM）怎么办？

**解决方法：**

```bash
# 在训练脚本中修改:
BATCH_SIZE=4              # 减小批大小
EMBED_DIM=36              # 减小模型大小
SEQUENCE_LENGTH=20        # 减小序列长度
```

或添加梯度累积：

```bash
CMD="$CMD --accumulate_grad_batches 4"
```

### Q2: 数据加载报错找不到文件

**检查清单：**

1. 确认数据目录结构正确
2. 确认 `subject_id` 与文件夹名匹配
3. 确认帧文件命名为 `frame_0.pt`, `frame_1.pt` ...

```bash
# 验证数据结构
ls -R ./data/YOUR_DATASET_MNI_to_TRs_minmax/img/sub001/
```

### Q3: 如何使用预训练模型？

**方法 1: 使用项目提供的预训练模型**

如果项目有预训练模型，下载后：

```bash
# 在训练脚本中设置:
PRETRAINED_MODEL="./pretrained/neurostorm_pretrained.ckpt"
```

**方法 2: 从 HuggingFace 加载**

```bash
# 在训练命令中添加:
--load_model_path "hf://username/model-name"
```

**方法 3: 自己预训练**

参考 `scripts/hcp_pretrain/` 中的预训练脚本。

### Q4: 类别不平衡怎么处理？

在 `models/lightning_model.py` 中添加类别权重：

```python
# 计算类别权重
class_counts = [num_class0, num_class1]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()

# 使用加权损失
self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
```

### Q5: 如何调整超参数？

**关键超参数：**

| 参数 | 小数据集 | 大数据集 |
|------|---------|---------|
| `learning_rate` | 5e-5 | 1e-4 |
| `batch_size` | 4-8 | 16-32 |
| `dropout` | 0.2-0.3 | 0.1 |
| `weight_decay` | 0.05 | 0.01 |
| `use_augmentation` | ✅ 开启 | 可选 |

**建议的调参顺序：**

1. 先用默认参数训练
2. 如果过拟合：增大 dropout / 增大 weight_decay / 开启数据增强
3. 如果欠拟合：增大模型 / 增加训练轮数 / 调整学习率

## 📚 进阶使用

### 多次实验对比

```bash
# 使用不同的数据集分割
for split_num in 0 1 2 3 4; do
  ./scripts/custom_downstream/train_classification.sh \
    --dataset_split_num $split_num
done
```

### 跨验证

```bash
# K-fold 交叉验证
for fold in {0..4}; do
  python main.py \
    ... \
    --dataset_split_num $fold \
    --train_split 0.8 \
    --val_split 0.2
done
```

### 集成学习

训练多个模型并集成预测结果。

## 📖 详细文档

- 完整指南: `CUSTOM_DATASET_GUIDE.md`
- 数据集模板: `datasets/custom_dataset_template.py`
- 示例脚本: `scripts/custom_downstream/`

## 💬 获取帮助

- 查看项目 README: `README.md`
- 参考现有脚本: `scripts/hcp_downstream/`
- 提交 Issue: [GitHub Issues](https://github.com/Xiang-Chen1207/NeuroSTORM/issues)

---

祝你实验顺利！🎉
