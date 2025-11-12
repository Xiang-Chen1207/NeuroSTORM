# 使用自定义数据集运行 Brain Foundation Model 下游任务指南

本指南将帮助你使用自己的 fMRI 数据集来运行 NeuroSTORM brain foundation model 的下游任务作为 baseline。

## 目录
1. [数据准备](#1-数据准备)
2. [添加自定义数据集类](#2-添加自定义数据集类)
3. [配置数据加载](#3-配置数据加载)
4. [创建训练脚本](#4-创建训练脚本)
5. [运行训练](#5-运行训练)

---

## 1. 数据准备

### 1.1 数据格式要求

你的数据需要组织成以下结构：

```
./data/YOUR_DATASET_MNI_to_TRs_minmax/
├── img/
│   ├── sub001/
│   │   ├── frame_0.pt
│   │   ├── frame_1.pt
│   │   ├── frame_2.pt
│   │   └── ...
│   ├── sub002/
│   │   └── ...
│   └── ...
└── metadata/
    └── metadata.csv
```

### 1.2 数据预处理

如果你的数据是原始 NIfTI 格式，需要先进行预处理：

```bash
# 使用项目提供的预处理脚本
python datasets/preprocessing_volume.py \
  --input_dir ./raw_data/your_dataset \
  --output_dir ./data/YOUR_DATASET_MNI_to_TRs_minmax \
  --img_size 96 96 96 \
  --normalize minmax
```

**预处理步骤包括：**
- 空间标准化到 MNI152 空间 (96x96x96)
- 时间维度分割（每个时间点保存为单独的 .pt 文件）
- 体素级归一化（minmax 或 zscore）

### 1.3 元数据文件 (metadata.csv)

创建一个包含标签信息的 CSV 文件：

```csv
subject_id,label,age,sex,site
sub001,0,25,M,site1
sub002,1,30,F,site1
sub003,0,28,M,site2
...
```

**必需字段：**
- `subject_id`: 被试 ID（需与 img/ 目录下的文件夹名匹配）
- 任务相关的标签列（如 `label`, `age`, `sex` 等）

---

## 2. 添加自定义数据集类

### 2.1 在 `datasets/fmri_datasets.py` 中添加数据集类

```python
class YourCustomDataset(Dataset):
    """
    自定义数据集类

    Args:
        dataset_path: 数据集根目录
        subject_dict: 被试字典 {subject_id: {"subject": id, "target": label}}
        sequence_length: 时间序列长度
        stride: 滑动窗口步长
        train: 是否为训练模式
    """
    def __init__(self, dataset_path, subject_dict, sequence_length=40,
                 stride=1, train=True):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'img')
        self.sequence_length = sequence_length
        self.stride = stride
        self.train = train
        self.subject_dict = subject_dict

        # 构建样本列表
        self.samples = []
        for subject_id, info in subject_dict.items():
            subject_folder = os.path.join(self.img_path, subject_id)

            # 获取该被试的所有时间帧
            frames = sorted([f for f in os.listdir(subject_folder)
                           if f.endswith('.pt')])
            num_frames = len(frames)

            # 使用滑动窗口创建样本
            for start_idx in range(0, num_frames - sequence_length + 1, stride):
                self.samples.append({
                    'subject_id': subject_id,
                    'start_frame': start_idx,
                    'target': info['target']
                })

        print(f"Dataset loaded: {len(self.samples)} samples from "
              f"{len(subject_dict)} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        start_frame = sample['start_frame']
        target = sample['target']

        # 加载时间序列
        frames = []
        subject_folder = os.path.join(self.img_path, subject_id)

        for i in range(start_frame, start_frame + self.sequence_length):
            frame_path = os.path.join(subject_folder, f'frame_{i}.pt')
            frame = torch.load(frame_path)
            frames.append(frame)

        # 拼接为 4D 张量: [C, H, W, T] 或 [H, W, D, T]
        img = torch.stack(frames, dim=-1)  # [H, W, D, T]

        # 确保是 5D: [B, C, H, W, T]
        if img.dim() == 4:
            img = img.unsqueeze(0)  # 添加通道维度

        return {
            'fmri': img,
            'target': torch.tensor(target, dtype=torch.float32),
            'subject_id': subject_id
        }
```

### 2.2 注册数据集

在 `datasets/fmri_datasets.py` 的末尾添加：

```python
# 在 get_dataset 函数中添加你的数据集
def get_dataset(dataset_name, **kwargs):
    datasets = {
        'HCP1200': HCP1200Dataset,
        'ABCD': ABCDDataset,
        'YOUR_DATASET': YourCustomDataset,  # 添加这行
        # ... 其他数据集
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return datasets[dataset_name](**kwargs)
```

---

## 3. 配置数据加载

### 3.1 在 `utils/data_module.py` 中添加任务配置

修改 `make_subject_dict` 方法，添加你的数据集和任务：

```python
def make_subject_dict(self):
    """创建被试字典和标签映射"""

    # 读取元数据
    if self.dataset_name == 'YOUR_DATASET':
        meta_path = os.path.join(self.image_path, 'metadata', 'metadata.csv')
        meta_df = pd.read_csv(meta_path)

        # 根据任务类型提取标签
        if self.task_name == 'diagnosis':  # 分类任务
            subject_dict = {}
            for _, row in meta_df.iterrows():
                subject_id = row['subject_id']
                label = int(row['label'])  # 0 or 1
                subject_dict[subject_id] = {
                    'subject': subject_id,
                    'target': label
                }

        elif self.task_name == 'age':  # 回归任务
            subject_dict = {}
            for _, row in meta_df.iterrows():
                subject_id = row['subject_id']
                age = float(row['age'])
                subject_dict[subject_id] = {
                    'subject': subject_id,
                    'target': age
                }

        return subject_dict

    # ... 保持其他数据集的原有逻辑
```

---

## 4. 创建训练脚本

### 4.1 下游分类任务示例

创建 `scripts/custom_downstream/train_classification.sh`:

```bash
#!/bin/bash

# 分类任务（例如：疾病诊断）
python main.py \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 50 \
  --dataset_name YOUR_DATASET \
  --image_path ./data/YOUR_DATASET_MNI_to_TRs_minmax \
  --downstream_task_type classification \
  --task_name diagnosis \
  --num_classes 2 \
  --model neurostorm \
  --embed_dim 72 \
  --depth 2 2 6 2 \
  --window_size 4 4 4 4 \
  --patch_size 6 6 6 1 \
  --img_size 96 96 96 40 \
  --sequence_length 40 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --train_split 0.7 \
  --val_split 0.15 \
  --output_path ./output/custom_downstream/classification \
  --load_model_path ./pretrained/neurostorm_pretrained.ckpt \
  --freeze_encoder false \
  --head_type v2 \
  --dropout 0.1 \
  --use_augmentation \
  --log_type tensorboard
```

### 4.2 下游回归任务示例

创建 `scripts/custom_downstream/train_regression.sh`:

```bash
#!/bin/bash

# 回归任务（例如：年龄预测）
python main.py \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 50 \
  --dataset_name YOUR_DATASET \
  --image_path ./data/YOUR_DATASET_MNI_to_TRs_minmax \
  --downstream_task_type regression \
  --task_name age \
  --model neurostorm \
  --embed_dim 72 \
  --depth 2 2 6 2 \
  --window_size 4 4 4 4 \
  --patch_size 6 6 6 1 \
  --img_size 96 96 96 40 \
  --sequence_length 40 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --normalize_label \
  --train_split 0.7 \
  --val_split 0.15 \
  --output_path ./output/custom_downstream/regression \
  --load_model_path ./pretrained/neurostorm_pretrained.ckpt \
  --freeze_encoder false \
  --head_type v2 \
  --dropout 0.1 \
  --log_type tensorboard
```

### 4.3 从头训练（不使用预训练模型）

```bash
#!/bin/bash

# 不加载预训练权重，从头训练
python main.py \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 100 \
  --dataset_name YOUR_DATASET \
  --image_path ./data/YOUR_DATASET_MNI_to_TRs_minmax \
  --downstream_task_type classification \
  --task_name diagnosis \
  --num_classes 2 \
  --model neurostorm \
  --embed_dim 72 \
  --depth 2 2 6 2 \
  --window_size 4 4 4 4 \
  --patch_size 6 6 6 1 \
  --img_size 96 96 96 40 \
  --sequence_length 40 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --output_path ./output/custom_downstream/from_scratch \
  --log_type tensorboard
```

---

## 5. 运行训练

### 5.1 准备工作

```bash
# 1. 确保环境配置正确
source set_env.sh

# 2. 检查数据目录结构
ls -R ./data/YOUR_DATASET_MNI_to_TRs_minmax

# 3. 创建输出目录
mkdir -p ./output/custom_downstream
mkdir -p ./scripts/custom_downstream
```

### 5.2 运行训练

```bash
# 给脚本添加执行权限
chmod +x ./scripts/custom_downstream/train_classification.sh

# 运行训练
./scripts/custom_downstream/train_classification.sh
```

### 5.3 监控训练

```bash
# 使用 TensorBoard 监控
tensorboard --logdir ./output/custom_downstream/classification/lightning_logs
```

### 5.4 评估结果

训练完成后，检查输出：

```bash
# 查看最佳模型
ls ./output/custom_downstream/classification/checkpoints/

# 查看日志
cat ./output/custom_downstream/classification/lightning_logs/version_0/metrics.csv
```

---

## 6. 关键参数说明

### 6.1 模型参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型类型 | `neurostorm` 或 `swift` |
| `--embed_dim` | 嵌入维度 | 36, 72, 144 |
| `--depth` | 每个阶段的层数 | `2 2 6 2` |
| `--window_size` | 窗口大小 | `4 4 4 4` |
| `--patch_size` | Patch 大小 | `6 6 6 1` |
| `--img_size` | 输入图像大小 | `96 96 96 40` |
| `--sequence_length` | 时间序列长度 | 20-80 |

### 6.2 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--learning_rate` | 学习率 | 1e-4 (fine-tune), 1e-3 (from scratch) |
| `--batch_size` | 批大小 | 4-16 (取决于 GPU 内存) |
| `--max_epochs` | 最大训练轮数 | 50-100 |
| `--weight_decay` | 权重衰减 | 0.01-0.05 |
| `--dropout` | Dropout 比例 | 0.1-0.3 |

### 6.3 数据参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--train_split` | 训练集比例 | 0.7 |
| `--val_split` | 验证集比例 | 0.15 |
| `--stride` | 滑动窗口步长 | 1-5 |
| `--use_augmentation` | 是否使用数据增强 | 小数据集建议开启 |

### 6.4 任务参数

| 参数 | 说明 | 值 |
|------|------|-----|
| `--downstream_task_type` | 任务类型 | `classification` 或 `regression` |
| `--num_classes` | 类别数 | 分类任务必需 |
| `--normalize_label` | 是否归一化标签 | 回归任务建议开启 |
| `--head_type` | 分类头类型 | `v1`, `v2`, `v3` |

---

## 7. 常见问题

### Q1: 显存不足怎么办？

```bash
# 减小批大小
--batch_size 4

# 减小模型大小
--embed_dim 36

# 减小序列长度
--sequence_length 20

# 使用梯度累积
--accumulate_grad_batches 4
```

### Q2: 如何使用预训练模型？

```bash
# 下载预训练模型（如果项目提供）
wget https://your-pretrained-model-url -O ./pretrained/neurostorm_pretrained.ckpt

# 在训练时加载
--load_model_path ./pretrained/neurostorm_pretrained.ckpt

# 选择是否冻结编码器
--freeze_encoder false  # 允许微调
--freeze_encoder true   # 只训练分类头
```

### Q3: 如何调整超参数？

建议使用网格搜索或随机搜索：

```python
# 创建 scripts/custom_downstream/hyperparameter_search.py
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
batch_sizes = [4, 8, 16]
dropout_rates = [0.1, 0.2, 0.3]

for lr in learning_rates:
    for bs in batch_sizes:
        for dropout in dropout_rates:
            # 运行训练...
```

### Q4: 如何处理类别不平衡？

在 `models/lightning_model.py` 中使用加权损失：

```python
# 计算类别权重
class_counts = [...]  # 每个类别的样本数
weights = 1.0 / torch.tensor(class_counts)
weights = weights / weights.sum()

# 在损失函数中使用
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 8. 下一步

完成 baseline 后，你可以：

1. **对比实验**：与其他方法（如 3D CNN、RNN）对比
2. **消融实验**：测试不同模型配置的影响
3. **跨数据集评估**：在其他数据集上测试泛化能力
4. **可视化分析**：使用 attention map 分析模型关注区域

---

## 参考资料

- 项目主页: [NeuroSTORM GitHub](https://github.com/Xiang-Chen1207/NeuroSTORM)
- 相关论文: 查看 README.md 中的引用
- 示例脚本: `scripts/` 目录下的各种训练脚本

如有问题，请查阅项目文档或提交 Issue。
