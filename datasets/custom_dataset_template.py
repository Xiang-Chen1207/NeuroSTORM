"""
自定义数据集类模板

使用此模板创建你自己的 fMRI 数据集类。
只需修改标记为 "TODO" 的部分即可。
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    """
    自定义 fMRI 数据集

    数据目录结构:
    dataset_path/
    ├── img/
    │   ├── sub001/
    │   │   ├── frame_0.pt
    │   │   ├── frame_1.pt
    │   │   └── ...
    │   ├── sub002/
    │   │   └── ...
    │   └── ...
    └── metadata/
        └── metadata.csv

    Args:
        dataset_path (str): 数据集根目录
        subject_dict (dict): 被试字典 {subject_id: {"subject": id, "target": label}}
        sequence_length (int): 时间序列长度（时间点数）
        stride (int): 滑动窗口步长
        train (bool): 是否为训练模式
        augmentation (bool): 是否使用数据增强
    """

    def __init__(
        self,
        dataset_path,
        subject_dict,
        sequence_length=40,
        stride=1,
        train=True,
        augmentation=False
    ):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'img')
        self.sequence_length = sequence_length
        self.stride = stride if train else sequence_length  # 测试时不重叠
        self.train = train
        self.augmentation = augmentation
        self.subject_dict = subject_dict

        # 构建样本列表
        self.samples = self._build_samples()

        print(f"CustomDataset loaded:")
        print(f"  - Subjects: {len(subject_dict)}")
        print(f"  - Samples: {len(self.samples)}")
        print(f"  - Mode: {'Train' if train else 'Test'}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Stride: {self.stride}")

    def _build_samples(self):
        """构建样本列表"""
        samples = []

        for subject_id, info in self.subject_dict.items():
            subject_folder = os.path.join(self.img_path, subject_id)

            # 检查被试文件夹是否存在
            if not os.path.exists(subject_folder):
                print(f"Warning: Subject folder not found: {subject_folder}")
                continue

            # 获取该被试的所有时间帧文件
            frame_files = sorted([
                f for f in os.listdir(subject_folder)
                if f.endswith('.pt') and f.startswith('frame_')
            ])

            if len(frame_files) == 0:
                print(f"Warning: No frames found for subject: {subject_id}")
                continue

            num_frames = len(frame_files)

            # TODO: 如果你的数据时间点太少，可以调整这里的逻辑
            if num_frames < self.sequence_length:
                print(f"Warning: Subject {subject_id} has only {num_frames} frames, "
                      f"less than required {self.sequence_length}")
                continue

            # 使用滑动窗口创建样本
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                samples.append({
                    'subject_id': subject_id,
                    'start_frame': start_idx,
                    'num_frames': num_frames,
                    'target': info['target']
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            dict: {
                'fmri': torch.Tensor, shape [1, H, W, D, T]
                'target': torch.Tensor, 标签
                'subject_id': str, 被试 ID
            }
        """
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        start_frame = sample['start_frame']
        target = sample['target']

        # 加载时间序列数据
        frames = []
        subject_folder = os.path.join(self.img_path, subject_id)

        for i in range(start_frame, start_frame + self.sequence_length):
            frame_path = os.path.join(subject_folder, f'frame_{i}.pt')

            try:
                frame = torch.load(frame_path, weights_only=True)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                # 使用零填充
                frame = torch.zeros(96, 96, 96)

            frames.append(frame)

        # 拼接为 4D 张量: [H, W, D, T]
        img = torch.stack(frames, dim=-1)

        # TODO: 数据增强（可选）
        if self.train and self.augmentation:
            img = self._augment(img)

        # 添加通道维度: [1, H, W, D, T]
        if img.dim() == 4:
            img = img.unsqueeze(0)

        # TODO: 根据你的任务类型调整标签格式
        # 分类任务: long 类型
        # 回归任务: float 类型
        if isinstance(target, (int, np.integer)):
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = torch.tensor(target, dtype=torch.float32)

        return {
            'fmri': img,
            'target': target,
            'subject_id': subject_id
        }

    def _augment(self, img):
        """
        数据增强（可选）

        TODO: 根据需要添加数据增强方法
        - 空间变换: 旋转、翻转、平移
        - 噪声添加: 高斯噪声
        - 时间变换: 时间反转、时间平移

        Args:
            img: torch.Tensor, shape [H, W, D, T]

        Returns:
            torch.Tensor: 增强后的图像
        """
        # 示例: 添加高斯噪声
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(img) * 0.01
            img = img + noise

        # 示例: 水平翻转
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[0])

        return img


# ============================================================================
# 数据集注册
# ============================================================================

def get_custom_dataset(dataset_name, **kwargs):
    """
    获取自定义数据集

    TODO: 在这里注册你的数据集类

    Args:
        dataset_name (str): 数据集名称
        **kwargs: 传递给数据集类的参数

    Returns:
        Dataset: 数据集实例
    """
    datasets = {
        'CUSTOM': CustomDataset,
        # TODO: 添加更多自定义数据集
        # 'YOUR_DATASET_NAME': YourDatasetClass,
    }

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(datasets.keys())}"
        )

    return datasets[dataset_name](**kwargs)


# ============================================================================
# 元数据加载函数
# ============================================================================

def load_custom_metadata(metadata_path, task_name):
    """
    加载自定义数据集的元数据

    TODO: 根据你的元数据格式修改此函数

    Args:
        metadata_path (str): 元数据文件路径 (通常是 .csv 文件)
        task_name (str): 任务名称 (例如: 'diagnosis', 'age', 'sex')

    Returns:
        dict: 被试字典 {subject_id: {"subject": id, "target": label}}
    """
    # 读取 CSV 文件
    meta_df = pd.read_csv(metadata_path)

    print(f"Loaded metadata from: {metadata_path}")
    print(f"  - Total subjects: {len(meta_df)}")
    print(f"  - Columns: {meta_df.columns.tolist()}")
    print(f"  - Task: {task_name}")

    subject_dict = {}

    # TODO: 根据任务类型提取标签
    if task_name == 'diagnosis':
        # 二分类任务示例
        for _, row in meta_df.iterrows():
            subject_id = str(row['subject_id'])
            label = int(row['label'])  # 假设标签列名为 'label'

            subject_dict[subject_id] = {
                'subject': subject_id,
                'target': label
            }

        print(f"  - Label distribution:")
        labels = [v['target'] for v in subject_dict.values()]
        for label_val in set(labels):
            count = labels.count(label_val)
            print(f"    Class {label_val}: {count} samples")

    elif task_name == 'age':
        # 回归任务示例
        for _, row in meta_df.iterrows():
            subject_id = str(row['subject_id'])
            age = float(row['age'])  # 假设年龄列名为 'age'

            subject_dict[subject_id] = {
                'subject': subject_id,
                'target': age
            }

        ages = [v['target'] for v in subject_dict.values()]
        print(f"  - Age range: {min(ages):.1f} - {max(ages):.1f}")
        print(f"  - Age mean: {np.mean(ages):.1f}")

    elif task_name == 'sex':
        # 性别分类示例
        for _, row in meta_df.iterrows():
            subject_id = str(row['subject_id'])
            sex = row['sex']  # 假设性别列名为 'sex'

            # 转换为数值标签
            label = 1 if sex in ['M', 'Male', 'male', '1'] else 0

            subject_dict[subject_id] = {
                'subject': subject_id,
                'target': label
            }

        labels = [v['target'] for v in subject_dict.values()]
        print(f"  - Male: {labels.count(1)} samples")
        print(f"  - Female: {labels.count(0)} samples")

    else:
        # TODO: 添加其他任务类型
        raise ValueError(f"Unsupported task: {task_name}")

    return subject_dict


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    """
    测试自定义数据集

    运行此脚本来测试你的数据集是否正确加载:
    python datasets/custom_dataset_template.py
    """

    # TODO: 修改为你的数据路径
    dataset_path = './data/YOUR_DATASET_MNI_to_TRs_minmax'
    metadata_path = os.path.join(dataset_path, 'metadata', 'metadata.csv')

    # TODO: 修改为你的任务名称
    task_name = 'diagnosis'

    print("=" * 80)
    print("Testing Custom Dataset")
    print("=" * 80)

    # 加载元数据
    print("\n1. Loading metadata...")
    subject_dict = load_custom_metadata(metadata_path, task_name)

    # 创建数据集
    print("\n2. Creating dataset...")
    dataset = CustomDataset(
        dataset_path=dataset_path,
        subject_dict=subject_dict,
        sequence_length=40,
        stride=5,
        train=True,
        augmentation=False
    )

    # 测试加载样本
    print("\n3. Testing sample loading...")
    sample = dataset[0]

    print(f"  - fMRI shape: {sample['fmri'].shape}")
    print(f"  - Target: {sample['target']}")
    print(f"  - Subject ID: {sample['subject_id']}")

    # 测试多个样本
    print("\n4. Testing multiple samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: shape={sample['fmri'].shape}, "
              f"target={sample['target'].item()}, "
              f"subject={sample['subject_id']}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
