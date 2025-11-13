"""
ADNI Dataset for NeuroSTORM

三分类任务: CN (正常), MCI (轻度认知障碍), AD (阿尔茨海默病)
直接读取 .nii.gz 文件，无需预处理为 .pt 格式
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path


class ADNIDataset(Dataset):
    """
    ADNI 数据集类 - 直接读取 nii.gz 文件

    数据格式:
        - 输入: .nii.gz 文件，维度 (96, 96, 96, 197)
        - 标签: 从文件路径中提取 (cn/ad/mci)
        - 任务: 三分类

    Args:
        data_list_file (str): 数据列表文件路径 (train.txt, test.txt, val.txt)
        sequence_length (int): 时间序列长度（从197帧中采样）
        stride (int): 滑动窗口步长
        train (bool): 是否为训练模式
        augmentation (bool): 是否使用数据增强
        normalize (str): 归一化方法 ('zscore', 'minmax', 'none')
    """

    # 标签映射
    LABEL_MAP = {
        'cn': 0,   # 认知正常
        'ad': 1,   # 阿尔茨海默病
        'mci': 2   # 轻度认知障碍
    }

    def __init__(
        self,
        data_list_file,
        sequence_length=40,
        stride=1,
        train=True,
        augmentation=False,
        normalize='zscore'
    ):
        self.data_list_file = data_list_file
        self.sequence_length = sequence_length
        self.stride = stride if train else sequence_length  # 测试时不重叠
        self.train = train
        self.augmentation = augmentation
        self.normalize = normalize

        # 读取数据列表
        self.file_paths, self.labels = self._load_data_list()

        # 构建样本列表（使用滑动窗口）
        self.samples = self._build_samples()

        print(f"\n{'='*60}")
        print(f"ADNI Dataset Loaded: {Path(data_list_file).name}")
        print(f"{'='*60}")
        print(f"Files: {len(self.file_paths)}")
        print(f"Samples (with sliding window): {len(self.samples)}")
        print(f"Mode: {'Train' if train else 'Test'}")
        print(f"Sequence length: {sequence_length}")
        print(f"Stride: {self.stride}")
        print(f"Augmentation: {augmentation}")
        print(f"Normalization: {normalize}")

        # 打印类别分布
        label_counts = {label: 0 for label in self.LABEL_MAP.keys()}
        for label in self.labels:
            for label_name, label_id in self.LABEL_MAP.items():
                if label == label_id:
                    label_counts[label_name] += 1
        print(f"\nLabel distribution:")
        for label_name, count in label_counts.items():
            print(f"  {label_name.upper()}: {count} files")
        print(f"{'='*60}\n")

    def _load_data_list(self):
        """从txt文件读取数据列表"""
        file_paths = []
        labels = []

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            file_path = line

            # 从路径中提取标签
            # 例如: .../mnispace/cn/xxx.nii.gz -> label = 'cn'
            path_parts = file_path.split('/')

            # 查找标签（cn, ad, mci）
            label = None
            for part in path_parts:
                part_lower = part.lower()
                if part_lower in self.LABEL_MAP:
                    label = self.LABEL_MAP[part_lower]
                    break

            if label is None:
                print(f"Warning: Cannot extract label from path: {file_path}")
                continue

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            file_paths.append(file_path)
            labels.append(label)

        if len(file_paths) == 0:
            raise ValueError(f"No valid files found in {self.data_list_file}")

        return file_paths, labels

    def _build_samples(self):
        """
        构建样本列表（使用滑动窗口）

        由于原始数据有197个时间点，而模型输入需要40个时间点，
        我们使用滑动窗口从每个文件中提取多个样本
        """
        samples = []

        for idx, (file_path, label) in enumerate(zip(self.file_paths, self.labels)):
            # 假设每个文件有197个时间点
            num_frames = 197

            if num_frames < self.sequence_length:
                # 如果时间点不足，只取一个样本（会进行padding）
                samples.append({
                    'file_idx': idx,
                    'file_path': file_path,
                    'start_frame': 0,
                    'label': label
                })
            else:
                # 使用滑动窗口
                for start_frame in range(0, num_frames - self.sequence_length + 1, self.stride):
                    samples.append({
                        'file_idx': idx,
                        'file_path': file_path,
                        'start_frame': start_frame,
                        'label': label
                    })

        return samples

    def _load_nifti(self, file_path):
        """
        加载 NIfTI 文件

        Args:
            file_path: .nii.gz 文件路径

        Returns:
            numpy.ndarray: 形状 (96, 96, 96, 197)
        """
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回零数组作为fallback
            return np.zeros((96, 96, 96, 197), dtype=np.float32)

    def _normalize(self, data):
        """
        归一化数据

        Args:
            data: numpy array or torch tensor

        Returns:
            normalized data
        """
        if self.normalize == 'zscore':
            mean = data.mean()
            std = data.std()
            if std > 0:
                data = (data - mean) / std
            return data

        elif self.normalize == 'minmax':
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
            return data

        elif self.normalize == 'none':
            return data

        else:
            raise ValueError(f"Unknown normalization method: {self.normalize}")

    def _augment(self, data):
        """
        数据增强

        Args:
            data: torch.Tensor, shape [H, W, D, T]

        Returns:
            augmented data
        """
        # 1. 随机水平翻转
        if torch.rand(1) < 0.5:
            data = torch.flip(data, dims=[0])

        # 2. 添加高斯噪声
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(data) * 0.01
            data = data + noise

        # 3. 随机时间反转
        if torch.rand(1) < 0.2:
            data = torch.flip(data, dims=[-1])

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            dict: {
                'fmri': torch.Tensor, shape [1, H, W, D, T] = [1, 96, 96, 96, 40]
                'target': torch.Tensor, label (0: CN, 1: AD, 2: MCI)
                'file_path': str
            }
        """
        sample = self.samples[idx]
        file_path = sample['file_path']
        start_frame = sample['start_frame']
        label = sample['label']

        # 加载 NIfTI 文件
        data = self._load_nifti(file_path)  # shape: (96, 96, 96, 197)

        # 提取时间窗口
        end_frame = start_frame + self.sequence_length
        if end_frame <= data.shape[-1]:
            data = data[..., start_frame:end_frame]  # shape: (96, 96, 96, 40)
        else:
            # 如果不够，进行padding
            available_frames = data[..., start_frame:]
            padding_frames = self.sequence_length - available_frames.shape[-1]
            data = np.pad(
                available_frames,
                ((0, 0), (0, 0), (0, 0), (0, padding_frames)),
                mode='edge'
            )

        # 转换为 torch tensor
        data = torch.from_numpy(data).float()

        # 归一化（在提取时间窗口后）
        if self.normalize != 'none':
            data = self._normalize(data)

        # 数据增强
        if self.train and self.augmentation:
            data = self._augment(data)

        # 添加通道维度: [1, H, W, D, T]
        data = data.unsqueeze(0)

        # 标签
        target = torch.tensor(label, dtype=torch.long)

        return {
            'fmri': data,
            'target': target,
            'file_path': file_path
        }


# ============================================================================
# 用于 data_module.py 的辅助函数
# ============================================================================

def get_adni_dataset(data_list_file, **kwargs):
    """
    创建 ADNI 数据集实例

    Args:
        data_list_file: 数据列表文件路径 (train.txt, test.txt, val.txt)
        **kwargs: 其他参数传递给 ADNIDataset

    Returns:
        ADNIDataset 实例
    """
    return ADNIDataset(data_list_file=data_list_file, **kwargs)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    """
    测试 ADNI 数据集加载

    运行: python datasets/adni_dataset.py
    """
    import sys

    # TODO: 修改为你的数据路径
    data_root = '/home/chenx/NeuroSTORM/data/ADNI'
    train_file = os.path.join(data_root, 'train.txt')

    print("Testing ADNI Dataset")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        print(f"Please make sure the file exists.")
        sys.exit(1)

    # 创建数据集
    print("\n1. Creating dataset...")
    dataset = ADNIDataset(
        data_list_file=train_file,
        sequence_length=40,
        stride=5,
        train=True,
        augmentation=False,
        normalize='zscore'
    )

    # 测试加载第一个样本
    print("\n2. Loading first sample...")
    try:
        sample = dataset[0]
        print(f"   fMRI shape: {sample['fmri'].shape}")
        print(f"   Target: {sample['target'].item()}")
        print(f"   File path: {sample['file_path']}")

        # 验证形状
        expected_shape = (1, 96, 96, 96, 40)
        if sample['fmri'].shape == expected_shape:
            print(f"   ✓ Shape is correct: {expected_shape}")
        else:
            print(f"   ✗ Shape mismatch! Expected {expected_shape}, got {sample['fmri'].shape}")

    except Exception as e:
        print(f"   Error loading sample: {e}")
        import traceback
        traceback.print_exc()

    # 测试多个样本
    print("\n3. Testing multiple samples...")
    num_test_samples = min(3, len(dataset))
    for i in range(num_test_samples):
        try:
            sample = dataset[i]
            label_name = [k for k, v in dataset.LABEL_MAP.items() if v == sample['target'].item()][0]
            print(f"   Sample {i}: shape={sample['fmri'].shape}, "
                  f"label={sample['target'].item()} ({label_name.upper()})")
        except Exception as e:
            print(f"   Sample {i}: Error - {e}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
