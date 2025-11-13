#!/usr/bin/env python3
"""
测试 ADNI 数据集加载

用法:
    python scripts/adni_downstream/test_data_loading.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from datasets.adni_dataset import ADNIDataset


def test_adni_dataset():
    """测试 ADNI 数据集加载"""

    # 数据路径
    data_root = '/home/chenx/NeuroSTORM/data/ADNI'
    train_file = os.path.join(data_root, 'train.txt')
    val_file = os.path.join(data_root, 'val.txt')
    test_file = os.path.join(data_root, 'test.txt')

    print("="*80)
    print("Testing ADNI Dataset Loading")
    print("="*80)

    # 检查文件是否存在
    print("\n1. Checking data files...")
    for name, path in [('train', train_file), ('val', val_file), ('test', test_file)]:
        if os.path.exists(path):
            print(f"   ✓ {name}.txt found: {path}")
            # 统计行数
            with open(path, 'r') as f:
                num_lines = len([line for line in f if line.strip()])
            print(f"     - Contains {num_lines} files")
        else:
            print(f"   ✗ {name}.txt NOT found: {path}")
            return False

    # 创建训练集
    print("\n2. Creating train dataset...")
    try:
        train_dataset = ADNIDataset(
            data_list_file=train_file,
            sequence_length=40,
            stride=20,
            train=True,
            augmentation=False,
            normalize='none'  # 数据已经归一化
        )
        print(f"   ✓ Train dataset created successfully")
    except Exception as e:
        print(f"   ✗ Error creating train dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建验证集
    print("\n3. Creating validation dataset...")
    try:
        val_dataset = ADNIDataset(
            data_list_file=val_file,
            sequence_length=40,
            stride=40,  # 验证集不重叠
            train=False,
            augmentation=False,
            normalize='none'
        )
        print(f"   ✓ Validation dataset created successfully")
    except Exception as e:
        print(f"   ✗ Error creating validation dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建测试集
    print("\n4. Creating test dataset...")
    try:
        test_dataset = ADNIDataset(
            data_list_file=test_file,
            sequence_length=40,
            stride=40,  # 测试集不重叠
            train=False,
            augmentation=False,
            normalize='none'
        )
        print(f"   ✓ Test dataset created successfully")
    except Exception as e:
        print(f"   ✗ Error creating test dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试加载第一个样本
    print("\n5. Testing sample loading...")
    try:
        sample = train_dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"   - fMRI shape: {sample['fmri'].shape}")
        print(f"   - Target: {sample['target'].item()}")
        label_name = [k for k, v in train_dataset.LABEL_MAP.items()
                     if v == sample['target'].item()][0]
        print(f"   - Label: {label_name.upper()}")
        print(f"   - File: {os.path.basename(sample['file_path'])}")

        # 验证形状
        expected_shape = (1, 96, 96, 96, 40)
        if sample['fmri'].shape == expected_shape:
            print(f"   ✓ Shape is correct: {expected_shape}")
        else:
            print(f"   ✗ Shape mismatch!")
            print(f"     Expected: {expected_shape}")
            print(f"     Got: {sample['fmri'].shape}")
            return False

    except Exception as e:
        print(f"   ✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试多个样本
    print("\n6. Testing multiple samples...")
    num_test_samples = min(5, len(train_dataset))
    for i in range(num_test_samples):
        try:
            sample = train_dataset[i]
            label_name = [k for k, v in train_dataset.LABEL_MAP.items()
                         if v == sample['target'].item()][0]
            print(f"   Sample {i}: shape={sample['fmri'].shape}, "
                  f"label={label_name.upper()} ({sample['target'].item()})")
        except Exception as e:
            print(f"   Sample {i}: Error - {e}")

    # 测试 DataLoader
    print("\n7. Testing DataLoader...")
    try:
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 使用0避免多进程问题
        )

        # 获取一个batch
        batch = next(iter(train_loader))
        print(f"   ✓ DataLoader works")
        print(f"   - Batch fMRI shape: {batch['fmri'].shape}")
        print(f"   - Batch target shape: {batch['target'].shape}")

    except Exception as e:
        print(f"   ✗ Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 统计类别分布
    print("\n8. Class distribution in datasets:")
    for dataset_name, dataset in [('Train', train_dataset),
                                   ('Val', val_dataset),
                                   ('Test', test_dataset)]:
        label_counts = {0: 0, 1: 0, 2: 0}
        for sample in dataset.samples:
            label_counts[sample['label']] += 1

        total = sum(label_counts.values())
        print(f"\n   {dataset_name} set ({total} samples):")
        for label_id, count in label_counts.items():
            label_name = [k for k, v in dataset.LABEL_MAP.items() if v == label_id][0]
            percentage = 100 * count / total if total > 0 else 0
            print(f"     {label_name.upper()}: {count} ({percentage:.1f}%)")

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nYou can now run the training script:")
    print("  bash scripts/adni_downstream/train_adni_classification.sh")
    print("="*80)

    return True


if __name__ == '__main__':
    success = test_adni_dataset()
    sys.exit(0 if success else 1)
