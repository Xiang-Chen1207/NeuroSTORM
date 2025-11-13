#!/bin/bash

################################################################################
# ADNI 数据集 - 三分类任务 (CN / AD / MCI)
#
# 数据集信息:
#   - 数据路径: /home/chenx/NeuroSTORM/data/ADNI
#   - 数据格式: .nii.gz (96, 96, 96, 197)
#   - 任务: 三分类 (CN=0, AD=1, MCI=2)
#   - 数据分割: 已提供 train.txt, val.txt, test.txt
#
# 使用方法:
#   chmod +x train_adni_classification.sh
#   ./train_adni_classification.sh
################################################################################

# ==================== 数据集配置 ====================

# 数据集名称
DATASET_NAME="ADNI"

# 数据路径（包含 train.txt, val.txt, test.txt 的目录）
DATA_PATH="/home/chenx/NeuroSTORM/data/ADNI"

# 任务类型
TASK_TYPE="classification"

# 类别数 (CN, AD, MCI = 3类)
NUM_CLASSES=3

# 输出路径
OUTPUT_PATH="./output/adni_downstream/classification"

# ==================== 模型配置 ====================

# 模型类型: neurostorm 或 swift
MODEL="neurostorm"

# 嵌入维度 (推荐: 36, 72, 144)
EMBED_DIM=72

# 每个阶段的层数
DEPTH="2 2 6 2"

# 窗口大小
WINDOW_SIZE="4 4 4 4"

# Patch 大小
PATCH_SIZE="6 6 6 1"

# 输入图像大小 (空间: 96x96x96, 时间: 40)
IMG_SIZE="96 96 96 40"

# 时间序列长度 (从197帧中采样40帧)
SEQUENCE_LENGTH=40

# 滑动窗口步长 (用于数据增强，从一个文件生成多个样本)
STRIDE=20

# ==================== 训练配置 ====================

# GPU 设置
ACCELERATOR="gpu"
DEVICES=1

# 训练轮数
MAX_EPOCHS=100

# 批大小 (根据显存调整)
BATCH_SIZE=8

# 评估批大小
EVAL_BATCH_SIZE=16

# 学习率
LEARNING_RATE=1e-4

# 权重衰减
WEIGHT_DECAY=0.01

# Attention Dropout Rate
ATTN_DROP_RATE=0.1

# 分类头版本 (v1: 简单线性, v2: MLP)
CLF_HEAD_VERSION="v2"

# 是否冻结特征提取器 (如果使用预训练模型)
FREEZE_FEATURE_EXTRACTOR=false

# ==================== 预训练模型 (可选) ====================

# 如果有预训练模型，取消下面的注释
# PRETRAINED_MODEL="./pretrained/neurostorm_pretrained.ckpt"

# ==================== 数据增强 ====================

# 是否使用数据增强
USE_AUGMENTATION=true

# ==================== 日志配置 ====================

# 日志类型: tensorboard 或 neptune
LOG_TYPE="tensorboard"

# 实验名称
EXP_NAME="adni_cn_ad_mci_neurostorm"

# ==================== 其他参数 ====================

# Worker数量
NUM_WORKERS=4

# 随机种子
SEED=42

# ==================== 开始训练 ====================

echo "========================================="
echo "ADNI Classification Training"
echo "========================================="
echo "Dataset: $DATASET_NAME"
echo "Data path: $DATA_PATH"
echo "Task: 3-class classification (CN/AD/MCI)"
echo "Model: $MODEL"
echo "Embedding dim: $EMBED_DIM"
echo "Sequence length: $SEQUENCE_LENGTH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Output: $OUTPUT_PATH"
echo "========================================="

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH/train.txt" ]; then
    echo "Error: $DATA_PATH/train.txt not found!"
    exit 1
fi

if [ ! -f "$DATA_PATH/val.txt" ]; then
    echo "Error: $DATA_PATH/val.txt not found!"
    exit 1
fi

if [ ! -f "$DATA_PATH/test.txt" ]; then
    echo "Error: $DATA_PATH/test.txt not found!"
    exit 1
fi

echo "✓ Data files found"
echo ""

# 构建训练命令
CMD="python main.py \
  --accelerator $ACCELERATOR \
  --devices $DEVICES \
  --max_epochs $MAX_EPOCHS \
  --dataset_name $DATASET_NAME \
  --image_path $DATA_PATH \
  --downstream_task_type $TASK_TYPE \
  --num_classes $NUM_CLASSES \
  --model $MODEL \
  --embed_dim $EMBED_DIM \
  --depths $DEPTH \
  --window_size $WINDOW_SIZE \
  --patch_size $PATCH_SIZE \
  --img_size $IMG_SIZE \
  --sequence_length $SEQUENCE_LENGTH \
  --stride_between_seq $STRIDE \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --default_root_dir $OUTPUT_PATH \
  --clf_head_version $CLF_HEAD_VERSION \
  --attn_drop_rate $ATTN_DROP_RATE \
  --num_workers $NUM_WORKERS \
  --seed $SEED"

# 添加数据增强
if [ "$USE_AUGMENTATION" = true ]; then
  echo "Data augmentation enabled"
  CMD="$CMD --augment_during_training"
fi

# 添加预训练模型（如果存在）
if [ -n "${PRETRAINED_MODEL:-}" ] && [ -f "$PRETRAINED_MODEL" ]; then
  echo "Loading pretrained model: $PRETRAINED_MODEL"
  CMD="$CMD --load_model_path $PRETRAINED_MODEL"

  # 是否冻结特征提取器
  if [ "$FREEZE_FEATURE_EXTRACTOR" = true ]; then
    echo "Freezing feature extractor"
    CMD="$CMD --freeze_feature_extractor"
  fi
else
  echo "Training from scratch (no pretrained model)"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# 执行训练
eval $CMD

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Results saved to: $OUTPUT_PATH"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $OUTPUT_PATH/lightning_logs"
echo ""
echo "Best model saved at:"
echo "  $OUTPUT_PATH/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Check validation metrics in TensorBoard"
echo "  2. Evaluate on test set"
echo "  3. Analyze confusion matrix for 3-class classification"
echo ""
