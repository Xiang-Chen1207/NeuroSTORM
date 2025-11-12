#!/bin/bash

################################################################################
# 从头训练 - 不使用预训练模型
#
# 使用场景:
#   - 没有合适的预训练模型
#   - 数据分布与预训练数据差异较大
#   - 作为 baseline 对比预训练模型的效果
#
# 注意: 从头训练通常需要更多的数据和训练时间
################################################################################

# ==================== 需要修改的参数 ====================

# TODO: 修改为你的数据集名称
DATASET_NAME="YOUR_DATASET"

# TODO: 修改为你的数据路径
DATA_PATH="./data/YOUR_DATASET_MNI_to_TRs_minmax"

# TODO: 修改为你的任务名称
TASK_NAME="diagnosis"

# TODO: 任务类型: classification 或 regression
TASK_TYPE="classification"

# TODO: 如果是分类任务，设置类别数
NUM_CLASSES=2

# TODO: 修改输出路径
OUTPUT_PATH="./output/custom_downstream/from_scratch_${TASK_NAME}"

# ==================== 模型参数 ====================

# 模型类型
MODEL="neurostorm"

# 嵌入维度 (从头训练可以使用较小的模型)
EMBED_DIM=36

# 每个阶段的层数
DEPTH="2 2 4 2"

# 窗口大小
WINDOW_SIZE="4 4 4 4"

# Patch 大小
PATCH_SIZE="6 6 6 1"

# 输入图像大小
IMG_SIZE="96 96 96 40"

# 时间序列长度
SEQUENCE_LENGTH=40

# ==================== 训练参数 ====================

# GPU 设置
ACCELERATOR="gpu"
DEVICES=1

# 训练轮数 (从头训练需要更多轮数)
MAX_EPOCHS=100

# 批大小
BATCH_SIZE=8

# 学习率 (从头训练使用较大的学习率)
LEARNING_RATE=1e-3

# 权重衰减
WEIGHT_DECAY=0.05

# Dropout (从头训练可以使用更大的 dropout 防止过拟合)
DROPOUT=0.2

# 分类头类型
HEAD_TYPE="v2"

# ==================== 数据分割 ====================

# 训练集比例
TRAIN_SPLIT=0.7

# 验证集比例
VAL_SPLIT=0.15

# 数据集分割编号
DATASET_SPLIT_NUM=0

# 滑动窗口步长
STRIDE=1

# ==================== 数据增强 ====================

# 强烈建议开启数据增强
USE_AUGMENTATION=true

# ==================== 学习率调度 ====================

# 学习率调度器类型 (可选: cosine, step, plateau)
LR_SCHEDULER="cosine"

# ==================== 早停 ====================

# 早停耐心值 (验证集指标不提升的最大轮数)
EARLY_STOP_PATIENCE=15

# ==================== 日志设置 ====================

# 日志类型
LOG_TYPE="tensorboard"

# ==================== 开始训练 ====================

echo "========================================="
echo "Starting Training From Scratch"
echo "========================================="
echo "Dataset: $DATASET_NAME"
echo "Task: $TASK_NAME ($TASK_TYPE)"
echo "Model: $MODEL (embed_dim=$EMBED_DIM)"
echo "Output: $OUTPUT_PATH"
echo "Max epochs: $MAX_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "========================================="

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 构建基础命令
CMD="python main.py \
  --accelerator $ACCELERATOR \
  --devices $DEVICES \
  --max_epochs $MAX_EPOCHS \
  --dataset_name $DATASET_NAME \
  --image_path $DATA_PATH \
  --downstream_task_type $TASK_TYPE \
  --task_name $TASK_NAME \
  --model $MODEL \
  --embed_dim $EMBED_DIM \
  --depth $DEPTH \
  --window_size $WINDOW_SIZE \
  --patch_size $PATCH_SIZE \
  --img_size $IMG_SIZE \
  --sequence_length $SEQUENCE_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --train_split $TRAIN_SPLIT \
  --val_split $VAL_SPLIT \
  --dataset_split_num $DATASET_SPLIT_NUM \
  --stride $STRIDE \
  --output_path $OUTPUT_PATH \
  --head_type $HEAD_TYPE \
  --dropout $DROPOUT \
  --log_type $LOG_TYPE"

# 添加类别数（如果是分类任务）
if [ "$TASK_TYPE" = "classification" ]; then
  CMD="$CMD --num_classes $NUM_CLASSES"
fi

# 添加标签归一化（如果是回归任务）
if [ "$TASK_TYPE" = "regression" ]; then
  CMD="$CMD --normalize_label"
fi

# 添加数据增强
if [ "$USE_AUGMENTATION" = true ]; then
  echo "Data augmentation enabled"
  CMD="$CMD --use_augmentation"
fi

echo ""
echo "Training configuration:"
echo "  - Training from scratch (no pretrained model)"
echo "  - Model size: $EMBED_DIM embedding dim"
echo "  - Max epochs: $MAX_EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Dropout: $DROPOUT"
echo "  - Data augmentation: $USE_AUGMENTATION"
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
echo "  1. Check validation metrics in tensorboard"
echo "  2. Compare with pretrained model results"
echo "  3. Try different hyperparameters if needed"
echo ""
