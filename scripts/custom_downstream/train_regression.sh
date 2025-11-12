#!/bin/bash

################################################################################
# 自定义数据集 - 回归任务训练脚本
#
# 使用场景: 年龄预测、认知分数预测等连续值预测任务
#
# 使用方法:
#   1. 修改下面的参数
#   2. chmod +x train_regression.sh
#   3. ./train_regression.sh
################################################################################

# ==================== 需要修改的参数 ====================

# TODO: 修改为你的数据集名称
DATASET_NAME="YOUR_DATASET"

# TODO: 修改为你的数据路径
DATA_PATH="./data/YOUR_DATASET_MNI_to_TRs_minmax"

# TODO: 修改为你的任务名称（例如: age, iq, cognitive_score）
TASK_NAME="age"

# TODO: 如果有预训练模型，修改此路径
PRETRAINED_MODEL="./pretrained/neurostorm_pretrained.ckpt"

# TODO: 修改输出路径
OUTPUT_PATH="./output/custom_downstream/${TASK_NAME}"

# ==================== 模型参数 ====================

# 模型类型
MODEL="neurostorm"

# 嵌入维度
EMBED_DIM=72

# 每个阶段的层数
DEPTH="2 2 6 2"

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

# 训练轮数
MAX_EPOCHS=50

# 批大小
BATCH_SIZE=8

# 学习率
LEARNING_RATE=1e-4

# 权重衰减
WEIGHT_DECAY=0.01

# Dropout
DROPOUT=0.1

# 回归头类型 (v1: 简单线性, v2: MLP)
HEAD_TYPE="v2"

# 是否冻结编码器
FREEZE_ENCODER=false

# 是否归一化标签（回归任务强烈建议开启）
NORMALIZE_LABEL=true

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

# 是否使用数据增强
USE_AUGMENTATION=true

# ==================== 日志设置 ====================

# 日志类型
LOG_TYPE="tensorboard"

# ==================== 开始训练 ====================

echo "=================================="
echo "Starting Regression Training"
echo "=================================="
echo "Dataset: $DATASET_NAME"
echo "Task: $TASK_NAME"
echo "Model: $MODEL"
echo "Output: $OUTPUT_PATH"
echo "=================================="

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 构建训练命令
CMD="python main.py \
  --accelerator $ACCELERATOR \
  --devices $DEVICES \
  --max_epochs $MAX_EPOCHS \
  --dataset_name $DATASET_NAME \
  --image_path $DATA_PATH \
  --downstream_task_type regression \
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
  --freeze_encoder $FREEZE_ENCODER \
  --head_type $HEAD_TYPE \
  --dropout $DROPOUT \
  --log_type $LOG_TYPE"

# 添加标签归一化（回归任务推荐）
if [ "$NORMALIZE_LABEL" = true ]; then
  echo "Label normalization enabled"
  CMD="$CMD --normalize_label"
fi

# 添加预训练模型（如果存在）
if [ -f "$PRETRAINED_MODEL" ]; then
  echo "Loading pretrained model: $PRETRAINED_MODEL"
  CMD="$CMD --load_model_path $PRETRAINED_MODEL"
else
  echo "No pretrained model found. Training from scratch."
fi

# 添加数据增强（如果开启）
if [ "$USE_AUGMENTATION" = true ]; then
  echo "Data augmentation enabled"
  CMD="$CMD --use_augmentation"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# 执行训练
eval $CMD

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo "Results saved to: $OUTPUT_PATH"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $OUTPUT_PATH/lightning_logs"
echo ""
echo "Best model saved at:"
echo "  $OUTPUT_PATH/checkpoints/"
echo ""
