#!/bin/bash

# 多GPU训练脚本
# 使用8个GPU，每个GPU使用batch_size=256，总有效batch_size=2048

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# 训练配置
NUM_GPUS=7  # 使用7个空闲GPU (1-7)
BATCH_SIZE=256  # 每个GPU的batch_size
MAX_SAMPLES=5000  # 使用5000个样本

echo "开始多GPU训练..."
echo "使用GPU数量: $NUM_GPUS"
echo "每个GPU的Batch Size: $BATCH_SIZE"
echo "总有效Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "使用样本数量: $MAX_SAMPLES"
echo "预计时间窗口数: $((MAX_SAMPLES * 270))"

# 执行多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    rvq_seamless_train.py \
    --mode train \
    --batch-size $BATCH_SIZE \
    --max-samples $MAX_SAMPLES

echo "训练完成!"