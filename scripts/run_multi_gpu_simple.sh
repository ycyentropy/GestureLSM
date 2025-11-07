#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# 训练配置
NUM_GPUS=7
BATCH_SIZE_PER_GPU=292  # 每个GPU的批次大小
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE_PER_GPU))
MAX_SAMPLES=5000

echo "=========================================="
echo "启动RVQ-VAE多GPU训练"
echo "=========================================="
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "GPU数量: $NUM_GPUS"
echo "每个GPU批次大小: $BATCH_SIZE_PER_GPU"
echo "总有效批次大小: $TOTAL_BATCH_SIZE"
echo "最大样本数: $MAX_SAMPLES"
echo "=========================================="

# 使用torch.distributed.launch启动多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    rvq_seamless_multi_gpu.py \
    --data_path data/seamless_motions \
    --body_part whole \
    --batch_size $BATCH_SIZE_PER_GPU \
    --total_iter 10000 \
    --max_samples $MAX_SAMPLES \
    --exp_name rvq_seamless_multi_gpu \
    --out_dir experiments