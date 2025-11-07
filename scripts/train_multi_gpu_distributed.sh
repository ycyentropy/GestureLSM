#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # 使用GPU 1-7，跳过正在使用的GPU 0

# 训练参数
TOTAL_BATCH_SIZE=2048  # 总有效batch_size
NUM_GPUS=7             # 使用的GPU数量
BATCH_SIZE_PER_GPU=$((TOTAL_BATCH_SIZE / NUM_GPUS))  # 每个GPU的batch_size
MAX_SAMPLES=5000       # 最大样本数

# 打印配置信息
echo "===== 多GPU训练配置 ====="
echo "使用的GPU: $CUDA_VISIBLE_DEVICES"
echo "GPU数量: $NUM_GPUS"
echo "总有效batch_size: $TOTAL_BATCH_SIZE"
echo "每个GPU的batch_size: $BATCH_SIZE_PER_GPU"
echo "最大样本数: $MAX_SAMPLES"
echo "========================"

# 使用torch.distributed.launch启动多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    rvq_seamless_train_distributed.py \
    --batch_size $BATCH_SIZE_PER_GPU \
    --max_samples $MAX_SAMPLES \
    --total_iter 10000 \
    --body_part whole \
    --exp_name rvq_seamless_multi_gpu \
    --print_iter 100 \
    --eval_iter 1000 \
    --lr 2e-4 \
    --commit 0.02