#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # 使用GPU 1-7，跳过正在使用的GPU 0

# 训练参数
TOTAL_BATCH_SIZE=4096  # 总有效batch_size，可以根据GPU内存调整
NUM_GPUS=7             # 使用的GPU数量
BATCH_SIZE_PER_GPU=$((TOTAL_BATCH_SIZE / NUM_GPUS))  # 每个GPU的batch_size
MAX_SAMPLES=999999     # 使用所有数据

# 打印配置信息
echo "===== 多GPU训练配置 ====="
echo "使用的GPU: $CUDA_VISIBLE_DEVICES"
echo "GPU数量: $NUM_GPUS"
echo "总有效batch_size: $TOTAL_BATCH_SIZE"
echo "每个GPU的batch_size: $BATCH_SIZE_PER_GPU"
echo "最大样本数: $MAX_SAMPLES (使用所有数据)"
echo "========================"

# 使用torch.distributed.launch启动多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    rvq_seamless_multi_gpu.py \
    --batch_size $BATCH_SIZE_PER_GPU \
    --max_samples $MAX_SAMPLES \
    --total_iter 20000 \
    --body_part whole \
    --exp_name rvq_seamless_all_data \
    --print_iter 100 \
    --eval_iter 1000 \
    --lr 2e-4 \
    --commit 0.02 \
    --data_path datasets/seamless_interaction