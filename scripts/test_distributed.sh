#!/bin/bash

# 简单的分布式训练测试脚本
echo "测试分布式训练..."

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 使用torch.distributed.launch而不是torchrun
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12355 \
    rvq_seamless_optimized.py \
    --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
    --out-dir outputs/rvqvae_seamless_optimized \
    --batch-size 1024 \
    --total-iter 30000 \
    --eval-iter 5000 \
    --print-iter 500 \
    --use-amp \
    --pin-memory \
    --non-blocking-transfer