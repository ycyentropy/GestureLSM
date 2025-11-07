#!/bin/bash

# 使用默认参数训练RVQ模型
# 这个脚本使用矫正后的默认参数运行rvq_seamless_cached_train.py

# 设置GPU环境
export CUDA_VISIBLE_DEVICES=0-6
export NCCL_DEBUG=INFO

# 创建输出目录
mkdir -p outputs/rvqvae_seamless

# 设置默认参数
DATA_PATH="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
WINDOW_SIZE=64
WINDOW_STRIDE=20

# 预计算窗口参数（缓存数据获取）
echo "预计算训练集窗口参数..."
python save_window_params.py \
    --data_path $DATA_PATH \
    --split train \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --multi_length_training

echo "预计算验证集窗口参数..."
python save_window_params.py \
    --data_path $DATA_PATH \
    --split val \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE

# 运行训练脚本（使用默认参数）
echo "开始训练..."
python rvq_seamless_cached_train.py

echo "训练完成！"