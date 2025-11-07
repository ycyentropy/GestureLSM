#!/bin/bash

# 使用修复后的多长度缓存进行多GPU训练

echo "=== 使用修复后缓存的多GPU训练 ==="

# 设置参数
N_GPUS=8
BATCH_SIZE=1024
CACHE_TRAIN="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
CACHE_VAL="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20_fixed.pkl"

echo "使用配置:"
echo "  GPU数量: $N_GPUS"
echo "  批次大小: $BATCH_SIZE"
echo "  训练缓存: $CACHE_TRAIN"
echo "  验证缓存: $CACHE_VAL"

# 检查缓存文件是否存在
if [ ! -f "$CACHE_TRAIN" ]; then
    echo "警告: 训练缓存文件不存在: $CACHE_TRAIN"
    echo "等待缓存生成完成..."
    # 等待缓存文件生成
    while [ ! -f "$CACHE_TRAIN" ]; do
        sleep 10
        echo "等待中..."
    done
    echo "训练缓存文件已生成!"
fi

if [ ! -f "$CACHE_VAL" ]; then
    echo "警告: 验证缓存文件不存在: $CACHE_VAL"
    echo "等待缓存生成完成..."
    # 等待缓存文件生成
    while [ ! -f "$CACHE_VAL" ]; do
        sleep 10
        echo "等待中..."
    done
    echo "验证缓存文件已生成!"
fi

echo "开始训练..."

# 使用torch.distributed.launch进行多GPU训练
# 注意：使用新修复的缓存文件路径
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPUS \
    --master_port=29501 \
    rvq_seamless_multi_gpu.py \
    --batch_size $BATCH_SIZE \
    --cache_train $CACHE_TRAIN \
    --cache_val $CACHE_VAL \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5

echo "训练完成!"