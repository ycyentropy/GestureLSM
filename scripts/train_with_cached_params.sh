#!/bin/bash

# 使用默认参数训练RVQ模型（跳过预计算，直接使用缓存）
# 这个脚本使用矫正后的默认参数运行rvq_seamless_cached_train.py

# 设置GPU环境
export CUDA_VISIBLE_DEVICES=1-7
export NCCL_DEBUG=INFO

# 创建输出目录
mkdir -p outputs/rvqvae_seamless

# 设置默认参数
DATA_PATH="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
WINDOW_SIZE=64
WINDOW_STRIDE=20

# 检查缓存文件是否存在
TRAIN_CACHE="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl"
VAL_CACHE="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20.pkl"

if [ -f "$TRAIN_CACHE" ] && [ -f "$VAL_CACHE" ]; then
    echo "缓存文件已存在，跳过预计算步骤，直接使用缓存"
    echo "训练集缓存: $TRAIN_CACHE"
    echo "验证集缓存: $VAL_CACHE"
else
    echo "缓存文件不存在，需要先运行预计算"
    exit 1
fi

# 运行训练脚本（使用默认参数）
echo "开始训练..."
python rvq_seamless_cached_train.py \
    --data_path $DATA_PATH \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --use_cache \
    --cache_path "$TRAIN_CACHE" \
    --val_cache_path "$VAL_CACHE" \
    --multi_length_training \
    --batch_size 512 \
    --eval_iter 5000 \
    --out_dir outputs/rvqvae_seamless/RVQVAE_Seamless_whole \
    # 使用默认参数

echo "训练完成！"