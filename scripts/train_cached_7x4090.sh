#!/bin/bash
# 使用缓存的窗口参数进行训练的启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NCCL_DEBUG=INFO

# 训练参数
DATA_PATH="datasets/seamless_interaction"
WINDOW_SIZE=60
WINDOW_STRIDE=30
BATCH_SIZE=32
EPOCHS=100
LR=2e-4
EXP_NAME="rvq_seamless_cached_7x4090"
OUT_DIR="experiments/rvq_seamless_cached_7x4090_whole"

# 缓存参数
USE_CACHE=true
SAVE_CACHE=true
CACHE_PATH=""  # 留空自动生成

# 分布式训练参数
WORLD_SIZE=7

# 创建输出目录
mkdir -p $OUT_DIR

# 首先预计算并保存窗口参数（只在主节点执行）
if [ "$SAVE_CACHE" = true ]; then
    echo "预计算并保存窗口参数..."
    python save_window_params.py \
        --data_path $DATA_PATH \
        --split train \
        --window_size $WINDOW_SIZE \
        --window_stride $WINDOW_STRIDE \
        --multi_length_training \
        --max_samples 1000 \
        --output_path $(dirname $DATA_PATH)/window_params
    
    python save_window_params.py \
        --data_path $DATA_PATH \
        --split dev \
        --window_size $WINDOW_SIZE \
        --window_stride $WINDOW_STRIDE \
        --multi_length_training \
        --max_samples 1000 \
        --output_path $(dirname $DATA_PATH)/window_params
fi

# 启动分布式训练
echo "启动分布式训练..."
torchrun \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    rvq_seamless_cached_train.py \
    --data_path $DATA_PATH \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --exp_name $EXP_NAME \
    --out_dir $OUT_DIR \
    --use_cache $USE_CACHE \
    --save_cache $SAVE_CACHE \
    --multi_length_training \
    --max_samples 1000 \
    --log_interval 100 \
    --save_interval 10