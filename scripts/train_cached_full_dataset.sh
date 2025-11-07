#!/bin/bash
# 使用缓存的窗口参数进行训练的启动脚本 - 使用全部数据集

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # 使用GPU 1-7，跳过GPU 0
export NCCL_DEBUG=INFO

# 训练参数
DATA_PATH="datasets/seamless_interaction"
WINDOW_SIZE=64
WINDOW_STRIDE=20
BATCH_SIZE=256
EPOCHS=100
LR=2e-4
EXP_NAME="rvq_seamless_cached_full_dataset"
OUT_DIR="experiments/rvq_seamless_cached_full_dataset"

# 模型参数（与rvq_seamless_multi_gpu.py默认参数一致）
NB_CODE=512
CODE_DIM=256
DOWN_T=2
STRIDE_T=2
WIDTH=512
DEPTH=3
DILATION_GROWTH_RATE=3
VQ_ACT="relu"
VQ_NORM=None
MU=0.99
NUM_QUANTIZERS=8
SHARED_CODEBOOK=false
QUANTIZE_DROPOUT_PROB=0.5

# 缓存参数
USE_CACHE=true
SAVE_CACHE=true
CACHE_PATH=""  # 留空自动生成

# 分布式训练参数
WORLD_SIZE=7  # 使用7个GPU，对应CUDA_VISIBLE_DEVICES中的7个GPU

# 创建输出目录
mkdir -p $OUT_DIR

# 首先预计算并保存窗口参数（只在主节点执行）
if [ "$SAVE_CACHE" = true ]; then
    echo "预计算并保存窗口参数（使用全部数据集）..."
    python save_window_params.py \
        --data_path $DATA_PATH \
        --split train \
        --window_size $WINDOW_SIZE \
        --window_stride $WINDOW_STRIDE \
        --multi_length_training \
        --save_path $(dirname $DATA_PATH)/window_params
    
    python save_window_params.py \
        --data_path $DATA_PATH \
        --split dev \
        --window_size $WINDOW_SIZE \
        --window_stride $WINDOW_STRIDE \
        --multi_length_training \
        --save_path $(dirname $DATA_PATH)/window_params
fi

# 启动分布式训练
echo "启动分布式训练..."
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "进程数量: $WORLD_SIZE"
echo "使用全部数据集进行训练"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29501 \
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
    --log_interval 100 \
    --save_interval 10 \
    --code_dim $CODE_DIM \
    --n_codebooks $NUM_QUANTIZERS \
    --codebook_size $NB_CODE \
    --down_t $DOWN_T \
    --stride_t $STRIDE_T \
    --depth $DEPTH \
    --dilation $DILATION_GROWTH_RATE \
    --vq_act $VQ_ACT \
    --emb_norm $MU \
    --width $WIDTH