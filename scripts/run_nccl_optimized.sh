#!/bin/bash

# 简单的NCCL优化启动脚本
# 使用方法: bash run_nccl_optimized.sh [配置类型]

CONFIG_TYPE=${1:-basic}  # 默认使用basic配置

echo "使用NCCL优化的配置: $CONFIG_TYPE"

# 设置基本参数
TRAIN_DATA_PATH="datasets/train"
VAL_DATA_PATH="datasets/val"
WINDOW_SIZE=64
WINDOW_STRIDE=20
BATCH_SIZE=1024
OUTPUT_DIR="checkpoints_nccl_optimized"

# 根据配置类型设置参数
case $CONFIG_TYPE in
    "basic")
        echo "使用基础NCCL优化配置"
        NUM_WORKERS=1
        PERSISTENT_WORKERS=false
        PIN_MEMORY=false
        USE_GPU_CACHE=false
        USE_PREFETCH=false
        ;;
    "gpu_cache")
        echo "使用GPU缓存配置"
        NUM_WORKERS=1
        PERSISTENT_WORKERS=false
        PIN_MEMORY=false
        USE_GPU_CACHE=true
        GPU_CACHE_SIZE=5000
        USE_PREFETCH=false
        ;;
    "prefetch")
        echo "使用预取配置"
        NUM_WORKERS=1
        PERSISTENT_WORKERS=false
        PIN_MEMORY=false
        USE_GPU_CACHE=false
        USE_PREFETCH=true
        PREFETCH_FACTOR=2
        ;;
    "conservative")
        echo "使用保守配置（最小资源使用）"
        NUM_WORKERS=0  # 不使用多进程
        PERSISTENT_WORKERS=false
        PIN_MEMORY=false
        USE_GPU_CACHE=false
        USE_PREFETCH=false
        BATCH_SIZE=512  # 减小批次大小
        ;;
    *)
        echo "未知配置类型，使用默认配置"
        NUM_WORKERS=1
        PERSISTENT_WORKERS=false
        PIN_MEMORY=false
        USE_GPU_CACHE=false
        USE_PREFETCH=false
        ;;
esac

# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30分钟超时
export NCCL_BLOCKING_WAIT=1  # 阻塞等待
export NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_NET_RETRY_COUNT=10
export NCCL_IB_DISABLE=1  # 禁用InfiniBand（如果不需要）
export NCCL_P2P_DISABLE=1  # 禁用P2P（如果GPU之间通信有问题）
export NCCL_TREE_THRESHOLD=0  # 强制使用树形算法

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA操作，便于调试

# 构建命令
CMD="python -m torch.distributed.launch --nproc_per_node=8 rvq_seamless_gpu_cache_nccl.py \
    --train_data_path $TRAIN_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --batch_size $BATCH_SIZE \
    --multi_length_training \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR"

# 添加可选参数
if [ "$PERSISTENT_WORKERS" = true ]; then
    CMD="$CMD --persistent_workers"
fi

if [ "$PIN_MEMORY" = true ]; then
    CMD="$CMD --pin_memory"
fi

if [ "$USE_GPU_CACHE" = true ]; then
    CMD="$CMD --use_gpu_cache --gpu_cache_size $GPU_CACHE_SIZE"
fi

if [ "$USE_PREFETCH" = true ]; then
    CMD="$CMD --use_prefetch --prefetch_factor $PREFETCH_FACTOR"
fi

# 执行命令
echo "执行命令: $CMD"
eval $CMD