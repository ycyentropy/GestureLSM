#!/bin/bash

# NCCL调试和优化启动脚本
# 使用方法: ./run_gpu_cache_nccl.sh [配置类型]

CONFIG_TYPE=${1:-"basic"}  # 默认使用基础配置

# 基本参数
TRAIN_DATA_PATH="datasets/train"
VAL_DATA_PATH="datasets/val"
WINDOW_SIZE=64
WINDOW_STRIDE=20
BATCH_SIZE=1024
NB_CODE=512
CODE_DIM=128
DOWN_T=2
STRIDE_T=2
WIDTH=512
DEPTH=3
LR=2e-4
TOTAL_ITER=300000

# 根据配置类型设置参数
case $CONFIG_TYPE in
  "basic")
    echo "使用基础配置 - GPU缓存数据集 + NCCL优化"
    USE_GPU_CACHE=true
    GPU_CACHE_SIZE=5000
    USE_PREFETCH=false
    NUM_WORKERS=1
    PIN_MEMORY=false
    PERSISTENT_WORKERS=false
    ;;
  "prefetch")
    echo "使用预取配置 - 预取数据加载器 + NCCL优化"
    USE_GPU_CACHE=false
    USE_PREFETCH=true
    PREFETCH_FACTOR=2
    NUM_WORKERS=1
    PIN_MEMORY=true
    PERSISTENT_WORKERS=true
    ;;
  "combined")
    echo "使用组合配置 - GPU缓存 + 预取 + NCCL优化"
    USE_GPU_CACHE=true
    GPU_CACHE_SIZE=3000
    USE_PREFETCH=true
    PREFETCH_FACTOR=2
    NUM_WORKERS=1
    PIN_MEMORY=false
    PERSISTENT_WORKERS=false
    ;;
  "conservative")
    echo "使用保守配置 - 最小NCCL通信"
    USE_GPU_CACHE=true
    GPU_CACHE_SIZE=2000
    USE_PREFETCH=false
    NUM_WORKERS=0  # 不使用额外工作进程
    PIN_MEMORY=false
    PERSISTENT_WORKERS=false
    BATCH_SIZE=512  # 减小批次大小
    ;;
  *)
    echo "未知配置类型: $CONFIG_TYPE"
    echo "可用配置: basic, prefetch, combined, conservative"
    exit 1
    ;;
esac

# 设置NCCL环境变量以解决超时问题
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30分钟超时
export NCCL_BLOCKING_WAIT=1  # 阻塞等待
export NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理

# 网络优化设置
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_NET_RETRY_COUNT=10
export NCCL_IB_DISABLE=1  # 禁用InfiniBand（如果不需要）
export NCCL_P2P_DISABLE=1  # 禁用P2P（如果GPU之间通信有问题）
export NCCL_TREE_THRESHOLD=0  # 强制使用树形算法

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA操作，便于调试

# 构建命令
CMD="python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=12355 \
  rvq_seamless_gpu_cache_nccl.py \
  --train_data_path $TRAIN_DATA_PATH \
  --val_data_path $VAL_DATA_PATH \
  --window_size $WINDOW_SIZE \
  --window_stride $WINDOW_STRIDE \
  --batch_size $BATCH_SIZE \
  --nb_code $NB_CODE \
  --code_dim $CODE_DIM \
  --down_t $DOWN_T \
  --stride_t $STRIDE_T \
  --width $WIDTH \
  --depth $DEPTH \
  --lr $LR \
  --total_iter $TOTAL_ITER \
  --multi_length_training \
  --mixed_precision \
  --num_workers $NUM_WORKERS"

# 添加配置特定参数
if [ "$USE_GPU_CACHE" = true ]; then
  CMD="$CMD --use_gpu_cache --gpu_cache_size $GPU_CACHE_SIZE"
fi

if [ "$USE_PREFETCH" = true ]; then
  CMD="$CMD --use_prefetch --prefetch_factor ${PREFETCH_FACTOR:-2}"
fi

if [ "$PIN_MEMORY" = true ]; then
  CMD="$CMD --pin_memory"
fi

if [ "$PERSISTENT_WORKERS" = true ]; then
  CMD="$CMD --persistent_workers"
fi

# 打印命令
echo "执行命令:"
echo $CMD

# 打印环境变量
echo "NCCL环境变量:"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "NCCL_TIMEOUT=$NCCL_TIMEOUT"
echo "NCCL_BLOCKING_WAIT=$NCCL_BLOCKING_WAIT"
echo "NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING"

# 执行命令
eval $CMD