#!/bin/bash

# 多GPU训练脚本 - 充分利用8个4090 GPU显存
# 使用8个GPU，每个GPU使用batch_size=1024，总有效batch_size=8192

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据路径
DATA_PATH="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
WINDOW_SIZE=64  # 与缓存文件匹配
WINDOW_STRIDE=20

# 训练配置
NUM_GPUS=8  # 使用8个GPU
BATCH_SIZE=1024  # 每个GPU的batch_size，增大以充分利用显存
WINDOW_SIZE=64  # 与缓存文件匹配的窗口大小
WIDTH=512  # 增大网络宽度
DEPTH=3  # 增大网络深度

# 缓存文件路径
TRAIN_CACHE="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl"
VAL_CACHE="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20.pkl"

echo "开始多GPU训练..."
echo "使用GPU数量: $NUM_GPUS"
echo "每个GPU的Batch Size: $BATCH_SIZE"
echo "总有效Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "窗口大小: $WINDOW_SIZE"
echo "网络宽度: $WIDTH"
echo "网络深度: $DEPTH"

# 创建输出目录
mkdir -p outputs/rvqvae_seamless_multi_gpu

# 执行多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12355 \
    rvq_seamless_multi_gpu.py \
    --data_path "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction" \
    --batch_size $BATCH_SIZE \
    --use_cache \
    --cache_path "$TRAIN_CACHE" \
    --val_cache_path "$VAL_CACHE" \
    --multi_length_training \
    --mixed_precision \
    --out_dir outputs/rvqvae_seamless_multi_gpu/RVQVAE_Seamless_whole \

echo "训练完成!"