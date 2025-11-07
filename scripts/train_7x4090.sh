#!/bin/bash

# 7个4090显卡最大化显存使用训练脚本 (GPU 1-7)
# 使用方法: bash train_7x4090.sh
# GPU 0保留给其他用户

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# 训练参数
DATA_PATH="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
OUT_DIR="experiments"
EXP_NAME="rvq_seamless_7x4090"

# 模型参数 - 基于RVQ-VAE架构文档配置
NB_CODE=1024          # 码本大小（与架构文档一致）
CODE_DIM=128           # 码本维度（与架构文档一致）
DOWN_T=2               # 下采样层数（与架构文档一致）
STRIDE_T=2             # 时间步长（与架构文档一致）
WIDTH=512              # 网络宽度（与架构文档一致）
DEPTH=3                # 网络深度（与架构文档一致）
DILATION_GROWTH_RATE=3 # 膨胀增长率（与架构文档一致）
VQ_ACT="relu"          # 激活函数（与架构文档一致）
NUM_QUANTIZERS=6       # 量化器数量（与架构文档一致）
SHARED_CODEBOOK=False  # 不共享码本
QUANTIZE_DROPOUT_PROB=0.5

# 训练参数 - 基于RVQ-VAE架构文档配置
BATCH_SIZE=256         # 每个GPU的批次大小
TOTAL_ITER=300000     # 总迭代次数（与架构文档一致）
LR=2e-4               # 学习率（与架构文档一致）
LR_SCHEDULER="[50000, 200000, 400000]"  # 学习率调度器里程碑
GAMMA=0.05            # 学习率衰减
COMMIT=0.02           # 提交损失（与架构文档一致）
LOSS_VEL=0.5          # 速度损失（与架构文档一致）
RECONS_LOSS="l1_smooth"

# 数据参数 - 基于RVQ-VAE架构文档配置
WINDOW_SIZE=64        # 窗口大小（帧数）
WINDOW_STRIDE=20       # 窗口步长（帧数）
MULTI_LENGTH_TRAINING=true  # 是否启用多长度训练
MAX_SAMPLES=None       # 使用整个数据集

# 输出参数
PRINT_ITER=100
EVAL_ITER=1000
SEED=42

# 运行训练命令
torchrun --nproc_per_node=7 --master_port=29500 rvq_seamless_multi_gpu.py \
    --data_path $DATA_PATH \
    --out_dir $OUT_DIR \
    --exp_name $EXP_NAME \
    --nb_code $NB_CODE \
    --code_dim $CODE_DIM \
    --down_t $DOWN_T \
    --stride_t $STRIDE_T \
    --width $WIDTH \
    --depth $DEPTH \
    --dilation_growth_rate $DILATION_GROWTH_RATE \
    --vq_act $VQ_ACT \
    --num_quantizers $NUM_QUANTIZERS \
    --shared_codebook $SHARED_CODEBOOK \
    --quantize_dropout_prob $QUANTIZE_DROPOUT_PROB \
    --batch_size $BATCH_SIZE \
    --total_iter $TOTAL_ITER \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --gamma $GAMMA \
    --commit $COMMIT \
    --loss_vel $LOSS_VEL \
    --recons_loss $RECONS_LOSS \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --multi_length_training \
    --print_iter $PRINT_ITER \
    --eval_iter $EVAL_ITER \
    --seed $SEED