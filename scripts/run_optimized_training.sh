#!/bin/bash

# RVQ-VAE 优化版训练脚本运行示例
# 使用方法: bash run_optimized_training.sh [模式]

# 设置默认模式
MODE=${1:-basic}

echo "======================================"
echo "RVQ-VAE 优化版训练脚本运行示例"
echo "======================================"

# 初始化训练状态
TRAINING_SUCCESS=0

case $MODE in
    "basic")
        echo "运行基础训练模式..."
        python /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
            --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
            --out-dir outputs/rvqvae_seamless_optimized \
            --batch-size 64 \
            --total-iter 10000 \
            --eval-iter 2000 \
            --print-iter 200 \
            --multi_length_training 0.5 0.75 1.0 1.25 1.5
        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            TRAINING_SUCCESS=1
        fi
        ;;
    
    "optimized")
        echo "运行优化训练模式..."
        python /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
            --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
            --out-dir outputs/rvqvae_seamless_optimized \
            --batch-size 64 \
            --total-iter 10000 \
            --eval-iter 2000 \
            --print-iter 200 \
            --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
            --auto-tune-batch-size \
            --adaptive-batching \
            --target-gpu-util 85.0 \
            --use-amp \
            --use-fast-collate \
            --pin-memory \
            --non-blocking-transfer \
            --persistent-workers
        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            TRAINING_SUCCESS=1
        fi
        ;;
    
    "cache")
        echo "运行缓存训练模式..."
        python /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
            --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
            --out-dir outputs/rvqvae_seamless_optimized \
            --batch-size 64 \
            --total-iter 10000 \
            --eval-iter 2000 \
            --print-iter 200 \
            --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
            --use-cache \
            --cache-path /home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl \
            --val-cache-path /home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20.pkl
        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            TRAINING_SUCCESS=1
        fi
        ;;
    
    "distributed")
        echo "运行分布式训练模式..."
        # 示例：使用8个GPU进行分布式训练
        TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONWARNINGS="ignore::FutureWarning" torchrun --nproc_per_node=8 --standalone /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
            --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
            --out-dir outputs/rvqvae_seamless_optimized \
            --batch-size 1024 \
            --total-iter 30000 \
            --eval-iter 5000 \
            --print-iter 500 \
            --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
            --use-amp \
            --pin-memory \
            --non-blocking-transfer \
            --use-cache \
            --cache-path /home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl \
            --val-cache-path /home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20.pkl
        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            TRAINING_SUCCESS=1
        fi
        ;;
    
    "benchmark")
        echo "运行基准测试模式..."
        python /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
            --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
            --out-dir outputs/rvqvae_seamless_optimized \
            --batch-size 64 \
            --total-iter 10000 \
            --eval-iter 2000 \
            --print-iter 200 \
            --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
            --benchmark-distributed \
            --auto-tune-performance
        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            TRAINING_SUCCESS=1
        fi
        ;;
    
    *)
        echo "未知模式: $MODE"
        echo "可用模式: basic, optimized, cache, distributed, benchmark"
        exit 1
        ;;
esac

# 只有在训练成功时才输出"训练完成！"
if [ $TRAINING_SUCCESS -eq 1 ]; then
    echo "训练完成！"
else
    echo "训练过程中出现错误，请检查日志以获取详细信息。"
    exit 1
fi