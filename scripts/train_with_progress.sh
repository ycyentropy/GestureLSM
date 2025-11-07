#!/bin/bash

echo "🚀 启动带进度显示的多GPU训练..."
echo "📊 使用2个GPU进行快速测试"
echo "📂 缓存文件: window_params_train_ws64_ws20_fixed.pkl"
echo "🎯 训练5次迭代，50个样本"

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29510 \
    rvq_seamless_multi_gpu_progress.py \
    --batch_size 32 \
    --cache_train datasets/window_params/window_params_train_ws64_ws20_fixed.pkl \
    --cache_val datasets/window_params/window_params_val_ws64_ws20_fixed.pkl \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
    --total_iter 5 \
    --max_samples 10000 \
    --eval_iter 5 \
    --out_dir experiments/rvq_seamless_progress_test

echo "✅ 训练启动完成！"