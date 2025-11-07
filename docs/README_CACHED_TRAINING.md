# 使用缓存窗口参数的训练脚本

## 概述

我已经为您创建了使用缓存窗口参数的训练脚本，这将显著减少每次训练时的数据加载时间，避免重复遍历所有样本来计算窗口参数。

## 文件说明

1. **save_window_params.py** - 用于预计算和保存窗口参数的脚本
2. **rvq_seamless_cached_train.py** - 修改后的训练脚本，支持加载缓存的窗口参数
3. **train_cached_7x4090.sh** - 使用缓存的窗口参数进行训练的启动脚本
4. **example_cached_dataset.py** - 使用示例和性能比较脚本

## 使用方法

### 1. 预计算并保存窗口参数

```bash
# 为训练集计算窗口参数
python save_window_params.py \
    --data_path data/seamless_motions \
    --split train \
    --window_size 60 \
    --window_stride 30 \
    --multi_length_training \
    --max_samples 1000 \
    --output_path data/window_params

# 为验证集计算窗口参数
python save_window_params.py \
    --data_path data/seamless_motions \
    --split dev \
    --window_size 60 \
    --window_stride 30 \
    --multi_length_training \
    --max_samples 1000 \
    --output_path data/window_params
```

### 2. 使用缓存的窗口参数进行训练

```bash
# 使用启动脚本进行训练
./train_cached_7x4090.sh
```

或者直接运行训练脚本：

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=7 \
    --master_port=29500 \
    rvq_seamless_cached_train.py \
    --data_path data/seamless_motions \
    --window_size 60 \
    --window_stride 30 \
    --batch_size 32 \
    --epochs 100 \
    --lr 2e-4 \
    --exp_name rvq_seamless_cached_7x4090 \
    --out_dir experiments/rvq_seamless_cached_7x4090_whole \
    --use_cache \
    --multi_length_training \
    --max_samples 1000
```

## 性能比较

您可以使用 `example_cached_dataset.py` 来比较使用缓存和常规方式加载数据集的性能差异：

```bash
# 比较加载时间
python example_cached_dataset.py --compare_loading_time

# 查看使用示例
python example_cached_dataset.py --show_example
```

## 实现原理

1. **预计算窗口参数**：在首次加载时，计算所有样本的窗口参数并保存到文件中。
2. **缓存加载**：后续训练时，直接从缓存文件加载窗口参数，避免重复计算。
3. **兼容性**：如果缓存文件不存在，会自动回退到常规方式加载。

## 参数说明

- `--use_cache`：是否使用缓存的窗口参数
- `--save_cache`：是否保存窗口参数到缓存
- `--cache_path`：指定缓存文件路径（可选，默认自动生成）

## 注意事项

1. 当数据集或窗口参数（窗口大小、步长等）发生变化时，需要重新生成缓存文件。
2. 缓存文件会保存在 `data/window_params/` 目录下，文件名包含窗口参数信息以避免混淆。
3. 使用多GPU训练时，只有主进程会保存缓存文件。

通过这种方式，您可以显著减少数据加载时间，特别是在数据集较大或窗口计算复杂的情况下。