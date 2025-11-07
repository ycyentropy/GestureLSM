# RVQ-VAE 优化版训练脚本使用说明

## 概述

`rvq_seamless_optimized.py` 是一个经过全面优化的 RVQ-VAE 训练脚本，专为 Seamless Interaction 数据集设计。该脚本包含以下优化功能：

1. **数据加载优化**：自动调整工作进程数、预取因子、使用快速批处理函数
2. **GPU内存优化**：自动调整批次大小、自适应批次处理、GPU内存监控
3. **分布式训练优化**：分布式基准测试、自动性能调优
4. **性能监控**：实时性能监控、自动保存、性能曲线绘制

## 快速开始

### 1. 基础训练

```bash
python /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
    --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
    --out-dir outputs/rvqvae_seamless_optimized \
    --batch-size 64 \
    --total-iter 10000 \
    --eval-iter 2000 \
    --print-iter 200
```

### 2. 使用运行脚本

我们提供了便捷的运行脚本，支持多种训练模式：

```bash
# 基础训练模式
bash run_optimized_training.sh basic

# 优化训练模式（启用所有优化功能）
bash run_optimized_training.sh optimized

# 缓存训练模式（使用预加载的数据缓存）
bash run_optimized_training.sh cache

# 分布式训练模式（使用2个GPU）
bash run_optimized_training.sh distributed

# 基准测试模式（运行性能基准测试）
bash run_optimized_training.sh benchmark
```

## 主要参数说明

### 数据加载优化参数

- `--num-workers`: 数据加载器工作进程数（默认自动优化）
- `--prefetch-factor`: 预取因子（默认自动优化）
- `--use-fast-collate`: 使用快速批处理函数（默认启用）
- `--pin-memory`: 使用固定内存（默认启用）
- `--non-blocking-transfer`: 使用非阻塞传输（默认启用）
- `--persistent-workers`: 保持工作进程活跃（默认启用）
- `--use-cache`: 是否使用缓存（默认禁用）

### GPU内存和计算优化参数

- `--max-batch-size`: 最大批次大小（默认256）
- `--auto-tune-batch-size`: 自动调整批次大小（默认启用）
- `--adaptive-batching`: 自适应批次大小（默认启用）
- `--target-gpu-util`: 目标GPU利用率（默认85.0%）
- `--gradient-accumulation-steps`: 梯度累积步数（默认1）
- `--use-amp`: 使用自动混合精度（默认启用）

### 分布式训练参数

- `--local-rank`: 本地进程排名（默认0）
- `--find-unused-parameters`: 查找未使用参数（默认禁用）
- `--benchmark-distributed`: 运行分布式基准测试（默认禁用）

### 性能监控参数

- `--auto-save-interval`: 自动保存间隔（默认3600秒）
- `--performance-log-interval`: 性能日志记录间隔（默认100迭代）
- `--auto-tune-performance`: 启用自动性能调优（默认禁用）

## 分布式训练

对于多GPU训练，可以使用以下命令：

```bash
# 使用2个GPU进行分布式训练
torchrun --nproc_per_node=2 /home/embodied/yangchenyu/GestureLSM/rvq_seamless_optimized.py \
    --data_path /home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction \
    --out-dir outputs/rvqvae_seamless_optimized \
    --batch-size 32 \
    --total-iter 10000 \
    --eval-iter 2000 \
    --print-iter 200
```

## 性能监控

脚本会自动记录以下性能指标：

- GPU内存使用情况
- 训练速度（迭代/秒）
- 批次大小变化
- 损失函数变化

性能报告和曲线图会保存在输出目录中。

## 故障排除

1. **内存不足**：减小 `--batch-size` 或启用 `--gradient-checkpointing`
2. **数据加载慢**：增加 `--num-workers` 或使用 `--use-cache`
3. **GPU利用率低**：启用 `--auto-tune-batch-size` 和 `--adaptive-batching`

## 注意事项

- 首次运行时，脚本会自动优化数据加载器参数
- 使用缓存可以显著提高数据加载速度，但需要额外的存储空间
- 分布式训练需要安装 PyTorch 的分布式组件
- 性能监控会增加少量开销，但可以帮助优化训练过程