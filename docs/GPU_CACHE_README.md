# GPU缓存数据加载器

这个修改版本的训练脚本实现了将数据存储到显存中的功能，以减少系统内存的使用，特别适合您当前遇到的高内存占用、低显存占用的情况。

## 主要功能

### 1. GPU内存数据集 (GPUMemoryDataset)
- 将数据预加载到GPU显存中
- 可配置缓存大小，平衡显存使用和性能
- 自动管理缓存，替换最少使用的数据

### 2. 预取数据加载器 (PrefetchDataLoader)
- 提前将数据移动到显存
- 使用非阻塞传输，提高数据加载效率
- 可配置预取因子，平衡显存使用和预取效果

### 3. 内存监控
- 实时监控系统和GPU内存使用情况
- 定期打印内存状态
- 自动清理未使用的缓存

## 使用方法

### 基本使用

```bash
# 使用基础配置（GPU缓存数据集）
./run_gpu_cache.sh basic

# 使用预取配置（预取数据加载器）
./run_gpu_cache.sh prefetch

# 使用组合配置（GPU缓存 + 预取）
./run_gpu_cache.sh combined

# 使用激进配置（最大GPU缓存）
./run_gpu_cache.sh aggressive

# 使用最小配置（最少系统内存使用）
./run_gpu_cache.sh minimal
```

### 自定义参数

您也可以直接调用Python脚本并自定义参数：

```bash
python rvq_seamless_gpu_cache.py \
  --train_data_path datasets/train \
  --val_data_path datasets/val \
  --window_size 64 \
  --window_stride 20 \
  --batch_size 1024 \
  --use_gpu_cache \
  --gpu_cache_size 5000 \
  --num_workers 1
```

## 主要参数说明

### GPU缓存参数
- `--use_gpu_cache`: 是否使用GPU缓存数据集
- `--gpu_cache_size`: GPU缓存数据集大小（样本数量）
- `--use_prefetch`: 是否使用预取数据加载器
- `--prefetch_factor`: 预取因子（预取的批次数）

### DataLoader参数
- `--num_workers`: DataLoader工作进程数（建议设为0或1）
- `--pin_memory`: 是否使用固定内存（使用GPU缓存时建议关闭）
- `--persistent_workers`: 是否保持工作进程活跃

## 配置建议

### 高显存（≥24GB）系统
- 使用`aggressive`配置
- GPU缓存大小：10000个样本
- 预取因子：3
- 工作进程数：0

### 中等显存（12-24GB）系统
- 使用`combined`配置
- GPU缓存大小：3000个样本
- 预取因子：2
- 工作进程数：1

### 低显存（≤12GB）系统
- 使用`minimal`配置
- GPU缓存大小：2000个样本
- 预取因子：1
- 工作进程数：0
- 批次大小：512

## 性能对比

| 配置 | 系统内存使用 | GPU显存使用 | 数据加载速度 | 适用场景 |
|------|-------------|------------|-------------|---------|
| 原始 | 高 | 低 | 中等 | 内存充足 |
| GPU缓存 | 低 | 中等 | 快 | 显存充足 |
| 预取 | 中等 | 低 | 快 | 内存和显存都有限 |
| 组合 | 最低 | 中高 | 最快 | 平衡性能和资源使用 |

## 注意事项

1. **显存限制**：确保GPU显存足够容纳缓存的数据，否则可能导致OOM错误
2. **批次大小**：使用GPU缓存时，可能需要减小批次大小以释放更多显存
3. **工作进程数**：使用GPU缓存时，建议将`num_workers`设为0或1，避免额外的内存使用
4. **内存监控**：脚本会定期打印内存使用情况，根据实际情况调整缓存大小

## 故障排除

### OOM错误
- 减小`--gpu_cache_size`
- 减小`--batch_size`
- 减小`--prefetch_factor`

### 性能不佳
- 增加`--gpu_cache_size`（如果显存允许）
- 增加`--prefetch_factor`
- 尝试不同的`--num_workers`设置

### 内存仍然很高
- 确保`--num_workers`设为0或1
- 确保`--pin_memory`为false
- 尝试`minimal`配置

## 技术实现细节

### GPUMemoryDataset
- 继承自PyTorch的Dataset类
- 使用LRU缓存策略管理显存中的数据
- 自动将数据转换为CUDA张量

### PrefetchDataLoader
- 包装标准DataLoader
- 使用后台线程预取数据
- 使用非阻塞CUDA传输

### 内存监控
- 使用`psutil`监控系统内存
- 使用`torch.cuda`监控GPU显存
- 定期调用`torch.cuda.empty_cache()`清理缓存