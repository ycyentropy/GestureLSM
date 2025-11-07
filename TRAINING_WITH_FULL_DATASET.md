# 使用全部数据集进行训练

本文档说明如何使用全部数据集进行训练，以及相关的注意事项。

## 文件说明

1. **train_cached_7x4090_fixed.sh** - 修改后的脚本，已移除max_samples参数，将使用全部数据集
2. **train_cached_full_dataset.sh** - 新创建的脚本，专门用于使用全部数据集进行训练

## 使用方法

### 方法1：使用修改后的脚本
```bash
./train_cached_7x4090_fixed.sh
```

### 方法2：使用新创建的脚本
```bash
./train_cached_full_dataset.sh
```

## 两种方法的区别

两个脚本功能相同，都会使用全部数据集进行训练，主要区别在于：

1. **实验名称不同**：
   - `train_cached_7x4090_fixed.sh`：实验名称为 `rvq_seamless_cached_7x4090_fixed`
   - `train_cached_full_dataset.sh`：实验名称为 `rvq_seamless_cached_full_dataset`

2. **输出目录不同**：
   - `train_cached_7x4090_fixed.sh`：输出到 `experiments/rvq_seamless_cached_7x4090_fixed`
   - `train_cached_full_dataset.sh`：输出到 `experiments/rvq_seamless_cached_full_dataset`

## 注意事项

1. **训练时间**：使用全部数据集（约9009个样本）会显著增加训练时间，预计是使用1000个样本的9倍左右。

2. **存储空间**：窗口参数缓存文件会更大，需要确保有足够的存储空间。

3. **内存使用**：虽然窗口参数是缓存的，但训练过程中仍需要足够的内存来处理数据。

4. **GPU资源**：确保所有7个GPU（1-7）都可用且没有被其他进程占用。

## 渐进式训练建议

如果你不确定是否直接使用全部数据集，可以考虑以下渐进式方法：

1. 先使用少量样本（如1000）验证训练流程
2. 确认无误后，增加到中等数量（如5000）
3. 最后使用全部数据进行最终训练

## 性能对比

| 样本数量 | 预计时间窗口数 | 相对训练时间 |
|---------|--------------|------------|
| 1000    | ~270,000     | 1x         |
| 5000    | ~1,350,000   | 5x         |
| 9009    | ~2,432,430   | 9x         |

## 故障排除

如果遇到问题，可以：

1. 检查GPU状态：`nvidia-smi`
2. 检查是否有进程占用GPU：`ps aux | grep python`
3. 查看训练日志，检查是否有错误信息

## 相关文件

- `save_window_params.py` - 预计算窗口参数的脚本
- `rvq_seamless_cached_train.py` - 支持缓存的训练脚本
- `GPU_CONFLICT_ANALYSIS.md` - GPU冲突问题分析文档