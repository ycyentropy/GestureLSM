# GPU分配冲突错误分析与解决方案

## 错误分析

从终端输出可以看到以下关键错误信息：

```
Duplicate GPU detected : rank 2 and rank 0 both on CUDA device 27000
Duplicate GPU detected : rank 3 and rank 0 both on CUDA device 27000
...
torch.distributed.DistBackendError: NCCL error in: ... invalid usage
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
```

## 主要问题

1. **GPU设备分配冲突**：多个进程(rank)被分配到了同一个GPU设备上，这是分布式训练中的严重问题。

2. **CUDA_VISIBLE_DEVICES设置不当**：
   - 在`train_cached_7x4090.sh`中设置为`0,1,2,3,4,5,6`（7个GPU）
   - 但在`train_7x4090.sh`中没有显式设置，默认可能使用所有可用GPU

3. **进程数量与GPU数量不匹配**：
   - 脚本中设置了`WORLD_SIZE=7`（7个进程）
   - 但可能实际可用的GPU数量不匹配

4. **端口冲突**：多个训练脚本可能使用了相同的端口(29500)

## 解决方案

1. **修改GPU设置**：确保每个进程使用不同的GPU
2. **统一GPU配置**：在所有训练脚本中使用一致的GPU设置
3. **检查可用GPU数量**：确保进程数量不超过可用GPU数量
4. **使用不同端口**：避免端口冲突

## 修复后的脚本

我创建了`train_cached_7x4090_fixed.sh`脚本，主要修改：

1. **GPU设置**：使用`CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`，跳过GPU 0
2. **端口更改**：使用29501端口，避免与现有训练冲突
3. **输出目录**：使用新的输出目录`experiments/rvq_seamless_cached_7x4090_fixed`
4. **实验名称**：使用新的实验名称`rvq_seamless_cached_7x4090_fixed`

## 使用方法

```bash
./train_cached_7x4090_fixed.sh
```

## 其他建议

1. **检查GPU使用情况**：在运行训练前，使用`nvidia-smi`检查GPU使用情况
2. **停止冲突进程**：确保没有其他训练进程在使用相同的GPU或端口
3. **逐步测试**：可以先使用较少的GPU进行测试，确认无误后再使用全部GPU

## 预防措施

1. **统一GPU配置**：在所有训练脚本中使用一致的GPU设置
2. **使用唯一端口**：为每个训练任务分配唯一的端口号
3. **添加GPU检查**：在训练脚本中添加GPU可用性检查
4. **使用环境变量**：通过环境变量管理GPU和端口配置