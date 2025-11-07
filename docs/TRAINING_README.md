# GestureLSM 训练指南

本文档介绍如何使用GestureLSM进行多GPU训练，包括环境配置、数据准备、训练启动和故障排除。

## 目录

- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [缓存生成](#缓存生成)
- [训练配置](#训练配置)
- [单GPU训练](#单gpu训练)
- [多GPU训练](#多gpu训练)
- [监控训练](#监控训练)
- [故障排除](#故障排除)

## 环境要求

### 硬件要求
- **GPU**: NVIDIA RTX 4090 (推荐8卡用于多GPU训练)
- **内存**: 至少32GB RAM
- **存储**: 至少100GB可用空间用于数据和模型

### 软件要求
- Python 3.10+
- PyTorch 2.0+
- CUDA 12.2+
- 其他依赖见 `requirements.txt`

## 数据准备

### 1. 数据集结构
确保数据集按以下结构组织：
```
datasets/seamless_interaction/
├── train/
│   ├── sample1/
│   │   ├── keypoints.npy
│   │   ├── pose.npy
│   │   ├── translation.npy
│   │   └── ...
│   └── ...
└── val/
    └── ...
```

### 2. 检查数据集
```bash
# 检查训练样本数量
python -c "
from dataloaders.seamless_interaction import SeamlessInteractionDataset
dataset = SeamlessInteractionDataset('datasets/seamless_interaction', split='train')
print(f'训练样本数: {len(dataset)}')
"
```

## 缓存生成

### 多长度训练缓存

多长度训练使用不同的窗口大小比例：[0.5, 0.75, 1.0, 1.25, 1.5]

#### 生成缓存文件
```bash
# 生成训练集缓存
python save_window_params.py \
    --data_path datasets/seamless_interaction \
    --split train \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
    --pose_fps 30 \
    --audio_fps 16000 \
    --save_path datasets/window_params/window_params_train_ws64_ws20_fixed.pkl

# 生成验证集缓存
python save_window_params.py \
    --data_path datasets/seamless_interaction \
    --split val \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
    --pose_fps 30 \
    --audio_fps 16000 \
    --save_path datasets/window_params/window_params_val_ws64_ws20_fixed.pkl
```

#### 验证缓存文件
```bash
# 检查缓存分布
python check_cache_distribution.py

# 测试缓存加载
python test_fixed_cache.py
```

## 训练配置

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 1024 | 批次大小 |
| `window_size` | 64 | 基础窗口大小(帧) |
| `window_stride` | 20 | 窗口步长(帧) |
| `multi_length_training` | [0.5,0.75,1.0,1.25,1.5] | 多长度训练比例 |
| `total_iter` | 10000 | 总训练迭代数 |
| `lr` | 0.0002 | 学习率 |
| `nb_code` | 1024 | 码本大小 |
| `num_quantizers` | 8 | 量化器数量 |

### 配置文件示例

创建配置文件 `configs/rvq_seamless_train.yaml`:
```yaml
batch_size: 1024
window_size: 64
window_stride: 20
multi_length_training: [0.5, 0.75, 1.0, 1.25, 1.5]
total_iter: 10000
lr: 0.0002
lr_scheduler: [5000, 8000]
nb_code: 1024
num_quantizers: 8
code_dim: 128
width: 512
depth: 3
recons_loss: "l1_smooth"
commit: 0.02
mu: 0.99
quantize_dropout_prob: 0.5
exp_name: "rvq_seamless_experiment"
out_dir: "experiments/rvq_seamless_experiment_whole"
seed: 42
print_iter: 100
eval_iter: 1000
```

## 单GPU训练

### 基础训练
```bash
# 使用预生成缓存
python rvq_seamless_train.py \
    --batch_size 256 \
    --window_size 64 \
    --window_stride 20 \
    --cache_train datasets/window_params/window_params_train_ws64_ws20_fixed.pkl \
    --cache_val datasets/window_params/window_params_val_ws64_ws20_fixed.pkl \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5

# 不使用缓存（实时计算窗口参数）
python rvq_seamless_train.py \
    --batch_size 256 \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5 \
    --use_cache False
```

### 测试训练
```bash
# 小样本快速测试
python rvq_seamless_train.py \
    --batch_size 32 \
    --max_samples 100 \
    --total_iter 100 \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5
```

## 多GPU训练

### 检查GPU状态
```bash
# 检查可用GPU
nvidia-smi

# 检查GPU内存使用
watch -n 1 nvidia-smi
```

### 启动多GPU训练

#### 方法1: 使用torch.distributed.launch (推荐)
```bash
# 8卡训练
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29501 \
    rvq_seamless_multi_gpu.py \
    --batch_size 1024 \
    --cache_train datasets/window_params/window_params_train_ws64_ws20_fixed.pkl \
    --cache_val datasets/window_params/window_params_val_ws64_ws20_fixed.pkl \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5

# 4卡训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29501 \
    rvq_seamless_multi_gpu.py \
    --batch_size 512 \
    --cache_train datasets/window_params/window_params_train_ws64_ws20_fixed.pkl \
    --cache_val datasets/window_params/window_params_val_ws64_ws20_fixed.pkl \
    --window_size 64 \
    --window_stride 20 \
    --multi_length_training 0.5 0.75 1.0 1.25 1.5
```

#### 方法2: 使用训练脚本
```bash
# 使用便捷脚本
./train_multi_gpu_fixed_cache.sh

# 自定义配置
./train_multi_gpu_custom.sh \
    --n_gpus 8 \
    --batch_size 1024 \
    --master_port 29501
```

### GPU内存优化
```bash
# 如果遇到GPU内存不足，可以：
# 1. 减少批次大小
--batch_size 512

# 2. 启用梯度检查点
--gradient_checkpointing True

# 3. 使用混合精度训练
--mixed_precision True
```

## 监控训练

### 实时监控
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控训练进程
ps aux | grep rvq_seamless

# 查看训练日志
tail -f experiments/rvq_seamless_experiment_whole/train.log
```

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir experiments/rvq_seamless_experiment_whole

# 访问 http://localhost:6006 查看训练曲线
```

### 检查训练输出
```bash
# 查看实验目录
ls -la experiments/rvq_seamless_experiment_whole/

# 查看模型检查点
ls -la experiments/rvq_seamless_experiment_whole/checkpoints/

# 查看最新日志
tail -n 50 experiments/rvq_seamless_experiment_whole/train.log
```

## 故障排除

### 常见问题

#### 1. GPU内存不足
**症状**: `CUDA out of memory` 错误
**解决方案**:
```bash
# 减少批次大小
--batch_size 256

# 启用梯度检查点
--gradient_checkpointing True

# 使用更少的GPU
--nproc_per_node 4
```

#### 2. 数据加载缓慢
**症状**: 数据加载阶段耗时过长
**解决方案**:
```bash
# 使用预生成缓存
--cache_train datasets/window_params/window_params_train_ws64_ws20_fixed.pkl

# 增加数据加载器worker数量
--num_workers 8
```

#### 3. 多长度训练窗口形状不匹配
**症状**: `RuntimeError: shape '[4,64,156]' is invalid for input of size XXX`
**解决方案**:
- 确保使用正确的collate函数处理变长窗口
- 检查缓存文件是否正确生成

#### 4. 缓存参数不匹配
**症状**: `警告: 缓存文件窗口大小与请求的窗口大小不匹配`
**解决方案**:
- 重新生成缓存文件确保参数一致
- 或使用警告提示的缓存参数值

#### 5. 进程挂起
**症状**: 训练进程无响应
**诊断和解决**:
```bash
# 检查进程状态
ps aux | grep rvq_seamless

# 检查GPU使用
nvidia-smi

# 终止挂起进程
pkill -f rvq_seamless

# 重新启动训练
```

### 调试技巧

#### 1. 小样本测试
```bash
# 使用少量样本快速测试配置
python rvq_seamless_train.py \
    --max_samples 10 \
    --total_iter 10 \
    --batch_size 2 \
    --window_size 64 \
    --window_stride 20
```

#### 2. 验证数据加载
```bash
# 测试数据集加载
python test_seamless_dataset.py

# 测试缓存数据集
python test_fixed_cache.py
```

#### 3. 检查模型架构
```bash
# 查看模型参数数量
python -c "
from models.rqvae import RQVAE
model = RQVAE(nb_code=1024, num_quantizers=8, code_dim=128)
print(f'模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

## 性能优化建议

### 1. 数据预处理
- 提前生成并验证缓存文件
- 使用SSD存储数据集
- 确保数据格式为NPY以加快加载速度

### 2. 批次大小调优
- 根据GPU内存调整批次大小
- 8卡RTX 4090建议使用batch_size=1024
- 4卡RTX 4090建议使用batch_size=512

### 3. 多长度训练优化
- 缓存多长度窗口参数避免重复计算
- 使用合适的collate函数处理变长序列
- 考虑使用padding策略统一批次内序列长度

### 4. 分布式训练优化
- 确保所有GPU可用且内存充足
- 使用NCCL后端优化通信
- 设置合适的master_port避免冲突

## 训练脚本参考

### 完整训练脚本示例
```bash
#!/bin/bash
# train_complete.sh

echo "=== 开始GestureLSM多GPU训练 ==="

# 配置参数
N_GPUS=8
BATCH_SIZE=1024
WINDOW_SIZE=64
WINDOW_STRIDE=20
MULTI_LENGTH="0.5 0.75 1.0 1.25 1.5"

# 缓存路径
CACHE_TRAIN="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
CACHE_VAL="datasets/window_params/window_params_val_ws64_ws20_fixed.pkl"

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 检查缓存文件
if [ ! -f "$CACHE_TRAIN" ]; then
    echo "错误: 训练缓存文件不存在: $CACHE_TRAIN"
    exit 1
fi

if [ ! -f "$CACHE_VAL" ]; then
    echo "错误: 验证缓存文件不存在: $CACHE_VAL"
    exit 1
fi

# 启动训练
echo "启动多GPU训练..."
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPUS \
    --master_port=29501 \
    rvq_seamless_multi_gpu.py \
    --batch_size $BATCH_SIZE \
    --cache_train $CACHE_TRAIN \
    --cache_val $CACHE_VAL \
    --window_size $WINDOW_SIZE \
    --window_stride $WINDOW_STRIDE \
    --multi_length_training $MULTI_LENGTH

echo "训练完成!"
```

使用方法:
```bash
chmod +x train_complete.sh
./train_complete.sh
```

---

**注意**: 训练前请确保：
1. 所有GPU可用且无其他程序占用
2. 数据集路径正确
3. 缓存文件已生成
4. 配置参数合理

如有问题，请参考[故障排除](#故障排除)部分或查看训练日志。