import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
import os
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

from models.vq.model import RVQVAE

def setup(rank, world_size, gpu_ids):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_ids[rank])

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train(rank, world_size, batch_size, max_samples, gpu_ids):
    """在单个GPU上运行训练函数"""
    setup(rank, world_size, gpu_ids)
    gpu_id = gpu_ids[rank]
    
    # 创建与训练脚本相同的参数
    args = argparse.Namespace()
    args.num_quantizers = 6
    args.shared_codebook = False
    args.quantize_dropout_prob = 0.2
    args.nb_code = 512
    args.code_dim = 256
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99
    
    # 创建模型
    model = RVQVAE(
        args,
        156,  # pose_dim
        args.nb_code,
        args.code_dim,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    )
    
    model.cuda(gpu_id)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    
    # 创建测试数据 (与训练脚本相同)
    seq_len = 64  # 窗口大小
    pose_dim = 156  # 全身姿态维度
    
    # 模拟数据加载器
    num_windows = max_samples * 270  # 每个样本约产生270个时间窗口
    num_batches = num_windows // (batch_size * world_size)
    
    if rank == 0:  # 只在主进程打印
        print(f"使用GPU: {gpu_ids}")
        print(f"GPU {gpu_id}: Batch Size: {batch_size}, 总有效Batch Size: {batch_size * world_size}")
        print(f"GPU {gpu_id}: 使用{max_samples}个样本，约{num_windows}个时间窗口，{num_batches}个批次")
    
    # 记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    
    # 模拟训练过程
    model.train()
    for i in range(5):  # 只运行5个批次作为测试
        optimizer.zero_grad()
        
        # 创建随机数据
        test_motion = torch.randn(batch_size, seq_len, pose_dim).cuda(gpu_id)
        test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(gpu_id)
        
        # 前向传播
        output = model(test_motion)
        pred_motion = output['rec_pose']
        loss_commit = output['commit_loss']
        
        # 计算损失 (与训练脚本相同)
        loss_motion = nn.L1Loss()(pred_motion, test_motion)
        loss = loss_motion + 0.02 * loss_commit
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if rank == 0:  # 只在主进程打印
            print(f"  批次 {i+1}/5: Loss = {loss.item():.6f}")
    
    # 记录峰值显存
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    used_memory = peak_memory - initial_memory
    
    if rank == 0:  # 只在主进程打印
        print(f"\nGPU {gpu_id} 显存使用情况:")
        print(f"  峰值显存占用: {peak_memory:.2f} MB")
        print(f"  实际使用显存: {used_memory:.2f} MB ({used_memory/1024:.2f} GB)")
        
        gpu_info = torch.cuda.get_device_properties(gpu_id)
        total_gpu_memory = gpu_info.total_memory / (1024 ** 3)  # GB
        print(f"  显存利用率: {used_memory/1024/total_gpu_memory * 100:.1f}%")
        
        # 计算总显存使用情况
        total_used_memory = used_memory * world_size
        total_gpu_memory_all = total_gpu_memory * world_size
        print(f"\n所有GPU总显存使用情况:")
        print(f"  总使用显存: {total_used_memory/1024:.2f} GB")
        print(f"  总可用显存: {total_gpu_memory_all:.2f} GB")
        print(f"  总显存利用率: {total_used_memory/1024/total_gpu_memory_all * 100:.1f}%")
        
        # 估算训练时间
        print(f"\n训练时间估算:")
        print(f"  每批次时间: ~0.5秒 (估算)")
        print(f"  总批次数: {num_batches}")
        print(f"  总训练时间: ~{num_batches * 0.5 / 3600:.2f}小时")
    
    cleanup()

def main():
    # 获取GPU信息
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")
    
    # 检查每个GPU的显存使用情况
    gpu_info_list = []
    for i in range(num_gpus):
        gpu_info = torch.cuda.get_device_properties(i)
        total_gpu_memory = gpu_info.total_memory / (1024 ** 3)  # GB
        gpu_info_list.append((i, gpu_info, total_gpu_memory))
        print(f"GPU {i}: {gpu_info.name}, 总显存: {total_gpu_memory:.2f} GB")
    
    # 检查GPU 0的显存使用情况
    torch.cuda.set_device(0)
    gpu0_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    print(f"GPU 0 当前使用显存: {gpu0_memory_used:.2f} GB")
    
    # 选择空闲的GPU (GPU 1-7)
    available_gpus = list(range(1, num_gpus))
    print(f"\n选择空闲GPU: {available_gpus}")
    
    # 测试配置
    world_size = len(available_gpus)  # 使用空闲GPU
    batch_size = 8192  # 每个GPU的batch_size
    max_samples = 5000  # 使用5000个样本
    
    print(f"\n测试配置:")
    print(f"使用GPU数量: {world_size}")
    print(f"每个GPU的Batch Size: {batch_size}")
    print(f"总有效Batch Size: {batch_size * world_size}")
    print(f"使用样本数量: {max_samples}")
    print(f"预计时间窗口数: {max_samples * 270}")
    
    print("\n开始多GPU训练测试...")
    mp.spawn(train,
             args=(world_size, batch_size, max_samples, available_gpus),
             nprocs=world_size,
             join=True)
    
    print("\n推荐的多GPU训练命令:")
    print(f"python -m torch.distributed.launch --nproc_per_node={world_size} rvq_seamless_train.py --mode train --batch-size {batch_size} --max-samples {max_samples}")
    
    # 提供单GPU训练的推荐配置
    print("\n单GPU训练推荐配置:")
    print(f"如果只使用一个GPU，推荐batch_size=8192，max_samples=5000")
    print(f"训练命令: python rvq_seamless_train.py --mode train --batch-size 8192 --max-samples 5000")

if __name__ == "__main__":
    main()