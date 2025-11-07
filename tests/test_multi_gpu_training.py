#!/usr/bin/env python3
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def test_training(rank, world_size, gpu_ids):
    """测试训练环境"""
    gpu_id = gpu_ids[rank]  # 使用指定的GPU ID
    print(f"Running training test on rank {rank}, using GPU {gpu_id}.")
    setup(rank, world_size)
    
    # 创建一个更大的模型，更接近实际模型
    model = torch.nn.Sequential(
        torch.nn.Conv1d(156, 512, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(512, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(256, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(128, 156, kernel_size=3, padding=1)
    ).cuda(gpu_id)
    
    ddp_model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True, output_device=gpu_id)
    
    # 创建一些随机数据，模拟实际训练数据
    batch_size = 256  # 每个GPU的batch_size
    seq_len = 60
    input_dim = 156
    
    inputs = torch.randn(batch_size, input_dim, seq_len).cuda(gpu_id)
    targets = torch.randn(batch_size, seq_len, input_dim).cuda(gpu_id)
    
    # 创建优化器
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=2e-4)
    
    # 模拟训练循环
    for i in range(5):
        # 前向传播
        outputs = ddp_model(inputs)
        
        # 转置输出以匹配目标形状
        outputs = outputs.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}, Iter {i}: Loss {loss.item():.5f}")
    
    print(f"Rank {rank}: 训练测试完成!")
    
    # 清理
    cleanup()

def main():
    """主函数，启动多GPU测试"""
    parser = argparse.ArgumentParser(description='多GPU训练测试')
    parser.add_argument('--num_gpus', type=int, default=7, help='使用的GPU数量')
    args = parser.parse_args()
    
    # 检测可用的GPU数量
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行GPU训练")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 指定要使用的GPU（跳过GPU 0，使用GPU 1-7）
    gpu_ids = list(range(1, min(args.num_gpus + 1, gpu_count)))  # [1, 2, 3, 4, 5, 6, 7]
    world_size = len(gpu_ids)
    print(f"使用GPU {gpu_ids} 进行测试，共 {world_size} 个GPU")
    
    # 启动多进程测试
    mp.spawn(test_training,
             args=(world_size, gpu_ids),
             nprocs=world_size,
             join=True)
    
    print("多GPU训练测试完成!")

if __name__ == "__main__":
    main()