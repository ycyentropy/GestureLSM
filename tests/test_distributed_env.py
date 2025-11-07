#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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

def test_distributed_training(rank, world_size, gpu_ids):
    """测试分布式训练环境"""
    gpu_id = gpu_ids[rank]  # 使用指定的GPU ID
    print(f"Running DDP test on rank {rank}, using GPU {gpu_id}.")
    setup(rank, world_size)
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 5).cuda(gpu_id)
    ddp_model = DDP(model, device_ids=[gpu_id])
    
    # 创建一些随机数据
    inputs = torch.randn(20, 10).cuda(gpu_id)
    outputs = ddp_model(inputs)
    loss = outputs.sum()
    
    # 反向传播
    loss.backward()
    
    print(f"Rank {rank}: Model output shape: {outputs.shape}, Loss: {loss.item()}")
    
    # 清理
    cleanup()

def main():
    """主函数，启动多GPU测试"""
    # 检测可用的GPU数量
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行GPU训练")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 指定要使用的GPU（跳过GPU 0，使用GPU 1-7）
    gpu_ids = list(range(1, min(8, gpu_count)))  # [1, 2, 3, 4, 5, 6, 7]
    world_size = len(gpu_ids)
    print(f"使用GPU {gpu_ids} 进行测试，共 {world_size} 个GPU")
    
    # 启动多进程测试
    mp.spawn(test_distributed_training,
             args=(world_size, gpu_ids),
             nprocs=world_size,
             join=True)
    
    print("分布式训练环境测试完成!")

if __name__ == "__main__":
    main()