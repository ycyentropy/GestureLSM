#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from rvq_seamless_train import main as single_gpu_main, get_args_parser

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

def train_multi_gpu(rank, world_size, args):
    """多GPU训练函数"""
    print(f"Running DDP example on rank {rank}.")
    setup(rank, world_size)
    
    # 设置当前使用的GPU
    torch.cuda.set_device(rank)
    
    # 修改参数以适应多GPU训练
    args.batch_size = args.batch_size // world_size  # 每个GPU的batch_size
    args.rank = rank
    args.world_size = world_size
    
    # 运行单GPU训练函数，但会使用分布式环境
    single_gpu_main(args)
    
    cleanup()

def main():
    """主函数，启动多GPU训练"""
    args = get_args_parser()
    
    # 检测可用的GPU数量
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行GPU训练")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 指定要使用的GPU数量（默认使用所有可用的GPU）
    world_size = args.world_size if hasattr(args, 'world_size') else gpu_count
    print(f"使用 {world_size} 个GPU进行训练")
    
    # 启动多进程训练
    mp.spawn(train_multi_gpu,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()