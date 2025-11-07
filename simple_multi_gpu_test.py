#!/usr/bin/env python3
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f'Running DDP example on rank {rank}.')
    setup(rank, world_size)
    # 创建模型并移动到GPU
    model = torch.nn.Linear(10, 10).to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print(f'Rank {rank}: Model successfully wrapped with DDP')
    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    world_size = 2
    run_demo(demo_basic, world_size)
    print('Multi-GPU test completed successfully!')