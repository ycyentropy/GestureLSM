#!/usr/bin/env python3
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
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

class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def test_distributed_training(rank, world_size):
    """测试分布式训练"""
    print(f"运行分布式训练测试，rank {rank}, world_size {world_size}")
    
    # 设置分布式环境
    setup(rank, world_size)
    
    # 创建模型并移至GPU
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 创建随机数据
    batch_size = 32
    input_size = 10
    data = torch.randn(batch_size, input_size).to(rank)
    target = torch.randn(batch_size, input_size).to(rank)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 训练循环
    for epoch in range(5):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # 清理
    cleanup()

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # 从环境变量获取world_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Starting distributed training with local_rank={args.local_rank}, world_size={world_size}")
    
    # 设置分布式环境
    setup(args.local_rank, world_size)
    
    # 创建模型并移至GPU
    model = SimpleModel().to(args.local_rank)
    ddp_model = DDP(model, device_ids=[args.local_rank])
    
    # 创建随机数据
    batch_size = 32
    input_size = 10
    data = torch.randn(batch_size, input_size).to(args.local_rank)
    target = torch.randn(batch_size, input_size).to(args.local_rank)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 训练循环
    for epoch in range(5):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if args.local_rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # 清理
    cleanup()

if __name__ == "__main__":
    main()