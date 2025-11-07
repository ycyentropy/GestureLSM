#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import argparse

def test_gpu_setup():
    """测试GPU设置"""
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查当前环境变量
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")

def test_simple_model():
    """测试简单模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    # 创建随机数据
    batch_size = 32
    input_size = 10
    data = torch.randn(batch_size, input_size).to(device)
    target = torch.randn(batch_size, input_size).to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 训练循环
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("简单模型测试完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    print(f"运行简单GPU测试，local_rank: {args.local_rank}")
    
    # 测试GPU设置
    test_gpu_setup()
    
    # 测试简单模型
    test_simple_model()

if __name__ == "__main__":
    main()