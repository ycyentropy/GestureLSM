#!/usr/bin/env python3
"""
测试GPU内存问题
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
import numpy as np

def test_gpu_memory():
    print("=== 测试GPU内存 ===")

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return

    print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")

    # 检查每个GPU的内存状态
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB

        print(f"GPU {i}:")
        print(f"  总内存: {total_memory:.1f}GB")
        print(f"  已预留: {reserved_memory:.1f}GB")
        print(f"  已分配: {allocated_memory:.1f}GB")
        print(f"  可用: {total_memory - reserved_memory:.1f}GB")

    # 测试创建模型
    print("\n=== 测试创建模型 ===")
    try:
        from models.vq.model import RVQVAE

        # 模型参数
        class Args:
            def __init__(self):
                self.code_dim = 256
                self.down_t = 2
                self.stride_t = 2
                self.width = 512
                self.depth = 3
                self.dilation_growth_rate = 3
                self.vq_act = 'relu'
                self.vq_norm = None
                self.num_quantizers = 8
                self.nb_code = 1024
                self.commit = 0.02
                self.mu = 0.99
                self.quantize_dropout_prob = 0.5
                self.recons_loss = 'l1_smooth'
                self.loss_vel = 0.0
                self.shared_codebook = False

        args = Args()
        input_dim = 156  # 姿态维度

        print("创建模型...")
        model = RVQVAE(
            args,
            input_dim,
            1024,  # nb_code
            128,  # code_dim
            128,  # embed_dim
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm
        )

        print(f"✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # 测试将模型移到GPU
        print("\n测试将模型移到GPU 0...")
        device = torch.device("cuda:0")
        model.to(device)
        print("✅ 模型成功移到GPU 0")

        # 测试创建批次数据
        print("\n测试创建批次数据...")
        batch_size = 64
        seq_len = 96  # 使用最大长度
        pose_dim = 156

        # 创建随机数据
        test_data = torch.randn(batch_size, seq_len, pose_dim, device=device)
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        print(f"   内存占用: {test_data.numel() * test_data.element_size() / 1024**3:.2f}GB")

        # 测试前向传播
        print("\n测试前向传播...")
        with torch.no_grad():
            output = model(test_data)
            print(f"✅ 前向传播成功")
            print(f"   输出形状: {[k.shape for k, v in output.items() for k in [k]] if isinstance(output, dict) else output.shape}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_memory()