#!/usr/bin/env python3
"""
最小化训练测试，找出卡住的确切位置
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
import numpy as np
from torch.utils.data import DataLoader
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
from models.vq.model import RVQVAE
import time

def minimal_train_test():
    print("=== 最小化训练测试 ===")

    # 1. 创建数据集
    print("\n1. 创建数据集...")
    dataset = CachedLazySeamlessInteractionWindowDataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        split="train",
        window_size=64,
        window_stride=20,
        multi_length_training=[1.0],  # 只用单长度避免复杂性
        load_video=False,
        load_audio=False,
        max_samples=10,  # 只用10个样本
        cache_path="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
    )
    print(f"✓ 数据集创建成功，长度: {len(dataset)}")

    # 2. 创建DataLoader
    print("\n2. 创建DataLoader...")
    def simple_collate_fn(batch):
        max_len = max(item['pose'].shape[0] for item in batch)
        batch_size = len(batch)
        pose_dim = batch[0]['pose'].shape[1]

        poses = torch.zeros(batch_size, max_len, pose_dim)
        for i, item in enumerate(batch):
            pose = item['pose']
            length = pose.shape[0]
            poses[i, :length] = pose

        return {'pose': poses, 'mask': torch.ones(batch_size, max_len, dtype=torch.bool)}

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=simple_collate_fn
    )
    print(f"✓ DataLoader创建成功，长度: {len(dataloader)}")

    # 3. 创建模型
    print("\n3. 创建模型...")
    class Args:
        def __init__(self):
            self.code_dim = 128
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
    model = RVQVAE(
        args,
        156,  # input_dim
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

    device = torch.device("cuda:0")
    model.to(device)
    model.train()
    print(f"✓ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 4. 加载均值标准差
    print("\n4. 加载均值标准差...")
    mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
    std_pose = np.load('mean_std/seamless_smplh_std.npy')
    print(f"✓ 均值标准差加载成功，形状: {mean_pose.shape}")

    # 5. 创建优化器
    print("\n5. 创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    print("✓ 优化器创建成功")

    # 6. 训练循环测试
    print("\n6. 开始训练循环测试...")

    max_iter = 3
    for iter_num in range(max_iter):
        print(f"\n--- 迭代 {iter_num + 1}/{max_iter} ---")

        # 6.1 获取数据批次
        print("6.1 获取数据批次...")
        start_time = time.time()

        try:
            for batch in dataloader:
                batch_time = time.time()
                print(f"✓ 批次获取成功，耗时: {batch_time - start_time:.2f}秒")
                break
        except Exception as e:
            print(f"✗ 批次获取失败: {e}")
            return

        # 6.2 数据移到GPU
        print("6.2 数据移到GPU...")
        try:
            gt_motion = batch['pose'].to(device)
            batch_mask = batch['mask'].to(device)
            print(f"✓ 数据移到GPU成功，形状: {gt_motion.shape}")
        except Exception as e:
            print(f"✗ 数据移到GPU失败: {e}")
            return

        # 6.3 数据标准化
        print("6.3 数据标准化...")
        try:
            mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
            std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
            mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)
            std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)
            gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
            print("✓ 数据标准化成功")
        except Exception as e:
            print(f"✗ 数据标准化失败: {e}")
            return

        # 6.4 前向传播
        print("6.4 前向传播...")
        try:
            pred_motion, loss_commit, perplexity = model(gt_motion).values()
            print(f"✓ 前向传播成功")
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            return

        # 6.5 计算损失
        print("6.5 计算损失...")
        try:
            loss_motion = torch.nn.functional.l1_loss(pred_motion, gt_motion)
            loss = loss_motion + args.commit * loss_commit
            print(f"✓ 损失计算成功: {loss.item():.5f}")
        except Exception as e:
            print(f"✗ 损失计算失败: {e}")
            return

        # 6.6 反向传播
        print("6.6 反向传播...")
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("✓ 反向传播成功")
        except Exception as e:
            print(f"✗ 反向传播失败: {e}")
            return

        print(f"✅ 迭代 {iter_num + 1} 完成")

    print("\n🎉 所有测试通过！训练流程正常。")

if __name__ == "__main__":
    minimal_train_test()