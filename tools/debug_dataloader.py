#!/usr/bin/env python3
"""
逐步调试数据加载器问题
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
from torch.utils.data import DataLoader
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

def test_dataloader_step_by_step():
    print("=== 开始逐步调试数据加载器 ===")

    # 1. 测试缓存数据集创建
    print("\n1. 创建缓存数据集...")
    try:
        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
            split="train",
            window_size=64,
            window_stride=20,
            multi_length_training=[0.5, 0.75, 1.0, 1.25, 1.5],
            load_video=False,
            load_audio=False,
            max_samples=10,  # 只用10个样本快速测试
            cache_path="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
        )
        print(f"✓ 数据集创建成功，长度: {len(dataset)}")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 测试获取单个样本
    print("\n2. 测试获取单个样本...")
    try:
        sample = dataset[0]
        print(f"✓ 样本获取成功")
        print(f"  样本键: {list(sample.keys())}")
        if 'pose' in sample:
            print(f"  姿态形状: {sample['pose'].shape}")
    except Exception as e:
        print(f"✗ 样本获取失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 测试获取多个样本
    print("\n3. 测试获取多个样本...")
    try:
        samples = []
        for i in range(min(4, len(dataset))):
            sample = dataset[i]
            samples.append(sample)
            print(f"  样本 {i}: 姿态形状 {sample['pose'].shape}")
        print(f"✓ 获取 {len(samples)} 个样本成功")
    except Exception as e:
        print(f"✗ 获取多个样本失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 测试简单的collate函数
    print("\n4. 测试简单collate函数...")
    try:
        def simple_collate_fn(batch):
            """最简单的collate函数，只处理pose"""
            max_len = max(item['pose'].shape[0] for item in batch)
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]

            poses = torch.zeros(batch_size, max_len, pose_dim)
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = pose

            return {'pose': poses}

        collated = simple_collate_fn(samples)
        print(f"✓ 简单collate成功，形状: {collated['pose'].shape}")
    except Exception as e:
        print(f"✗ 简单collate失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 测试创建DataLoader
    print("\n5. 测试创建DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            drop_last=True,
            collate_fn=simple_collate_fn
        )
        print(f"✓ DataLoader创建成功，长度: {len(dataloader)}")
    except Exception as e:
        print(f"✗ DataLoader创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 测试获取第一个批次
    print("\n6. 测试获取第一个批次...")
    try:
        import time
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            end_time = time.time()
            print(f"✓ 批次 {i} 获取成功")
            print(f"  批次形状: {batch['pose'].shape}")
            print(f"  耗时: {end_time - start_time:.2f}秒")
            if i >= 0:  # 只测试第一个批次
                break
    except Exception as e:
        print(f"✗ 获取批次失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ 所有测试通过！数据加载器工作正常。")

if __name__ == "__main__":
    test_dataloader_step_by_step()