#!/usr/bin/env python3
"""
测试修复后的数据加载器
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

def test_fixed_dataloader():
    """测试修复后的数据加载器"""
    print("=== 测试修复后的数据加载器 ===")

    # 设置参数
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
    train_cache = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl"

    try:
        # 创建数据集
        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path=data_path,
            split="train",
            window_size=64,  # 这个参数会被调整为缓存的参数
            window_stride=20,  # 这个参数会被调整为缓存的参数
            cache_path=train_cache,
            load_audio=False,
            load_video=False,
            max_samples=100  # 限制样本数以便快速测试
        )
        print(f"数据集创建成功，总窗口数: {len(dataset)}")

        # 测试多个样本
        print("\n=== 测试前10个样本 ===")
        for i in range(10):
            try:
                sample = dataset[i]
                if 'pose' in sample:
                    print(f"样本 {i}: 姿态形状 {sample['pose'].shape}")
                else:
                    print(f"样本 {i}: 无姿态数据")
            except Exception as e:
                print(f"样本 {i} 错误: {e}")
                break

        print("\n=== 测试数据加载器功能 ===")

        # 测试数据加载
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            collate_fn=lambda batch: {
                'pose': torch.stack([item['pose'] for item in batch]),
                'translation': torch.stack([item['translation'] for item in batch])
            }
        )

        print("数据加载器创建成功")

        # 测试第一个批次
        print("\n=== 测试第一个批次 ===")
        for batch in dataloader:
            print(f"批次姿态形状: {batch['pose'].shape}")
            print(f"批次平移形状: {batch['translation'].shape}")
            break

        print("\n=== 测试成功 ===")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_dataloader()