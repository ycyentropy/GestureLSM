#!/usr/bin/env python3
"""
简单测试训练脚本
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

def simple_collate_fn(batch):
    """
    简单的collate函数，处理不同大小的窗口
    """
    # 找到最大的序列长度
    max_len = max(item['pose'].shape[0] for item in batch)
    pose_dim = batch[0]['pose'].shape[1]

    # 创建填充后的批次
    poses = torch.zeros(len(batch), max_len, pose_dim)
    translations = torch.zeros(len(batch), max_len, 3)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item['pose'].shape[0]
        poses[i, :seq_len] = item['pose']

        # 处理translation - 确保形状匹配
        translation = item['translation']
        if translation.shape[0] != seq_len:
            # 如果translation的长度与seq_len不匹配，取前seq_len个
            translation = translation[:seq_len]
        translations[i, :seq_len] = translation

        masks[i, :seq_len] = True

    return {
        'pose': poses,
        'translation': translations,
        'mask': masks
    }

def test_simple_training():
    """测试简单训练"""
    print("=== 测试简单训练 ===")

    # 设置参数
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
    train_cache = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl"

    try:
        # 创建数据集
        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path=data_path,
            split="train",
            window_size=64,
            window_stride=20,
            cache_path=train_cache,
            load_audio=False,
            load_video=False,
            max_samples=50  # 限制样本数
        )
        print(f"数据集创建成功，总窗口数: {len(dataset)}")

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=simple_collate_fn
        )

        print("数据加载器创建成功")

        # 测试第一个批次
        print("\n=== 测试数据加载 ===")
        for i, batch in enumerate(dataloader):
            print(f"批次 {i}:")
            print(f"  姿态形状: {batch['pose'].shape}")
            print(f"  平移形状: {batch['translation'].shape}")
            print(f"  掩码形状: {batch['mask'].shape}")

            # 简单的前向传播测试
            if i >= 2:  # 只测试前3个批次
                break

        print("\n=== 测试成功！数据加载器已修复 ===")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_training()