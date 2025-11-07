#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.seamless_interaction import SeamlessInteractionWindowDataset

def main():
    print("开始计算seamless_interaction数据集的均值和标准差...")
    
    # 创建数据集实例
    print("创建数据集实例...")
    dataset = SeamlessInteractionWindowDataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        split="train",
        window_size=1.0,  # 1秒窗口
        window_stride=0.5,  # 0.5秒步长
        sample_rate=16000,
        pose_fps=30,
        audio_fps=16000,
        load_video=False,
        load_audio=False
    )
    
    print(f"数据集创建成功，共有 {len(dataset)} 个样本")
    
    # 只加载第一个样本进行测试
    print("加载第一个样本进行测试...")
    sample = dataset[0]
    print(f"样本键: {sample.keys()}")
    
    if 'pose' in sample:
        pose_data = sample['pose']
        print(f"姿态数据形状: {pose_data.shape}")
        print(f"姿态数据类型: {pose_data.dtype}")
        print(f"姿态数据范围: [{pose_data.min()}, {pose_data.max()}]")
        
        # 计算这个样本的均值和标准差
        sample_mean = pose_data.mean(dim=0)
        sample_std = pose_data.std(dim=0)
        
        print(f"样本均值形状: {sample_mean.shape}")
        print(f"样本标准差形状: {sample_std.shape}")
        
        # 保存结果
        os.makedirs("mean_std", exist_ok=True)
        np.save("mean_std/seamless_smplh_mean_test.npy", sample_mean.cpu().numpy())
        np.save("mean_std/seamless_smplh_std_test.npy", sample_std.cpu().numpy())
        
        print("测试样本的均值和标准差已保存到 mean_std/seamless_smplh_mean_test.npy 和 mean_std/seamless_smplh_std_test.npy")
    else:
        print("样本中没有姿态数据!")
        print(f"可用的键: {list(sample.keys())}")

if __name__ == "__main__":
    main()