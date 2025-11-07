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
    print("开始测试seamless_interaction数据集加载...")
    
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
    
    # 只测试数据集长度，不加载实际数据
    print("数据集测试完成!")
    
    # 创建一个假的均值和标准差文件，基于BEAT数据集的形状
    print("创建示例均值和标准差文件...")
    
    # 从BEAT数据集加载现有的均值和标准差作为参考
    beat_mean_path = "mean_std/beatx_2_330_mean.npy"
    beat_std_path = "mean_std/beatx_2_330_std.npy"
    
    if os.path.exists(beat_mean_path) and os.path.exists(beat_std_path):
        beat_mean = np.load(beat_mean_path)
        beat_std = np.load(beat_std_path)
        print(f"BEAT均值形状: {beat_mean.shape}, 标准差形状: {beat_std.shape}")
        
        # 创建seamless_interaction的均值和标准差文件，使用相同的形状
        seamless_mean = np.zeros_like(beat_mean)
        seamless_std = np.ones_like(beat_std)
        
        # 保存结果
        os.makedirs("mean_std", exist_ok=True)
        np.save("mean_std/seamless_smplh_mean.npy", seamless_mean)
        np.save("mean_std/seamless_smplh_std.npy", seamless_std)
        
        print("seamless_interaction的均值和标准差文件已创建!")
        print("文件保存在 mean_std/seamless_smplh_mean.npy 和 mean_std/seamless_smplh_std.npy")
    else:
        print("找不到BEAT数据集的均值和标准差文件作为参考!")

if __name__ == "__main__":
    main()