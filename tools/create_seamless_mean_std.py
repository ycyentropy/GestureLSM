#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

def main():
    print("创建seamless_interaction数据集的均值和标准差文件...")
    
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
        print("创建默认的均值和标准差文件...")
        
        # 创建默认的均值和标准差文件，假设姿态数据维度为156
        seamless_mean = np.zeros(156)
        seamless_std = np.ones(156)
        
        # 保存结果
        os.makedirs("mean_std", exist_ok=True)
        np.save("mean_std/seamless_smplh_mean.npy", seamless_mean)
        np.save("mean_std/seamless_smplh_std.npy", seamless_std)
        
        print("默认的seamless_interaction均值和标准差文件已创建!")
        print("文件保存在 mean_std/seamless_smplh_mean.npy 和 mean_std/seamless_smplh_std.npy")

if __name__ == "__main__":
    main()