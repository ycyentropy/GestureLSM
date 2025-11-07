#!/usr/bin/env python3
"""
计算seamless_interaction数据集的均值和标准差（简化版）
使用惰性加载避免内存问题
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lazy_window_dataset import LazySeamlessInteractionWindowDataset

def compute_mean_std_lazy():
    """使用惰性加载计算seamless_interaction数据集的均值和标准差"""
    
    # 数据集路径
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train"
    
    print("创建惰性加载数据集...")
    start_time = time.time()
    
    # 创建惰性加载数据集
    dataset = LazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        pose_fps=30,
        window_size=64/30,  # 64帧窗口 (约2.13秒)
        window_stride=20/30,  # 20帧步长 (约0.67秒)
        load_video=False,
        load_audio=False,
        max_samples=1000  # 限制样本数量以加快处理速度
    )
    
    end_time = time.time()
    print(f"数据集创建完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总窗口数: {len(dataset)}")
    
    # 初始化累加器
    sum_poses = None
    sum_sq_poses = None
    count = 0
    max_samples = 500  # 只处理500个窗口
    
    # 随机采样窗口
    sampled_indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    
    print(f"计算均值和标准差（处理{len(sampled_indices)}个采样窗口）...")
    start_time = time.time()
    
    # 遍历采样的窗口
    for i, idx in enumerate(tqdm(sampled_indices)):
        try:
            # 获取单个窗口
            window = dataset[idx]
            
            # 获取姿态数据
            if 'pose' in window:
                poses = window["pose"]  # [seq_len, pose_dim]
                
                # 转换为numpy
                poses_np = poses.cpu().numpy()
                
                # 计算当前窗口的和
                window_sum = np.sum(poses_np, axis=0)  # [pose_dim]
                window_sum_sq = np.sum(poses_np ** 2, axis=0)  # [pose_dim]
                window_count = poses_np.shape[0]  # 总帧数
                
                # 更新累加器
                if sum_poses is None:
                    sum_poses = window_sum
                    sum_sq_poses = window_sum_sq
                else:
                    sum_poses += window_sum
                    sum_sq_poses += window_sum_sq
                
                count += window_count
                
                # 打印进度
                if i % 50 == 0:
                    print(f"已处理 {i+1} 个窗口，共 {count} 个帧")
        except Exception as e:
            print(f"处理窗口 {idx} 时出错: {e}")
            continue
    
    end_time = time.time()
    print(f"数据处理完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总共处理了 {count} 个帧")
    
    if count == 0:
        print("没有找到姿态数据，使用默认值")
        # 使用默认值
        mean_pose = np.zeros(156)
        std_pose = np.ones(156)
    else:
        # 计算均值和标准差
        mean_pose = sum_poses / count
        std_pose = np.sqrt(sum_sq_poses / count - mean_pose ** 2)
        
        # 确保标准差不为0
        std_pose = np.maximum(std_pose, 1e-8)
    
    # 保存均值和标准差
    os.makedirs("mean_std", exist_ok=True)
    np.save("mean_std/seamless_smplh_mean.npy", mean_pose)
    np.save("mean_std/seamless_smplh_std.npy", std_pose)
    
    print(f"均值和标准差已保存到: mean_std/seamless_smplh_mean.npy, mean_std/seamless_smplh_std.npy")
    print(f"数据集总帧数: {count}")
    print(f"姿态维度: {mean_pose.shape}")
    print(f"均值范围: [{mean_pose.min():.6f}, {mean_pose.max():.6f}]")
    print(f"标准差范围: [{std_pose.min():.6f}, {std_pose.max():.6f}]")
    
    return mean_pose, std_pose

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    compute_mean_std_lazy()