#!/usr/bin/env python3
"""
计算seamless_interaction数据集的均值和标准差
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.seamless_interaction import SeamlessInteractionWindowDataset
from dataloaders.seamless_interaction_features import extract_pose_features, create_joint_mask

def compute_mean_std():
    """计算seamless_interaction数据集的均值和标准差"""
    
    # 数据集路径
    data_path = "./datasets/seamless_interaction"
    
    # 创建数据集
    dataset = SeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        sample_rate=16000,
        pose_fps=30,
        audio_fps=16000,
        window_size=2.0,  # 2秒窗口
        window_stride=0.5,  # 50%重叠
        load_video=False,
        load_audio=False,
        normalize=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        drop_last=False
    )
    
    # 初始化累加器
    sum_poses = None
    sum_sq_poses = None
    count = 0
    
    # 遍历数据集
    print("计算均值和标准差...")
    for batch in tqdm(dataloader):
        # 获取姿态数据
        poses = batch["poses"]  # [batch_size, seq_len, pose_dim]
        
        # 转换为numpy
        poses_np = poses.cpu().numpy()
        
        # 计算当前批次的和
        batch_sum = np.sum(poses_np, axis=(0, 1))  # [pose_dim]
        batch_sum_sq = np.sum(poses_np ** 2, axis=(0, 1))  # [pose_dim]
        batch_count = poses_np.shape[0] * poses_np.shape[1]  # 总帧数
        
        # 更新累加器
        if sum_poses is None:
            sum_poses = batch_sum
            sum_sq_poses = batch_sum_sq
        else:
            sum_poses += batch_sum
            sum_sq_poses += batch_sum_sq
        
        count += batch_count
    
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
    
    return mean_pose, std_pose

if __name__ == "__main__":
    compute_mean_std()