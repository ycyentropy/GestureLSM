#!/usr/bin/env python3
"""
使用惰性加载计算seamless数据集的均值和标准差
这个脚本不会一次性加载所有数据，而是在需要时才加载每个窗口
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# 导入自定义数据集
from lazy_window_dataset import LazySeamlessInteractionWindowDataset

def compute_seamless_mean_std_lazy(
    data_path: str = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train",
    output_dir: str = "mean_std",
    batch_size: int = 32,
    max_samples: int = 1000,  # 限制样本数量以加快计算
    random_seed: int = 42
):
    """
    使用惰性加载计算seamless数据集的均值和标准差
    
    Args:
        data_path: 数据集路径
        output_dir: 输出目录
        batch_size: 批次大小
        max_samples: 最大样本数
        random_seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("创建惰性加载数据集...")
    start_time = time.time()
    
    # 创建惰性加载数据集
    dataset = LazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        pose_fps=30,
        window_size=2.0,
        window_stride=0.5,
        load_video=False,
        load_audio=False,
        max_samples=max_samples
    )
    
    end_time = time.time()
    print(f"数据集创建完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总窗口数: {len(dataset)}")
    
    # 自定义collate函数，处理不同长度的序列
    def custom_collate_fn(batch):
        result = {}
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['pose', 'keypoints', 'emotion_scores', 'expression']:
                # 对于序列数据，使用列表而不是堆叠
                result[key] = [item[key] for item in batch]
            else:
                try:
                    # 对于非序列数据，尝试堆叠
                    result[key] = torch.stack([item[key] for item in batch])
                except:
                    # 如果堆叠失败，使用列表
                    result[key] = [item[key] for item in batch]
        
        return result
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 使用0个worker以避免多进程问题
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    # 初始化统计变量
    pose_dim = None
    total_samples = 0
    sum_pose = None
    sum_pose_sq = None
    
    print("开始计算均值和标准差...")
    start_time = time.time()
    
    # 遍历数据集
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="处理批次")):
        # 获取姿态数据
        if 'pose' in batch:
            # 现在batch['pose']是一个列表，每个元素是一个张量
            poses_list = batch['pose']  # List of [seq_len, pose_dim]
            
            # 获取姿态维度
            if pose_dim is None:
                pose_dim = poses_list[0].shape[-1]
                sum_pose = torch.zeros(pose_dim, dtype=torch.float32)
                sum_pose_sq = torch.zeros(pose_dim, dtype=torch.float32)
            
            # 处理每个窗口的姿态数据
            for poses in poses_list:
                # 展平序列维度，将所有帧视为独立样本
                poses_flat = poses.view(-1, pose_dim)  # [seq_len, pose_dim]
                
                # 累加统计量
                batch_sum = torch.sum(poses_flat, dim=0)
                batch_sum_sq = torch.sum(poses_flat ** 2, dim=0)
                batch_count = poses_flat.shape[0]
                
                sum_pose += batch_sum
                sum_pose_sq += batch_sum_sq
                total_samples += batch_count
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"已处理 {batch_idx+1} 个批次，共 {total_samples} 个帧")
    
    end_time = time.time()
    print(f"数据处理完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总共处理了 {total_samples} 个帧")
    
    # 计算均值和标准差
    mean_pose = sum_pose / total_samples
    std_pose = torch.sqrt(sum_pose_sq / total_samples - mean_pose ** 2)
    
    # 转换为numpy数组
    mean_pose_np = mean_pose.numpy()
    std_pose_np = std_pose.numpy()
    
    # 保存结果
    mean_path = os.path.join(output_dir, "seamless_smplh_mean.npy")
    std_path = os.path.join(output_dir, "seamless_smplh_std.npy")
    
    np.save(mean_path, mean_pose_np)
    np.save(std_path, std_pose_np)
    
    print(f"均值已保存到: {mean_path}")
    print(f"标准差已保存到: {std_path}")
    
    # 打印统计信息
    print("\n姿态数据统计:")
    print(f"维度: {pose_dim}")
    print(f"均值范围: [{mean_pose_np.min():.6f}, {mean_pose_np.max():.6f}]")
    print(f"标准差范围: [{std_pose_np.min():.6f}, {std_pose_np.max():.6f}]")
    
    return mean_pose_np, std_pose_np


def compute_seamless_mean_std_sampled_lazy(
    data_path: str = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train",
    output_dir: str = "mean_std",
    num_samples: int = 1000,  # 采样数量
    random_seed: int = 42
):
    """
    使用惰性加载和采样计算seamless数据集的均值和标准差
    从每个批次中采样一个样本
    
    Args:
        data_path: 数据集路径
        output_dir: 输出目录
        num_samples: 采样数量
        random_seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("创建惰性加载数据集...")
    start_time = time.time()
    
    # 创建惰性加载数据集
    dataset = LazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        pose_fps=30,
        window_size=2.0,
        window_stride=0.5,
        load_video=False,
        load_audio=False
    )
    
    end_time = time.time()
    print(f"数据集创建完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总窗口数: {len(dataset)}")
    
    # 初始化统计变量
    pose_dim = None
    total_samples = 0
    sum_pose = None
    sum_pose_sq = None
    
    print("开始采样计算均值和标准差...")
    start_time = time.time()
    
    # 随机采样窗口
    sampled_indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(tqdm(sampled_indices, desc="采样窗口")):
        # 获取单个窗口
        window = dataset[idx]
        
        # 获取姿态数据
        if 'pose' in window:
            poses = window['pose']  # [seq_len, pose_dim]
            
            # 获取姿态维度
            if pose_dim is None:
                pose_dim = poses.shape[-1]
                sum_pose = torch.zeros(pose_dim, dtype=torch.float32)
                sum_pose_sq = torch.zeros(pose_dim, dtype=torch.float32)
            
            # 展平序列维度
            poses_flat = poses.view(-1, pose_dim)  # [seq_len, pose_dim]
            
            # 累加统计量
            batch_sum = torch.sum(poses_flat, dim=0)
            batch_sum_sq = torch.sum(poses_flat ** 2, dim=0)
            batch_count = poses_flat.shape[0]
            
            sum_pose += batch_sum
            sum_pose_sq += batch_sum_sq
            total_samples += batch_count
            
            # 打印进度
            if i % 100 == 0:
                print(f"已采样 {i+1} 个窗口，共 {total_samples} 个帧")
    
    end_time = time.time()
    print(f"数据处理完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"总共处理了 {total_samples} 个帧")
    
    # 计算均值和标准差
    mean_pose = sum_pose / total_samples
    std_pose = torch.sqrt(sum_pose_sq / total_samples - mean_pose ** 2)
    
    # 转换为numpy数组
    mean_pose_np = mean_pose.numpy()
    std_pose_np = std_pose.numpy()
    
    # 保存结果
    mean_path = os.path.join(output_dir, "seamless_smplh_mean_sampled.npy")
    std_path = os.path.join(output_dir, "seamless_smplh_std_sampled.npy")
    
    np.save(mean_path, mean_pose_np)
    np.save(std_path, std_pose_np)
    
    print(f"均值已保存到: {mean_path}")
    print(f"标准差已保存到: {std_path}")
    
    # 打印统计信息
    print("\n姿态数据统计:")
    print(f"维度: {pose_dim}")
    print(f"均值范围: [{mean_pose_np.min():.6f}, {mean_pose_np.max():.6f}]")
    print(f"标准差范围: [{std_pose_np.min():.6f}, {std_pose_np.max():.6f}]")
    
    return mean_pose_np, std_pose_np


if __name__ == "__main__":
    print("计算seamless数据集的均值和标准差（惰性加载版本）")
    
    # 方法1：使用惰性加载处理所有数据
    print("\n方法1: 使用惰性加载处理所有数据")
    mean1, std1 = compute_seamless_mean_std_lazy(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train",
        output_dir="mean_std",
        batch_size=32,
        max_samples=500  # 限制样本数量
    )
    
    # 方法2：使用惰性加载和采样
    print("\n方法2: 使用惰性加载和采样")
    mean2, std2 = compute_seamless_mean_std_sampled_lazy(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train",
        output_dir="mean_std",
        num_samples=1000  # 采样1000个窗口
    )
    
    print("\n完成！")