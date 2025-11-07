import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloaders.seamless_interaction import SeamlessInteractionWindowDataset
import random

def compute_seamless_mean_std_sampled():
    """计算seamless数据集的均值和标准差，从每个batch中采样一条数据"""
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 数据集路径
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train"
    
    # 创建数据集
    print("创建数据集...")
    dataset = SeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        pose_fps=30,
        window_size=2.0,
        window_stride=0.5,
        load_video=False,
        load_audio=False,
        normalize=False  # 不使用标准化，以便计算真实的均值和标准差
    )
    
    # 创建数据加载器
    batch_size = 32  # 设置批次大小
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    # 初始化变量
    total_samples = 0
    pose_sum = None
    pose_sq_sum = None
    
    # 遍历数据加载器
    print("开始计算均值和标准差...")
    for batch_idx, batch_data in enumerate(dataloader):
        # 从当前批次中随机选择一个样本
        sample_idx = random.randint(0, len(batch_data['poses']) - 1)
        pose_data = batch_data['poses'][sample_idx]  # [seq_len, pose_dim]
        
        # 获取姿态数据的维度
        seq_len, pose_dim = pose_data.shape
        
        # 展平数据以便计算
        flat_pose = pose_data.reshape(-1, pose_dim)  # [seq_len, pose_dim]
        
        # 更新总和
        if pose_sum is None:
            pose_sum = np.zeros(pose_dim)
            pose_sq_sum = np.zeros(pose_dim)
        
        pose_sum += np.sum(flat_pose.numpy(), axis=0)
        pose_sq_sum += np.sum(np.square(flat_pose.numpy()), axis=0)
        
        total_samples += seq_len
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f"已处理 {batch_idx}/{len(dataloader)} 个批次")
    
    # 计算均值和标准差
    mean_pose = pose_sum / total_samples
    std_pose = np.sqrt(pose_sq_sum / total_samples - np.square(mean_pose))
    
    # 确保标准差不为零
    std_pose = np.maximum(std_pose, 1e-8)
    
    print(f"总样本数: {total_samples}")
    print(f"均值形状: {mean_pose.shape}")
    print(f"标准差形状: {std_pose.shape}")
    
    # 创建保存目录
    save_dir = "./mean_std"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存均值和标准差
    mean_path = os.path.join(save_dir, "seamless_smplh_mean_sampled.npy")
    std_path = os.path.join(save_dir, "seamless_smplh_std_sampled.npy")
    
    np.save(mean_path, mean_pose)
    np.save(std_path, std_pose)
    
    print(f"均值已保存到: {mean_path}")
    print(f"标准差已保存到: {std_path}")
    
    # 打印一些统计信息
    print("\n均值统计:")
    print(f"最小值: {np.min(mean_pose):.6f}, 最大值: {np.max(mean_pose):.6f}")
    print(f"均值: {np.mean(mean_pose):.6f}, 标准差: {np.std(mean_pose):.6f}")
    
    print("\n标准差统计:")
    print(f"最小值: {np.min(std_pose):.6f}, 最大值: {np.max(std_pose):.6f}")
    print(f"均值: {np.mean(std_pose):.6f}, 标准差: {np.std(std_pose):.6f}")
    
    return mean_pose, std_pose

if __name__ == "__main__":
    compute_seamless_mean_std_sampled()