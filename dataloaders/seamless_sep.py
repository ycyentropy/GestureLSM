import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import librosa
from pathlib import Path
import lmdb
import pickle
from tqdm import tqdm

from .seamless_interaction import SeamlessInteractionDataset, SeamlessInteractionWindowDataset
from .data_tools import joints_list
from utils import rotation_conversions as rc


class SeamlessCustomDataset(Dataset):
    """
    为RVQ-VAE训练设计的Seamless_Interaction数据集加载器
    
    该类适配了SeamlessInteractionWindowDataset，使其与现有的RVQ-VAE训练流程兼容
    """
    
    def __init__(self, args, split="train", build_cache=True):
        """
        初始化数据集
        
        Args:
            args: 配置参数
            split: 数据集分割 ('train', 'val', 'test')
            build_cache: 是否构建缓存
        """
        self.args = args
        self.split = split
        self.build_cache = build_cache
        
        # 设置数据集路径
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise ValueError(f"数据集路径不存在: {data_path}")
        
        # 创建基础数据集
        self.base_dataset = SeamlessInteractionDataset(
            data_path=data_path,
            split=split,
            sample_rate=args.audio_sr,
            pose_fps=args.pose_fps,
            audio_fps=args.audio_fps,
            window_size=getattr(args, 'seamless_window_size', 2.0),
            window_stride=getattr(args, 'seamless_window_stride', 0.5),
            load_video=getattr(args, 'seamless_load_video', False),
            load_audio=getattr(args, 'seamless_load_audio', True),
            normalize=getattr(args, 'seamless_normalize', True)
        )
        
        # 创建窗口数据集
        self.window_dataset = SeamlessInteractionWindowDataset(
            data_path=data_path,
            split=split,
            sample_rate=args.audio_sr,
            pose_fps=args.pose_fps,
            audio_fps=args.audio_fps,
            window_size=getattr(args, 'seamless_window_size', 2.0),
            window_stride=getattr(args, 'seamless_window_stride', 0.5),
            load_video=getattr(args, 'seamless_load_video', False),
            load_audio=getattr(args, 'seamless_load_audio', True),
            normalize=getattr(args, 'seamless_normalize', True)
        )
        
        # 设置关节点列表
        self.ori_joint_list = joints_list.get(args.ori_joints, joints_list['beat_smplx_joints'])
        self.tar_joint_list = joints_list.get(args.tar_joints, joints_list['beat_smplx_full'])
        
        # 创建关节点掩码
        self.joint_mask = self._create_joint_mask()
        
        # 设置缓存路径
        self.cache_path = os.path.join(args.cache_path, split)
        os.makedirs(self.cache_path, exist_ok=True)
        
        # 构建缓存
        if build_cache:
            self.build_cache()
        
        # 加载均值和标准差
        self.mean_pose = None
        self.std_pose = None
        self._load_mean_std()
        
        # 设置窗口大小
        self.window_size = args.pose_length
        
        # 加载缓存索引
        self.cache_keys = self._load_cache_keys()
    
    def _create_joint_mask(self):
        """创建关节点掩码"""
        # 对于SMPL-H模型，我们有52个关节点（包括手部）
        # 这里创建一个简单的掩码，选择所有关节点
        mask = np.ones(52, dtype=bool)
        return mask
    
    def build_cache(self):
        """构建LMDB缓存"""
        print(f"构建{self.split}数据集缓存...")
        
        # 创建LMDB环境
        env = lmdb.open(self.cache_path, map_size=int(1e11))
        
        with env.begin(write=True) as txn:
            # 处理每个窗口样本
            for idx in tqdm(range(len(self.window_dataset)), desc=f"处理{self.split}数据"):
                sample = self.window_dataset[idx]
                
                # 处理姿态数据
                if 'pose' in sample:
                    pose_data = sample['pose'].numpy() if isinstance(sample['pose'], torch.Tensor) else sample['pose']
                    
                    # 确保姿态数据是正确的形状
                    if pose_data.shape[1] != 156:  # SMPL-H姿态参数应该是156维
                        continue
                    
                    # 应用关节点掩码
                    masked_pose = pose_data[:, self._map_smplh_to_mask()]
                    
                    # 转换为6D表示
                    pose_6d = self._axis_angle_to_6d(masked_pose)
                    
                    # 标准化
                    if self.mean_pose is not None and self.std_pose is not None:
                        pose_6d = (pose_6d - self.mean_pose) / (self.std_pose + 1e-8)
                    
                    # 存储到缓存
                    key = f"{idx:08d}".encode('ascii')
                    value = pickle.dumps(pose_6d.astype(np.float32))
                    txn.put(key, value)
        
        env.close()
        print(f"缓存构建完成: {self.cache_path}")
    
    def _map_smplh_to_mask(self):
        """将SMPL-H关节点映射到掩码"""
        # 这里简化处理，假设所有关节点都被选择
        # 实际应用中可能需要更复杂的映射
        return list(range(52))  # SMPL-H有52个关节点
    
    def _axis_angle_to_6d(self, pose_data):
        """将轴角表示转换为6D表示"""
        # 重塑姿态数据为 (T, N, 3)
        T, D = pose_data.shape
        N = D // 3
        pose_reshaped = pose_data.reshape(T, N, 3)
        
        # 转换为6D表示
        pose_6d = rc.axis_angle_to_6d(pose_reshaped)
        
        # 展平为 (T, N*6)
        pose_6d = pose_6d.reshape(T, N*6)
        
        return pose_6d
    
    def _load_mean_std(self):
        """加载均值和标准差"""
        mean_path = os.path.join("mean_std", "seamless_smplh_mean.npy")
        std_path = os.path.join("mean_std", "seamless_smplh_std.npy")
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.mean_pose = np.load(mean_path)
            self.std_pose = np.load(std_path)
            print(f"加载均值和标准差: {mean_path}, {std_path}")
        else:
            print(f"均值和标准差文件不存在: {mean_path}, {std_path}")
            print("将使用运行时计算的统计量")
    
    def _load_cache_keys(self):
        """加载缓存键"""
        env = lmdb.open(self.cache_path, readonly=True, lock=False)
        
        with env.begin() as txn:
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        
        env.close()
        return keys
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.cache_keys)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 从缓存加载
        env = lmdb.open(self.cache_path, readonly=True, lock=False)
        
        with env.begin() as txn:
            key = self.cache_keys[idx].encode('ascii')
            value = txn.get(key)
        
        env.close()
        
        # 反序列化
        pose_data = pickle.loads(value)
        
        # 转换为PyTorch张量
        pose_tensor = torch.from_numpy(pose_data)
        
        # 确保数据是正确的形状 (window_size, dim)
        if pose_tensor.shape[0] > self.window_size:
            # 如果数据太长，随机裁剪
            start = np.random.randint(0, pose_tensor.shape[0] - self.window_size + 1)
            pose_tensor = pose_tensor[start:start+self.window_size]
        elif pose_tensor.shape[0] < self.window_size:
            # 如果数据太短，填充零
            padding = torch.zeros(self.window_size - pose_tensor.shape[0], pose_tensor.shape[1])
            pose_tensor = torch.cat([pose_tensor, padding], dim=0)
        
        return pose_tensor
    
    def calculate_mean_velocity(self):
        """计算关节点平均速度"""
        print("计算关节点平均速度...")
        
        velocities = []
        
        for idx in tqdm(range(min(1000, len(self))), desc="计算速度"):
            pose_data = self[idx]
            
            # 计算速度
            velocity = pose_data[1:] - pose_data[:-1]
            velocities.append(velocity.abs().mean().item())
        
        mean_velocity = np.mean(velocities)
        print(f"平均速度: {mean_velocity}")
        
        return mean_velocity


def create_seamless_mean_std(args):
    """
    创建seamless_interaction数据集的均值和标准差文件
    
    Args:
        args: 配置参数
    """
    print("创建seamless_interaction数据集的均值和标准差文件...")
    
    # 创建数据集实例
    dataset = SeamlessCustomDataset(args, "train", build_cache=False)
    
    # 收集所有姿态数据
    all_poses = []
    
    for idx in tqdm(range(min(1000, len(dataset))), desc="收集姿态数据"):
        pose_data = dataset[idx]
        all_poses.append(pose_data.numpy())
    
    # 合并所有姿态数据
    all_poses = np.concatenate(all_poses, axis=0)
    
    # 计算均值和标准差
    mean_pose = np.mean(all_poses, axis=0)
    std_pose = np.std(all_poses, axis=0)
    
    # 保存均值和标准差
    os.makedirs("mean_std", exist_ok=True)
    np.save("mean_std/seamless_smplh_mean.npy", mean_pose)
    np.save("mean_std/seamless_smplh_std.npy", std_pose)
    
    print(f"均值和标准差已保存到: mean_std/seamless_smplh_mean.npy, mean_std/seamless_smplh_std.npy")
    
    return mean_pose, std_pose