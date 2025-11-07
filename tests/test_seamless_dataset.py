#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.seamless_interaction import SeamlessInteractionWindowDataset

def test_seamless_dataset():
    """测试seamless_interaction数据集的加载"""
    print("开始测试seamless_interaction数据集加载...")
    
    try:
        # 创建数据集实例
        dataset = SeamlessInteractionWindowDataset(
            data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
            split="train",
            window_size=2.0,
            window_stride=0.5,
            sample_rate=16000,
            pose_fps=30,
            audio_fps=16000,
            load_video=False,
            load_audio=False
        )
        
        print(f"数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本键: {sample.keys()}")
            
            if 'pose' in sample:
                pose = sample['pose']
                print(f"姿态数据形状: {pose.shape}")
                print(f"姿态数据类型: {type(pose)}")
                
                # 检查姿态数据是否为numpy数组
                if isinstance(pose, np.ndarray):
                    print(f"姿态数据范围: [{pose.min():.4f}, {pose.max():.4f}]")
                    print(f"姿态数据均值: {pose.mean():.4f}")
                    print(f"姿态数据标准差: {pose.std():.4f}")
            
            if 'id' in sample:
                print(f"样本ID: {sample['id']}")
        
        # 测试数据加载器
        def collate_fn(batch):
            """自定义批处理函数，处理不同长度的序列"""
            # 找到批次中最长的序列
            max_len = max([item['pose'].shape[0] for item in batch])
            
            # 创建填充后的批次数据
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]
            
            poses = torch.zeros(batch_size, max_len, pose_dim)
            masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
            
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = torch.from_numpy(pose)
                masks[i, :length] = True
            
            return {
                'pose': poses,
                'mask': masks,
                'id': [item['id'] for item in batch]
            }
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 设置为0以避免多进程问题
            collate_fn=collate_fn,
            drop_last=True
        )
        
        print("数据加载器创建成功")
        
        # 测试一个批次
        for batch in dataloader:
            print(f"批次姿态数据形状: {batch['pose'].shape}")
            print(f"批次掩码形状: {batch['mask'].shape}")
            print(f"批次ID: {batch['id']}")
            break  # 只测试一个批次
        
        print("数据集测试成功!")
        return True
        
    except Exception as e:
        print(f"数据集测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_seamless_dataset()