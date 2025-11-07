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
        print("步骤1: 创建数据集实例...")
        # 创建数据集实例
        dataset = SeamlessInteractionWindowDataset(
            data_path="datasets/seamless_interaction",
            split="train",
            window_size=2.0,
            window_stride=0.5,
            sample_rate=16000,
            pose_fps=30,
            audio_fps=16000,
            load_video=False,
            load_audio=False
        )
        
        print(f"步骤2: 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 获取一个样本
        if len(dataset) > 0:
            print("步骤3: 获取第一个样本...")
            sample = dataset[0]
            print(f"步骤4: 样本键: {sample.keys()}")
            
            if 'pose' in sample:
                pose = sample['pose']
                print(f"步骤5: 姿态数据形状: {pose.shape}")
                print(f"步骤6: 姿态数据类型: {type(pose)}")
                
                # 检查姿态数据是否为numpy数组
                if isinstance(pose, np.ndarray):
                    print(f"步骤7: 姿态数据范围: [{pose.min():.4f}, {pose.max():.4f}]")
                    print(f"步骤8: 姿态数据均值: {pose.mean():.4f}")
                    print(f"步骤9: 姿态数据标准差: {pose.std():.4f}")
            
            if 'id' in sample:
                print(f"步骤10: 样本ID: {sample['id']}")
        
        print("步骤11: 数据集测试成功!")
        return True
        
    except Exception as e:
        print(f"数据集测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_seamless_dataset()