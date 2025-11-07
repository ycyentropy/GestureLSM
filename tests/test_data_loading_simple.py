#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
from dataloaders.seamless_interaction import SeamlessInteractionDataset

def test_data_loading():
    print("开始测试数据加载...")
    
    # 测试SeamlessInteractionDataset
    print("\n=== 测试SeamlessInteractionDataset ===")
    base_dataset = SeamlessInteractionDataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        split="train",
        window_size=2.0,
        window_stride=0.5,
        load_audio=False,
        load_video=False,
        normalize=True
    )
    
    print(f"基础数据集大小: {len(base_dataset)}")
    
    # 测试获取第一个样本
    print("尝试获取第一个样本...")
    try:
        first_sample = base_dataset[0]
        print(f"成功获取第一个样本，键: {list(first_sample.keys())}")
        if 'pose' in first_sample:
            print(f"第一个样本的姿态形状: {first_sample['pose'].shape}")
    except Exception as e:
        print(f"获取第一个样本时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试CachedLazySeamlessInteractionWindowDataset
    print("\n=== 测试CachedLazySeamlessInteractionWindowDataset ===")
    window_dataset = CachedLazySeamlessInteractionWindowDataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        window_size=64,
        window_stride=20,
        split="train",
        multi_length_training=[0.8, 1.0, 1.2],
        audio_fps=16000,
        pose_fps=30,
        cache_path="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20.pkl"
    )
    
    print(f"窗口数据集大小: {len(window_dataset)}")
    
    # 测试获取第一个样本
    print("尝试获取第一个窗口样本...")
    try:
        first_window = window_dataset[0]
        print(f"成功获取第一个窗口样本，键: {list(first_window.keys())}")
        if 'pose' in first_window:
            print(f"第一个窗口样本的姿态形状: {first_window['pose'].shape}")
    except Exception as e:
        print(f"获取第一个窗口样本时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试创建批次
    print("\n=== 测试创建批次 ===")
    try:
        batch_size = 4
        batch_indices = list(range(min(batch_size, len(window_dataset))))
        batch = [window_dataset[i] for i in batch_indices]
        print(f"成功创建批次，大小: {len(batch)}")
        print(f"批次中第一个样本的键: {list(batch[0].keys())}")
        if 'pose' in batch[0]:
            print(f"批次中第一个样本的姿态形状: {batch[0]['pose'].shape}")
        
        # 测试collate_fn
        def custom_collate_fn(batch):
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
                poses[i, :length] = pose
                masks[i, :length] = True
            
            return {
                'pose': poses,
                'mask': masks,
                'id': [item['id'] for item in batch]
            }
        
        collated_batch = custom_collate_fn(batch)
        print(f"成功合并批次，键: {list(collated_batch.keys())}")
        if 'pose' in collated_batch:
            print(f"合并后的姿态形状: {collated_batch['pose'].shape}")
    except Exception as e:
        print(f"创建批次时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n数据加载测试完成！")

if __name__ == "__main__":
    test_data_loading()