#!/usr/bin/env python3
"""
诊断训练脚本卡住的问题
"""

import os
import sys
import time
import pickle
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lazy_window_dataset_progress import create_progress_dataset

def test_dataset_loading():
    """测试数据集加载"""
    print("🔍 开始诊断数据集加载问题...")
    
    # 测试缓存文件加载
    cache_train = 'datasets/window_params/window_params_train_ws64_ws20_fixed.pkl'
    cache_val = 'datasets/window_params/window_params_val_ws64_ws20_fixed.pkl'
    
    print(f"📂 检查缓存文件是否存在...")
    print(f"  训练缓存: {os.path.exists(cache_train)}")
    print(f"  验证缓存: {os.path.exists(cache_val)}")
    
    if os.path.exists(cache_train):
        print(f"📊 加载训练缓存文件...")
        start_time = time.time()
        try:
            with open(cache_train, 'rb') as f:
                cache_data = pickle.load(f)
            load_time = time.time() - start_time
            print(f"✅ 训练缓存加载成功，耗时: {load_time:.2f}秒")
            print(f"  窗口数量: {cache_data.get('total_windows', 'N/A')}")
            print(f"  基础数据集大小: {len(cache_data.get('base_dataset_indices', []))}")
        except Exception as e:
            print(f"❌ 训练缓存加载失败: {e}")
            return
    
    # 测试数据集创建
    print(f"🏗️  创建数据集...")
    dataset_start = time.time()
    
    try:
        # 只创建一个小的数据集进行测试
        train_dataset = create_progress_dataset(
            data_path='datasets/seamless_interaction',
            split="train",
            window_size=64,
            window_stride=20,
            multi_length_training=[1.0],  # 只使用单一长度
            load_video=False,
            load_audio=False,
            max_samples=10,  # 只使用10个样本
            cache_path=cache_train,
            show_progress=True,
            progress_interval=1
        )
        
        dataset_time = time.time() - dataset_start
        print(f"✅ 数据集创建成功，耗时: {dataset_time:.2f}秒")
        print(f"📊 数据集大小: {len(train_dataset)}")
        
        # 测试数据加载
        print(f"🔄 测试数据加载...")
        load_start = time.time()
        
        for i in range(min(5, len(train_dataset))):
            try:
                sample = train_dataset[i]
                if i == 0:
                    print(f"  样本键: {list(sample.keys())}")
                    if 'pose' in sample:
                        print(f"  姿态形状: {sample['pose'].shape}")
            except Exception as e:
                print(f"❌ 加载样本 {i} 失败: {e}")
                return
        
        load_time = time.time() - load_start
        print(f"✅ 数据加载测试成功，耗时: {load_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 数据集诊断完成，一切正常！")

def test_distributed_setup():
    """测试分布式设置"""
    print("\n🔍 开始诊断分布式设置问题...")
    
    try:
        print(f"🔧 检查CUDA可用性...")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print(f"🔧 检查NCCL可用性...")
        try:
            if torch.distributed.is_nccl_available():
                print(f"  NCCL可用: 是")
            else:
                print(f"  NCCL可用: 否")
        except Exception as e:
            print(f"  NCCL检查失败: {e}")
        
    except Exception as e:
        print(f"❌ 分布式设置检查失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 分布式设置诊断完成！")

if __name__ == "__main__":
    test_dataset_loading()
    test_distributed_setup()