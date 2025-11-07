#!/usr/bin/env python3
"""
测试带进度显示的数据加载器
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
import time
from lazy_window_dataset_progress import create_progress_dataset

def test_progress_dataloader():
    print("=== 测试带进度显示的数据加载器 ===")

    # 创建训练数据集（带进度显示）
    print("\n🏗️  创建训练数据集...")
    start_time = time.time()

    train_dataset = create_progress_dataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        split="train",
        window_size=64,
        window_stride=20,
        multi_length_training=[0.5, 0.75, 1.0, 1.25, 1.5],
        load_video=False,
        load_audio=False,
        max_samples=50,  # 使用50个样本进行测试
        cache_path="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl",
        show_progress=True,
        progress_interval=100  # 每100个窗口显示一次详细进度
    )

    dataset_time = time.time() - start_time
    print(f"✅ 训练数据集创建完成，耗时: {dataset_time:.2f}秒")
    print(f"📊 总窗口数: {len(train_dataset):,}")

    # 测试获取前几个样本
    print(f"\n🔍 测试获取前5个样本...")
    sample_start = time.time()

    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"样本 {i}: 姿态形状 {sample['pose'].shape}")

    sample_time = time.time() - sample_start
    print(f"✅ 样本获取完成，耗时: {sample_time:.2f}秒")

    # 创建验证数据集（带进度显示）
    print(f"\n🏗️  创建验证数据集...")
    val_start = time.time()

    val_dataset = create_progress_dataset(
        data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
        split="val",
        window_size=64,
        window_stride=20,
        multi_length_training=[0.5, 0.75, 1.0, 1.25, 1.5],
        load_video=False,
        load_audio=False,
        max_samples=20,  # 使用20个样本进行测试
        cache_path="datasets/window_params/window_params_val_ws64_ws20_fixed.pkl",
        show_progress=True,
        progress_interval=50
    )

    val_time = time.time() - val_start
    print(f"✅ 验证数据集创建完成，耗时: {val_time:.2f}秒")
    print(f"📊 验证集窗口数: {len(val_dataset):,}")

    print(f"\n🎉 带进度显示的数据加载器测试完成！")
    print(f"⏱️  总耗时: {dataset_time + val_time:.2f}秒")

if __name__ == "__main__":
    test_progress_dataloader()