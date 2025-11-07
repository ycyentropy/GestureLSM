#!/usr/bin/env python3
"""
重新生成正确的缓存文件
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import pickle
from lazy_window_dataset import LazySeamlessInteractionWindowDataset

def regenerate_cache():
    """重新生成正确的缓存文件"""
    print("=== 重新生成缓存文件 ===")

    # 设置参数
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"

    # 创建正确的缓存
    window_size = 64
    window_stride = 20
    pose_fps = 30

    print(f"参数设置:")
    print(f"  data_path: {data_path}")
    print(f"  window_size: {window_size}")
    print(f"  window_stride: {window_stride}")
    print(f"  pose_fps: {pose_fps}")

    # 创建训练集缓存
    print("\n=== 生成训练集缓存 ===")
    train_dataset = LazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        window_size=window_size,
        window_stride=window_stride,
        pose_fps=pose_fps,
        max_samples=100,  # 限制样本数以便快速测试
        load_audio=False,
        load_video=False
    )

    # 保存训练集缓存
    train_cache_data = {
        'window_counts': train_dataset.window_counts,
        'cumulative_windows': train_dataset.cumulative_windows,
        'window_params': train_dataset.window_params,
        'total_windows': train_dataset.total_windows,
        'base_dataset_indices': train_dataset.base_dataset_indices,
        'window_size': window_size,
        'window_stride': window_stride,
        'multi_length_training': train_dataset.multi_length_training,
        'pose_fps': pose_fps,
        'audio_fps': 16000,
        'split': "train",
        'data_path': data_path
    }

    train_cache_path = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20_correct.pkl"
    with open(train_cache_path, 'wb') as f:
        pickle.dump(train_cache_data, f)

    print(f"训练集缓存已保存到: {train_cache_path}")
    print(f"总窗口数: {train_dataset.total_windows}")

    # 检查前几个窗口参数
    print(f"前10个窗口参数:")
    for i in range(min(10, len(train_dataset.window_params))):
        print(f"  窗口 {i}: 长度={train_dataset.window_params[i][0]}, 步长={train_dataset.window_params[i][1]}")

    print("\n=== 缓存生成完成 ===")

if __name__ == "__main__":
    regenerate_cache()