#!/usr/bin/env python3
"""
重新生成正确的多长度训练缓存文件
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

from save_window_params import save_window_params

def regenerate_cache():
    """重新生成正确的多长度缓存"""
    print("=== 重新生成正确的多长度训练缓存 ===")

    # 设置参数
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
    window_size = 64
    window_stride = 20
    multi_length_training = [0.5, 0.75, 1.0, 1.25, 1.5]
    pose_fps = 30
    audio_fps = 16000

    print(f"参数设置:")
    print(f"  data_path: {data_path}")
    print(f"  window_size: {window_size}")
    print(f"  window_stride: {window_stride}")
    print(f"  multi_length_training: {multi_length_training}")
    print(f"  pose_fps: {pose_fps}")
    print(f"  audio_fps: {audio_fps}")

    # 重新生成训练集缓存
    print("\n=== 重新生成训练集缓存 ===")
    train_cache_path = save_window_params(
        data_path=data_path,
        split="train",
        window_size=window_size,
        window_stride=window_stride,
        multi_length_training=multi_length_training,
        pose_fps=pose_fps,
        audio_fps=audio_fps,
        max_samples=None,  # 使用所有样本
        save_path="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
    )

    # 重新生成验证集缓存
    print("\n=== 重新生成验证集缓存 ===")
    val_cache_path = save_window_params(
        data_path=data_path,
        split="val",
        window_size=window_size,
        window_stride=window_stride,
        multi_length_training=multi_length_training,
        pose_fps=pose_fps,
        audio_fps=audio_fps,
        max_samples=None,  # 使用所有样本
        save_path="/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20_fixed.pkl"
    )

    print("\n=== 缓存生成完成 ===")
    print(f"训练集缓存: {train_cache_path}")
    print(f"验证集缓存: {val_cache_path}")

    # 验证缓存文件内容
    import pickle

    print("\n=== 验证缓存文件内容 ===")
    with open(train_cache_path, 'rb') as f:
        train_data = pickle.load(f)

    print(f"训练集缓存验证:")
    print(f"  window_size: {train_data['window_size']}")
    print(f"  window_stride: {train_data['window_stride']}")
    print(f"  multi_length_training: {train_data['multi_length_training']}")
    print(f"  total_windows: {train_data['total_windows']}")

    # 检查前20个窗口参数
    print(f"前20个窗口参数:")
    unique_params = set()
    param_counts = {}
    for i in range(min(20, len(train_data['window_params']))):
        length, stride = train_data['window_params'][i]
        unique_params.add((length, stride))
        param_counts[(length, stride)] = param_counts.get((length, stride), 0) + 1
        print(f"  窗口 {i}: 长度={length}, 步长={stride}")

    print(f"\n参数分布统计:")
    for param, count in param_counts.items():
        print(f"  长度={param[0]}, 步长={param[1]}: {count} 个窗口")

if __name__ == "__main__":
    regenerate_cache()