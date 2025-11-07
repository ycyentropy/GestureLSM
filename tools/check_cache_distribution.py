#!/usr/bin/env python3
"""
检查缓存中窗口长度的分布
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import pickle
from collections import Counter

def check_cache_distribution():
    """检查缓存中窗口长度的分布"""
    print("=== 检查缓存中窗口长度分布 ===")

    cache_path = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    window_params = cache_data['window_params']
    print(f"总窗口数: {len(window_params)}")

    # 统计不同长度的窗口数量
    length_counts = Counter()
    param_counts = Counter()

    # 检查前1000个窗口
    sample_size = min(1000, len(window_params))
    print(f"\n检查前{sample_size}个窗口的长度分布:")

    for i in range(sample_size):
        length, stride = window_params[i]
        length_counts[length] += 1
        param_counts[(length, stride)] += 1

    print("长度分布:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        percentage = (count / sample_size) * 100
        print(f"  长度 {length}: {count} 个窗口 ({percentage:.1f}%)")

    print("\n参数分布 (长度, 步长):")
    for param in sorted(param_counts.keys()):
        count = param_counts[param]
        percentage = (count / sample_size) * 100
        print(f"  {param}: {count} 个窗口 ({percentage:.1f}%)")

    # 检查更多样本，看看是否有其他长度
    print(f"\n检查所有窗口的长度分布...")
    all_length_counts = Counter()

    for length, stride in window_params:
        all_length_counts[length] += 1

    print("完整数据集的长度分布:")
    total_windows = len(window_params)
    for length in sorted(all_length_counts.keys()):
        count = all_length_counts[length]
        percentage = (count / total_windows) * 100
        print(f"  长度 {length}: {count:,} 个窗口 ({percentage:.1f}%)")

    # 检查预期的长度
    expected_lengths = [32, 48, 64, 80, 96]
    print(f"\n预期长度: {expected_lengths}")
    actual_lengths = sorted(all_length_counts.keys())
    print(f"实际长度: {actual_lengths}")

    missing_lengths = set(expected_lengths) - set(actual_lengths)
    extra_lengths = set(actual_lengths) - set(expected_lengths)

    if missing_lengths:
        print(f"缺少的长度: {sorted(missing_lengths)}")
    if extra_lengths:
        print(f"额外的长度: {sorted(extra_lengths)}")

    if not missing_lengths and not extra_lengths:
        print("✓ 所有预期长度都存在，没有额外长度")

if __name__ == "__main__":
    check_cache_distribution()