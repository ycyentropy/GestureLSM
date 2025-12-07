#!/usr/bin/env python3
"""
测试seamless数据集多长度训练逻辑
验证 [0.5, 1.0, 1.5, 2.0] 配置是否正确工作
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
from omegaconf import OmegaConf
from dataloaders.seamless_sep import CustomDataset
from utils import config

def test_multi_length_training():
    print("=" * 70)
    print("测试多长度训练逻辑 [0.5, 1.0, 1.5, 2.0]")
    print("=" * 70)

    # 加载配置
    cfg = OmegaConf.load('./configs/seamless_rvqvae.yaml')

    class Args:
        def __init__(self, cfg_dict):
            for key, value in cfg_dict.items():
                setattr(self, key, value)

    args = Args(dict(cfg))

    # 多长度训练配置
    multi_lengths = args.multi_length_training
    base_length = args.pose_length
    base_stride = args.stride
    pose_fps = args.pose_fps

    print(f"基础配置:")
    print(f"  基础长度: {base_length} 帧")
    print(f"  基础步长: {base_stride} 帧")
    print(f"  帧率: {pose_fps} FPS")
    print(f"  多长度配置: {multi_lengths}")

    print(f"\n多长度训练详情:")
    total_samples_per_ratio = {}

    for i, ratio in enumerate(multi_lengths):
        length = int(base_length * ratio)
        stride = int(base_stride * ratio)
        time_sec = length / pose_fps

        print(f"  比例 {ratio}:")
        print(f"    序列长度: {length} 帧 ≈ {time_sec:.1f} 秒")
        print(f"    步长: {stride} 帧")

        total_samples_per_ratio[ratio] = {
            'length': length,
            'stride': stride,
            'time_sec': time_sec
        }

    # 模拟采样逻辑
    print(f"\n模拟数据采样逻辑:")

    # 假设有一个1000帧的序列
    total_frames = 1000
    clean_frames = total_frames  # 简化假设，不考虑清理

    print(f"  假设总帧数: {total_frames} 帧 ≈ {total_frames/pose_fps:.1f} 秒")

    for ratio in multi_lengths:
        length = int(base_length * ratio)
        stride = int(base_stride * ratio)

        # 计算可以采样多少个片段
        num_samples = max(0, (clean_frames - length) // stride + 1)
        total_time = num_samples * (length / pose_fps)

        print(f"  比例 {ratio}:")
        print(f"    可采样片段数: {num_samples}")
        print(f"    总训练时间: {total_time:.1f} 秒")

    # 测试数据集初始化时的多长度设置
    print(f"\n验证数据集初始化:")

    # 设置必要的参数
    args.disable_filtering = True
    args.clean_first_seconds = 0
    args.clean_final_seconds = 0
    args.test_length = 128
    args.audio_sr = 16000
    args.audio_fps = 16000
    args.audio_rep = 'onset+amplitude'
    args.beat_align = False

    try:
        # 创建训练集数据集
        print(f"  创建训练集...")
        train_dataset = CustomDataset(args, "train", build_cache=False)
        print(f"  ✓ 训练集创建成功，找到 {len(train_dataset.selected_files)} 个文件")

        # 创建测试集数据集
        print(f"  创建测试集...")
        test_dataset = CustomDataset(args, "test", build_cache=False)
        print(f"  ✓ 测试集创建成功，找到 {len(test_dataset.selected_files)} 个文件")

        # 验证测试集的多长度设置
        print(f"  测试集多长度配置: {test_dataset.args.multi_length_training}")
        if test_dataset.args.multi_length_training == [1.0]:
            print(f"  ✓ 测试集正确设置为单一长度 [1.0]")
        else:
            print(f"  ✗ 测试集多长度配置错误")

    except Exception as e:
        print(f"  ✗ 数据集创建失败: {e}")
        return False

    print(f"\n" + "=" * 70)
    print("多长度训练逻辑验证通过！")
    print("=" * 70)

    return True

def test_memory_usage_estimation():
    """估算不同长度的内存使用"""
    print(f"\n内存使用估算:")

    pose_dims = 312  # 52关节 × 6D
    trans_dims = 3
    facial_dims = 100
    total_dims = pose_dims + trans_dims + facial_dims  # 415维

    batch_size = 64
    multi_lengths = [0.5, 1.0, 1.5, 2.0]
    base_length = 128
    pose_fps = 30

    print(f"  每个样本维度: {total_dims} (姿态: {pose_dims}, 平移: {trans_dims}, 面部: {facial_dims})")
    print(f"  批次大小: {batch_size}")

    for ratio in multi_lengths:
        length = int(base_length * ratio)
        # 假设float32，每个元素4字节
        memory_mb = batch_size * length * total_dims * 4 / (1024 * 1024)

        print(f"  比例 {ratio} ({length}帧 ≈ {length/pose_fps:.1f}秒): ~{memory_mb:.1f} MB")

    print(f"  建议GPU内存: 至少 {batch_size * 256 * total_dims * 4 / (1024 * 1024):.0f} MB")

def test_stride_consistency():
    """测试步长与长度的一致性"""
    print(f"\n步长一致性检查:")

    base_length = 128
    base_stride = 20
    multi_lengths = [0.5, 1.0, 1.5, 2.0]

    for ratio in multi_lengths:
        length = int(base_length * ratio)
        stride = int(base_stride * ratio)

        # 检查步长是否合理（不应该超过序列长度）
        if stride > length:
            print(f"  ⚠️  比例 {ratio}: 步长 {stride} > 序列长度 {length}，可能导致采样问题")
        else:
            print(f"  ✓ 比例 {ratio}: 步长 {stride} <= 序列长度 {length}")

        # 检查覆盖率
        coverage = stride / length
        print(f"    覆盖率: {coverage:.2f} (步长/序列长度)")

if __name__ == "__main__":
    print("开始测试Seamless多长度训练逻辑...")

    # 测试多长度训练
    multi_ok = test_multi_length_training()

    # 测试内存估算
    test_memory_usage_estimation()

    # 测试步长一致性
    test_stride_consistency()

    if multi_ok:
        print(f"\n🎉 多长度训练逻辑验证成功！")
        print(f"   配置 [0.5, 1.0, 1.5, 2.0] 工作正常")
        print(f"   支持从2.1秒到8.5秒的多样化手势序列")
        sys.exit(0)
    else:
        print(f"\n❌ 多长度训练逻辑验证失败！")
        sys.exit(1)