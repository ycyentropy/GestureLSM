#!/usr/bin/env python3
"""
测试seamless数据加载器的简单脚本
验证52关节点配置和多长度训练逻辑
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
from omegaconf import OmegaConf
from dataloaders.seamless_sep import CustomDataset
from utils import config

def test_seamless_loader():
    print("=" * 60)
    print("测试 Seamless 数据加载器")
    print("=" * 60)

    # 加载配置文件
    cfg = OmegaConf.load('./configs/seamless_rvqvae.yaml')
    print(f"✓ 配置文件加载成功")

    # 转换为args对象
    class Args:
        def __init__(self, cfg_dict):
            for key, value in cfg_dict.items():
                setattr(self, key, value)

    args = Args(dict(cfg))
    print(f"✓ Args对象创建成功")

    # 检查关键配置
    print(f"\n关键配置检查:")
    print(f"  数据集路径: {args.data_path}")
    print(f"  数据集类型: {args.dataset}")
    print(f"  姿态表示: {args.pose_rep}")
    print(f"  原始关节点: {args.ori_joints}")
    print(f"  目标关节点: {args.tar_joints}")
    print(f"  姿态维度: {args.pose_dims}")
    print(f"  基础长度: {args.pose_length}")
    print(f"  基础步长: {args.stride}")
    print(f"  多长度训练: {args.multi_length_training}")

    # 检查多长度训练配置
    print(f"\n多长度训练详情:")
    for i, ratio in enumerate(args.multi_length_training):
        length = int(args.pose_length * ratio)
        stride = int(args.stride * ratio)
        time_sec = length / args.pose_fps
        print(f"  比例 {ratio}: {length}帧 ≈ {time_sec:.1f}秒, 步长={stride}")

    # 检查数据路径是否存在
    if not os.path.exists(args.data_path):
        print(f"\n⚠️  警告: 数据路径不存在 - {args.data_path}")
        print("  请确保seamless_interaction数据集已正确放置")
        return False

    # 检查归一化文件
    norm_files = [
        './mean_std_seamless/seamless_2_312_mean.npy',
        './mean_std_seamless/seamless_2_312_std.npy',
        './mean_std_seamless/seamless_2_trans_mean.npy',
        './mean_std_seamless/seamless_2_trans_std.npy'
    ]

    print(f"\n归一化文件检查:")
    for file_path in norm_files:
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(f"  ✓ {os.path.basename(file_path)}: 形状 {data.shape}")
        else:
            print(f"  ✗ {os.path.basename(file_path)}: 文件不存在")
            return False

    # 尝试创建数据集（不实际构建缓存）
    print(f"\n尝试创建数据集对象...")
    try:
        # 设置一些必要的参数
        args.disable_filtering = True
        args.clean_first_seconds = 0
        args.clean_final_seconds = 0
        args.test_length = 128
        args.audio_sr = 16000
        args.audio_fps = 16000
        args.audio_rep = 'onset+amplitude'
        args.beat_align = False

        # 尝试扫描目录
        if os.path.exists(args.data_path):
            dataset = CustomDataset(args, "train", build_cache=False)
            print(f"✓ 数据集对象创建成功")
            print(f"  找到 {len(dataset.selected_files)} 个NPZ文件")

            if len(dataset.selected_files) > 0:
                print(f"  示例文件: {dataset.selected_files[0]}")
            else:
                print(f"  ⚠️  没有找到NPZ文件，请检查数据集结构")
        else:
            print(f"  ⚠️  数据路径不存在，跳过数据集创建")

    except Exception as e:
        print(f"  ✗ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n" + "=" * 60)
    print("测试完成 - 配置验证通过！")
    print("=" * 60)
    return True

def test_joint_configuration():
    print("\n关节点配置验证:")

    from dataloaders.data_tools import joints_list

    # 检查seamless关节点配置
    seamless_joints = joints_list.get('seamless_smplh_joints')
    seamless_full = joints_list.get('seamless_smplh_full')

    if seamless_joints and seamless_full:
        print(f"  ✓ seamless_smplh_joints: {len(seamless_joints)} 个关节点")
        print(f"  ✓ seamless_smplh_full: {len(seamless_full)} 个关节点")

        # 验证一致性
        if len(seamless_joints) == len(seamless_full):
            print(f"  ✓ 关节点配置一致")
        else:
            print(f"  ✗ 关节点配置不一致")
            return False

        # 显示前几个关节点
        print(f"  前10个关节点: {list(seamless_joints.keys())[:10]}")

        # 验证维度计算
        joint_count = len(seamless_full)
        pose_dims_6d = joint_count * 6
        print(f"  关节数量: {joint_count}")
        print(f"  6D表示维度: {pose_dims_6d}")

        if pose_dims_6d == 312:
            print(f"  ✓ 维度计算正确: {pose_dims_6d}")
        else:
            print(f"  ✗ 维度计算错误: 期望312，实际{pose_dims_6d}")
            return False
    else:
        print(f"  ✗ 无法找到seamless关节点配置")
        return False

    return True

if __name__ == "__main__":
    print("开始测试Seamless数据加载器配置...")

    # 测试关节点配置
    joint_ok = test_joint_configuration()

    # 测试数据加载器
    loader_ok = test_seamless_loader()

    if joint_ok and loader_ok:
        print(f"\n🎉 所有测试通过！Seamless数据加载器配置正确。")
        sys.exit(0)
    else:
        print(f"\n❌ 测试失败，请检查配置。")
        sys.exit(1)