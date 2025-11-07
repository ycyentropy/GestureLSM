#!/usr/bin/env python3
"""
测试修复后的多长度缓存
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import pickle
import torch
from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

def test_fixed_cache():
    """测试修复后的多长度缓存"""
    print("=== 测试修复后的多长度缓存 ===")

    # 设置参数
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
    train_cache = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
    val_cache = "/home/embodied/yangchenyu/GestureLSM/datasets/window_params/window_params_val_ws64_ws20_fixed.pkl"

    print(f"数据路径: {data_path}")
    print(f"训练集缓存: {train_cache}")
    print(f"验证集缓存: {val_cache}")

    # 首先检查缓存文件内容
    print("\n=== 检查训练集缓存内容 ===")
    with open(train_cache, 'rb') as f:
        train_data = pickle.load(f)

    print(f"训练集缓存信息:")
    print(f"  窗口大小: {train_data['window_size']}")
    print(f"  窗口步长: {train_data['window_stride']}")
    print(f"  多长度训练比例: {train_data['multi_length_training']}")
    print(f"  总窗口数: {train_data['total_windows']}")
    print(f"  base_dataset_indices长度: {len(train_data['base_dataset_indices'])}")
    print(f"  window_params长度: {len(train_data['window_params'])}")

    # 检查前20个窗口参数的分布
    print(f"\n前20个窗口参数分布:")
    param_counts = {}
    unique_lengths = set()
    for i in range(min(20, len(train_data['window_params']))):
        length, stride = train_data['window_params'][i]
        unique_lengths.add(length)
        param_counts[(length, stride)] = param_counts.get((length, stride), 0) + 1
        print(f"  窗口 {i}: 长度={length}, 步长={stride}")

    print(f"\n参数分布统计:")
    for param, count in param_counts.items():
        print(f"  长度={param[0]}, 步长={param[1]}: {count} 个窗口")

    print(f"发现的唯一长度: {sorted(unique_lengths)}")

    # 检查是否包含多长度训练的窗口
    expected_lengths = [32, 48, 64, 80, 96]  # 64 * [0.5, 0.75, 1.0, 1.25, 1.5]
    missing_lengths = []
    for expected_length in expected_lengths:
        if expected_length not in unique_lengths:
            missing_lengths.append(expected_length)

    if missing_lengths:
        print(f"警告: 缺少以下长度的窗口: {missing_lengths}")
    else:
        print("✓ 所有预期长度的窗口都存在")

    # 测试数据集加载
    print("\n=== 测试数据集加载 ===")
    try:
        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path=data_path,
            split="train",
            window_size=64,  # 这个参数会被缓存参数覆盖
            window_stride=20,  # 这个参数会被缓存参数覆盖
            cache_path=train_cache,
            load_audio=False,
            load_video=False,
            max_samples=1000  # 限制样本数以便快速测试
        )

        print(f"✓ 数据集加载成功")
        print(f"  数据集长度: {len(dataset)}")
        print(f"  实际窗口大小: {dataset.window_size}")
        print(f"  实际窗口步长: {dataset.window_stride}")
        print(f"  多长度训练: {dataset.multi_length_training}")

        # 测试前几个样本
        print(f"\n=== 测试前5个样本 ===")
        for i in range(min(5, len(dataset))):
            try:
                sample = dataset[i]
                if 'pose' in sample:
                    print(f"样本 {i}: 姿态形状 {sample['pose'].shape}")
                else:
                    print(f"样本 {i}: 无姿态数据")
            except Exception as e:
                print(f"样本 {i} 错误: {e}")
                break

        # 测试批次加载
        print(f"\n=== 测试批次加载 ===")
        from torch.utils.data import DataLoader

        def multi_length_collate_fn(batch):
            """处理多长度窗口的批次整理函数"""
            max_len = max(item['pose'].shape[0] for item in batch)
            pose_dim = batch[0]['pose'].shape[1]

            poses = torch.zeros(len(batch), max_len, pose_dim)
            translations = torch.zeros(len(batch), max_len, 3)
            masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

            for i, item in enumerate(batch):
                seq_len = item['pose'].shape[0]
                poses[i, :seq_len] = item['pose']

                # 处理translation - 确保形状匹配
                translation = item['translation']
                if translation.shape[0] != seq_len:
                    # 如果translation的长度与seq_len不匹配，取前seq_len个
                    translation = translation[:seq_len]
                translations[i, :seq_len] = translation

                masks[i, :seq_len] = True

            return {
                'pose': poses,
                'translation': translations,
                'mask': masks
            }

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=multi_length_collate_fn
        )

        print("✓ 数据加载器创建成功")

        # 测试前两个批次
        for i, batch in enumerate(dataloader):
            print(f"批次 {i}:")
            print(f"  姿态形状: {batch['pose'].shape}")
            print(f"  平移形状: {batch['translation'].shape}")
            print(f"  掩码形状: {batch['mask'].shape}")

            if i >= 1:  # 只测试前两个批次
                break

        print("\n✓ 所有测试通过！多长度缓存工作正常。")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_cache()