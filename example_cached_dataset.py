#!/usr/bin/env python3
"""
示例脚本：如何使用缓存的窗口参数
"""

import os
import torch
from torch.utils.data import DataLoader
from save_window_params import save_window_params, CachedLazySeamlessInteractionWindowDataset


def example_save_and_load():
    """
    示例：保存和加载窗口参数
    """
    # 1. 首次运行时，计算并保存窗口参数
    print("=== 步骤1: 计算并保存窗口参数 ===")
    save_path = save_window_params(
        data_path='data/seamless_motions',
        split='train',
        window_size=60,
        window_stride=30,
        multi_length_training=[0.5, 0.75, 1.0, 1.25, 1.5],
        max_samples=100  # 限制为100个样本以加快速度
    )
    print(f"窗口参数已保存到: {save_path}\n")
    
    # 2. 后续运行时，直接加载缓存的窗口参数
    print("=== 步骤2: 加载缓存的窗口参数创建数据集 ===")
    cached_dataset = CachedLazySeamlessInteractionWindowDataset(
        data_path='data/seamless_motions',
        split='train',
        window_size=60,
        window_stride=30,
        multi_length_training=[0.5, 0.75, 1.0, 1.25, 1.5],
        max_samples=100,
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        cache_path=save_path  # 指定缓存文件路径
    )
    
    print(f"缓存数据集大小: {len(cached_dataset)} 个窗口\n")
    
    # 3. 测试数据加载
    print("=== 步骤3: 测试数据加载 ===")
    sample = cached_dataset[0]
    print(f"第一个窗口的键: {list(sample.keys())}")
    if 'pose' in sample:
        print(f"姿态数据形状: {sample['pose'].shape}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        cached_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x  # 简单的collate函数
    )
    
    batch = next(iter(dataloader))
    print(f"批次大小: {len(batch)}")
    print("成功加载批次数据!\n")
    
    # 4. 保存当前数据集的窗口参数
    print("=== 步骤4: 保存当前数据集的窗口参数 ===")
    new_save_path = cached_dataset.save_window_params()
    print(f"当前数据集的窗口参数已保存到: {new_save_path}")


def compare_loading_time():
    """
    比较使用缓存和不使用缓存的加载时间
    """
    import time
    
    print("=== 比较加载时间 ===")
    
    # 参数设置
    data_path = 'data/seamless_motions'
    split = 'train'
    window_size = 60
    window_stride = 30
    multi_length_training = [0.5, 0.75, 1.0, 1.25, 1.5]
    max_samples = 100
    
    # 1. 不使用缓存的加载时间
    print("1. 不使用缓存的加载时间:")
    from lazy_window_dataset import LazySeamlessInteractionWindowDataset
    
    start_time = time.time()
    dataset_no_cache = LazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        multi_length_training=multi_length_training,
        max_samples=max_samples,
        load_video=False,
        load_audio=False
    )
    no_cache_time = time.time() - start_time
    print(f"   不使用缓存加载时间: {no_cache_time:.2f} 秒")
    print(f"   数据集大小: {len(dataset_no_cache)} 个窗口\n")
    
    # 2. 保存窗口参数
    print("2. 保存窗口参数:")
    save_path = save_window_params(
        data_path=data_path,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        multi_length_training=multi_length_training,
        max_samples=max_samples
    )
    print(f"   窗口参数已保存到: {save_path}\n")
    
    # 3. 使用缓存的加载时间
    print("3. 使用缓存的加载时间:")
    start_time = time.time()
    dataset_with_cache = CachedLazySeamlessInteractionWindowDataset(
        data_path=data_path,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        multi_length_training=multi_length_training,
        max_samples=max_samples,
        load_video=False,
        load_audio=False,
        cache_path=save_path
    )
    cache_time = time.time() - start_time
    print(f"   使用缓存加载时间: {cache_time:.2f} 秒")
    print(f"   数据集大小: {len(dataset_with_cache)} 个窗口\n")
    
    # 4. 比较结果
    print("4. 比较结果:")
    if no_cache_time > 0:
        speedup = no_cache_time / cache_time
        print(f"   加速比: {speedup:.2f}x")
        print(f"   时间节省: {((no_cache_time - cache_time) / no_cache_time * 100):.1f}%")


def integrate_with_training_script():
    """
    示例：如何将缓存集成到训练脚本中
    """
    print("=== 集成到训练脚本的示例 ===")
    
    # 1. 首先检查是否存在缓存文件
    data_path = 'data/seamless_motions'
    split = 'train'
    window_size = 60
    window_stride = 30
    multi_length_training = [0.5, 0.75, 1.0, 1.25, 1.5]
    max_samples = 100
    
    # 尝试自动查找缓存文件
    cache_dir = os.path.join(os.path.dirname(data_path), 'window_params')
    filename = f"window_params_{split}_ws{window_size}_ws{window_stride}"
    if max_samples is not None:
        filename += f"_max{max_samples}"
    filename += ".pkl"
    cache_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(cache_path):
        print(f"找到缓存文件: {cache_path}")
        print("使用CachedLazySeamlessInteractionWindowDataset加载数据集...")
        
        # 使用缓存加载数据集
        train_dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path=data_path,
            split=split,
            window_size=window_size,
            window_stride=window_stride,
            multi_length_training=multi_length_training,
            max_samples=max_samples,
            load_video=False,
            load_audio=False,
            cache_path=cache_path  # 使用缓存文件
        )
    else:
        print(f"未找到缓存文件: {cache_path}")
        print("使用LazySeamlessInteractionWindowDataset加载数据集...")
        
        # 使用常规方式加载数据集
        from lazy_window_dataset import LazySeamlessInteractionWindowDataset
        train_dataset = LazySeamlessInteractionWindowDataset(
            data_path=data_path,
            split=split,
            window_size=window_size,
            window_stride=window_stride,
            multi_length_training=multi_length_training,
            max_samples=max_samples,
            load_video=False,
            load_audio=False
        )
        
        # 保存窗口参数以供下次使用
        print("保存窗口参数以供下次使用...")
        save_path = save_window_params(
            data_path=data_path,
            split=split,
            window_size=window_size,
            window_stride=window_stride,
            multi_length_training=multi_length_training,
            max_samples=max_samples
        )
        print(f"窗口参数已保存到: {save_path}")
    
    print(f"数据集加载完成，大小: {len(train_dataset)} 个窗口")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x  # 简单的collate函数
    )
    
    # 测试加载一个批次
    batch = next(iter(train_loader))
    print(f"成功加载批次，大小: {len(batch)}")
    print("训练数据准备就绪!")


if __name__ == "__main__":
    # 运行示例
    print("运行示例1: 保存和加载窗口参数")
    example_save_and_load()
    print("\n" + "="*50 + "\n")
    
    print("运行示例2: 比较加载时间")
    compare_loading_time()
    print("\n" + "="*50 + "\n")
    
    print("运行示例3: 集成到训练脚本")
    integrate_with_training_script()