#!/usr/bin/env python3
"""
Seamless_Interaction数据集解析器使用示例

这个脚本展示了如何使用Seamless_Interaction数据集解析器加载和处理数据。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.seamless_interaction import SeamlessInteractionDataset, SeamlessInteractionWindowDataset
from dataloaders.seamless_interaction_features import (
    extract_text_features, extract_vad_features, extract_pose_features,
    extract_keypoint_features, extract_emotion_features, extract_expression_features,
    align_audio_to_motion, create_joint_mask, extract_audio_features
)


def main():
    # 设置数据集路径
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction"
    
    # 创建数据集实例
    print("创建数据集实例...")
    dataset = SeamlessInteractionDataset(
        data_path=data_path,
        split="train",
        sample_rate=16000,
        pose_fps=30,
        audio_fps=16000,
        window_size=2.0,
        window_stride=0.5,
        load_video=False,
        load_audio=True,
        normalize=True
    )
    
    # 创建时间窗口数据集实例
    window_dataset = SeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        sample_rate=16000,
        pose_fps=30,
        audio_fps=16000,
        window_size=2.0,
        window_stride=0.5,
        load_video=False,
        load_audio=True,
        normalize=True
    )
    
    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"窗口数据集大小: {len(window_dataset)} 个窗口")
    
    # 获取第一个样本的信息
    print("\n样本信息:")
    sample_info = dataset.get_sample_info(0)
    for key, value in sample_info.items():
        print(f"  {key}: {value}")
    
    # 加载第一个样本
    print("\n加载第一个样本...")
    sample = dataset[0]
    
    # 打印样本的键和数据形状
    print("样本数据:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: 列表，长度 {len(value)}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 提取文本特征
    print("\n提取文本特征...")
    if 'transcript' in sample:
        text_features = extract_text_features(
            sample['transcript'], 
            max_words=50
        )
        print("文本特征:")
        for key, value in text_features.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    
    # 提取VAD特征
    print("\n提取VAD特征...")
    if 'vad' in sample:
        vad_features = extract_vad_features(
            sample['vad'], 
            sample_rate=30.0,
            max_frames=120  # 4秒 * 30帧/秒
        )
        print(f"VAD特征: {vad_features.shape} {vad_features.dtype}")
    
    # 提取姿态特征
    print("\n提取姿态特征...")
    if 'pose' in sample:
        # 创建关节掩码（示例：只使用身体和头部关节）
        joint_names = ['global_orient', 'body_pose']  # 简化示例
        all_joint_names = ['global_orient'] * 3 + ['body_pose'] * 63 + ['left_hand_pose'] * 45 + ['right_hand_pose'] * 45
        joint_mask = create_joint_mask(joint_names, all_joint_names)
        
        pose_features = extract_pose_features(
            sample['pose'], 
            joint_mask=joint_mask,
            normalize=True
        )
        print(f"姿态特征: {pose_features.shape} {pose_features.dtype}")
    
    # 提取关键点特征
    print("\n提取关键点特征...")
    if 'keypoints' in sample:
        keypoint_features = extract_keypoint_features(
            sample['keypoints'], 
            normalize=True,
            confidence_threshold=0.5
        )
        print(f"关键点特征: {keypoint_features.shape} {keypoint_features.dtype}")
    
    # 提取情绪特征
    print("\n提取情绪特征...")
    if 'emotion_scores' in sample:
        emotion_features = extract_emotion_features(
            sample['emotion_scores'], 
            normalize=True
        )
        print(f"情绪特征: {emotion_features.shape} {emotion_features.dtype}")
    
    # 提取表情特征
    print("\n提取表情特征...")
    if 'expression' in sample:
        expression_features = extract_expression_features(
            sample['expression'], 
            normalize=True
        )
        print(f"表情特征: {expression_features.shape} {expression_features.dtype}")
    
    # 对齐音频和运动数据
    print("\n对齐音频和运动数据...")
    if 'audio' in sample and 'pose' in sample:
        aligned_audio = align_audio_to_motion(
            sample['audio'], 
            motion_length=len(sample['pose']),
            audio_fps=16000,
            motion_fps=30
        )
        print(f"原始音频: {sample['audio'].shape}")
        print(f"对齐后音频: {aligned_audio.shape}")
        
        # 提取音频特征
        audio_features = extract_audio_features(
            aligned_audio,
            sample_rate=16000,
            n_mels=80
        )
        print(f"音频特征: {audio_features.shape} {audio_features.dtype}")
    
    # 加载一个时间窗口样本
    print("\n加载时间窗口样本...")
    window_sample = window_dataset[0]
    
    # 打印窗口样本的键和数据形状
    print("窗口样本数据:")
    for key, value in window_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: 列表，长度 {len(value)}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    from torch.utils.data import DataLoader
    
    batch_size = 4
    dataloader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0以避免多进程问题
        collate_fn=custom_collate_fn
    )
    
    # 加载一个批次的数据
    print(f"\n加载一个批次的数据 (batch_size={batch_size})...")
    batch = next(iter(dataloader))
    
    # 打印批次数据的键和数据形状
    print("批次数据:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: 列表，长度 {len(value)}")
        else:
            print(f"  {key}: {type(value)}")


def custom_collate_fn(batch):
    """
    自定义批处理函数，用于处理不同长度的序列数据
    
    Args:
        batch: 样本列表
        
    Returns:
        批处理后的数据
    """
    # 初始化结果字典
    result = {}
    
    # 获取所有键
    keys = batch[0].keys()
    
    for key in keys:
        # 收集所有值
        values = [item[key] for item in batch]
        
        # 处理不同类型的数据
        if isinstance(values[0], torch.Tensor):
            # 如果是张量，尝试堆叠
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                # 如果形状不匹配，使用列表
                result[key] = values
        elif isinstance(values[0], np.ndarray):
            # 如果是numpy数组，转换为张量并尝试堆叠
            try:
                result[key] = torch.stack([torch.from_numpy(v) for v in values])
            except RuntimeError:
                # 如果形状不匹配，使用列表
                result[key] = [torch.from_numpy(v) for v in values]
        elif isinstance(values[0], list):
            # 如果是列表，保持为列表
            result[key] = values
        else:
            # 其他类型，保持为列表
            result[key] = values
    
    return result


if __name__ == "__main__":
    main()