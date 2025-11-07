#!/usr/bin/env python3
"""
保存和加载窗口参数的工具脚本
这个脚本可以预计算窗口参数并保存到文件，以便在后续训练中直接加载，避免重复计算
"""

import os
import pickle
import json
import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple
from lazy_window_dataset import LazySeamlessInteractionWindowDataset
from dataloaders.seamless_interaction import SeamlessInteractionDataset


def save_window_params(
    data_path: str,
    split: str,
    window_size: int,
    window_stride: int,
    multi_length_training: List[float],
    pose_fps: int = 30,
    audio_fps: int = 16000,
    max_samples: Optional[int] = None,
    save_path: Optional[str] = None
) -> str:
    """
    预计算并保存窗口参数
    
    Args:
        data_path: 数据集路径
        split: 数据集分割 ('train', 'val', 'test')
        window_size: 窗口大小(帧数)
        window_stride: 窗口步长(帧数)
        multi_length_training: 多长度训练比例列表
        pose_fps: 姿态帧率
        audio_fps: 音频帧率
        max_samples: 最大样本数
        save_path: 保存路径，如果为None则自动生成
        
    Returns:
        保存的文件路径
    """
    # 创建基础数据集
    base_dataset = SeamlessInteractionDataset(
        data_path=data_path,
        split=split,
        sample_rate=16000,
        pose_fps=pose_fps,
        audio_fps=audio_fps,
        window_size=window_size / pose_fps,  # 转换为秒数
        window_stride=window_stride / pose_fps,  # 转换为秒数
        transform=None,
        load_video=False,
        load_audio=False,
        normalize=True
    )
    
    # 限制基础数据集大小
    if max_samples is not None and max_samples < len(base_dataset):
        base_dataset_indices = list(range(max_samples))
        print(f"基础数据集大小限制为: {max_samples}")
    else:
        base_dataset_indices = list(range(len(base_dataset)))
    
    # 如果是测试集，只使用1.0比例
    if split == "test":
        multi_length_training = [1.0]
    
    # 预计算每个样本的窗口数量，但不实际创建窗口
    window_counts = []
    cumulative_windows = [0]  # 累计窗口数，用于索引映射
    window_params = []  # 保存每个窗口的参数(长度, 步长)
    
    print("预计算窗口数量...")
    print(f"总样本数: {len(base_dataset_indices)}")
    print(f"窗口大小: {window_size} 帧, 窗口步长: {window_stride} 帧")
    print(f"多长度训练比例: {multi_length_training}")
    
    for i in range(len(base_dataset_indices)):
        # 获取样本信息而不加载实际数据
        sample_idx = base_dataset_indices[i]
        sample_info = base_dataset.get_sample_info(sample_idx)
        seq_len = sample_info['sequence_length']
        
        if seq_len is None:
            # 如果没有序列数据，计为1个窗口
            window_count = 1
            window_params.append((window_size, window_stride))
            print(f"样本 {i+1}/{len(base_dataset_indices)} (索引: {sample_idx}): 无序列长度数据，计为1个窗口")
        else:
            # 为每个长度比例计算窗口数量
            sample_window_count = 0
            ratio_window_counts = {}  # 记录每个比例的窗口数
            for ratio in multi_length_training:
                if split == "test":
                    # 测试时使用整个序列作为单个窗口
                    cut_length = seq_len
                    stride = seq_len
                else:
                    # 训练时使用比例调整窗口大小和步长
                    cut_length = int(window_size * ratio)
                    stride = int(window_stride * ratio)
                
                # 计算窗口数量，使用BEAT数据集的计算方式
                # 确保至少有一个窗口，即使序列长度小于窗口大小
                if seq_len <= cut_length:
                    num_subdivision = 1
                else:
                    num_subdivision = (seq_len - cut_length) // stride + 1
                sample_window_count += num_subdivision
                ratio_window_counts[ratio] = num_subdivision
                
                # 保存每个窗口的参数
                for _ in range(num_subdivision):
                    window_params.append((cut_length, stride))
            
            window_count = sample_window_count
            # 打印详细信息，包括每个比例的窗口数
            ratio_str = ", ".join([f"{ratio}:{count}" for ratio, count in ratio_window_counts.items()])
            print(f"样本 {i+1}/{len(base_dataset_indices)} (索引: {sample_idx}): 序列长度={seq_len}, 各比例窗口数[{ratio_str}], 总窗口数={window_count}")
        
        window_counts.append(window_count)
        cumulative_windows.append(cumulative_windows[-1] + window_count)
        
        # 每处理100个样本输出一次进度
        if (i + 1) % 100 == 0 or i == len(base_dataset_indices) - 1:
            print(f"已处理 {i+1}/{len(base_dataset_indices)} 个样本，当前总窗口数: {cumulative_windows[-1]}")
    
    total_windows = cumulative_windows[-1]
    print(f"预计算完成，总共有 {total_windows} 个时间窗口")
    
    # 准备保存的数据
    window_data = {
        'window_counts': window_counts,
        'cumulative_windows': cumulative_windows,
        'window_params': window_params,
        'total_windows': total_windows,
        'base_dataset_indices': base_dataset_indices,
        'window_size': window_size,
        'window_stride': window_stride,
        'multi_length_training': multi_length_training,
        'pose_fps': pose_fps,
        'audio_fps': audio_fps,
        'split': split,
        'data_path': data_path
    }
    
    # 确定保存路径
    if save_path is None:
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(data_path), 'window_params')
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"window_params_{split}_ws{window_size}_ws{window_stride}"
        if max_samples is not None:
            filename += f"_max{max_samples}"
        filename += ".pkl"
        save_path = os.path.join(save_dir, filename)
    
    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump(window_data, f)
    
    # 同时保存一个JSON格式的元数据文件，便于查看
    metadata_path = save_path.replace('.pkl', '_metadata.json')
    metadata = {
        'total_windows': total_windows,
        'window_size': window_size,
        'window_stride': window_stride,
        'multi_length_training': multi_length_training,
        'split': split,
        'data_path': data_path,
        'num_samples': len(base_dataset_indices)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"窗口参数已保存到: {save_path}")
    print(f"元数据已保存到: {metadata_path}")
    
    return save_path


def load_window_params(save_path: str) -> Dict:
    """
    加载预计算的窗口参数
    
    Args:
        save_path: 保存的窗口参数文件路径
        
    Returns:
        包含窗口参数的字典
    """
    with open(save_path, 'rb') as f:
        window_data = pickle.load(f)
    
    print(f"已加载窗口参数，总共有 {window_data['total_windows']} 个时间窗口")
    return window_data


class CachedLazySeamlessInteractionWindowDataset(LazySeamlessInteractionWindowDataset):
    """
    使用缓存的窗口参数的LazySeamlessInteractionWindowDataset
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        base_dataset: Optional[SeamlessInteractionDataset] = None,
        split: str = "train",
        sample_rate: int = 16000,
        pose_fps: int = 30,
        audio_fps: int = 16000,
        window_size: int = 64,  # 改为帧数单位
        window_stride: int = 32,  # 改为帧数单位
        multi_length_training: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],  # 多长度训练比例
        transform=None,
        load_video: bool = False,
        load_audio: bool = False,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        cache_path: Optional[str] = None
    ):
        """
        初始化数据集，使用缓存的窗口参数
        
        Args:
            cache_path: 缓存的窗口参数文件路径，如果为None则尝试自动查找
            其他参数与LazySeamlessInteractionWindowDataset相同
        """
        # 保存基础数据集
        if isinstance(base_dataset, SeamlessInteractionDataset):
            self.base_dataset = base_dataset
        else:
            self.base_dataset = SeamlessInteractionDataset(
                data_path=data_path,
                split=split,
                sample_rate=sample_rate,
                pose_fps=pose_fps,
                audio_fps=audio_fps,
                window_size=window_size / pose_fps,  # 转换为秒数传递给基础数据集
                window_stride=window_stride / pose_fps,  # 转换为秒数传递给基础数据集
                transform=transform,
                load_video=load_video,
                load_audio=load_audio,
                normalize=normalize
            )
        
        # 保存原始窗口参数
        self.ori_window_size = window_size
        self.ori_window_stride = window_stride
        self.multi_length_training = multi_length_training
        self.pose_fps = pose_fps
        self.audio_fps = audio_fps
        
        # 如果是测试集，只使用1.0比例
        if split == "test":
            self.multi_length_training = [1.0]
        
        # 尝试加载缓存的窗口参数
        if cache_path is None:
            # 尝试自动查找缓存文件
            cache_dir = os.path.join(os.path.dirname(data_path), 'window_params')
            filename = f"window_params_{split}_ws{window_size}_ws{window_stride}"
            if max_samples is not None:
                filename += f"_max{max_samples}"
            filename += ".pkl"
            cache_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(cache_path):
            print(f"加载缓存的窗口参数: {cache_path}")
            window_data = load_window_params(cache_path)
            
            # 验证参数是否匹配
            if (window_data['window_size'] != window_size or
                window_data['window_stride'] != window_stride or
                window_data['multi_length_training'] != self.multi_length_training or
                window_data['split'] != split):
                print("警告: 缓存的参数与当前参数不匹配，将重新计算窗口参数")
                self._compute_windows(data_path, split, pose_fps, audio_fps, max_samples)
            else:
                # 使用缓存的参数
                self.window_counts = window_data['window_counts']
                self.cumulative_windows = window_data['cumulative_windows']
                self.window_params = window_data['window_params']
                self.total_windows = window_data['total_windows']
                self.base_dataset_indices = window_data['base_dataset_indices']
                
                # 如果max_samples指定了，但缓存的数据更多，需要截断
                if max_samples is not None and len(self.base_dataset_indices) > max_samples:
                    self.base_dataset_indices = list(range(max_samples))
                    # 重新计算窗口数量
                    self._recompute_windows_for_max_samples(max_samples)
                
                print(f"成功加载缓存的窗口参数，总共有 {self.total_windows} 个时间窗口")
        else:
            print(f"未找到缓存文件: {cache_path}，将重新计算窗口参数")
            self._compute_windows(data_path, split, pose_fps, audio_fps, max_samples)
    
    def _compute_windows(self, data_path, split, pose_fps, audio_fps, max_samples):
        """计算窗口参数，与父类相同的逻辑"""
        # 限制基础数据集大小
        if max_samples is not None and max_samples < len(self.base_dataset):
            self.base_dataset_indices = list(range(max_samples))
            print(f"基础数据集大小限制为: {max_samples}")
        else:
            self.base_dataset_indices = list(range(len(self.base_dataset)))
        
        # 预计算每个样本的窗口数量，但不实际创建窗口
        self.window_counts = []
        self.cumulative_windows = [0]  # 累计窗口数，用于索引映射
        self.window_params = []  # 保存每个窗口的参数(长度, 步长)
        
        print("预计算窗口数量...")
        print(f"总样本数: {len(self.base_dataset_indices)}")
        print(f"窗口大小: {self.ori_window_size} 帧, 窗口步长: {self.ori_window_stride} 帧")
        print(f"多长度训练比例: {self.multi_length_training}")
        
        for i in range(len(self.base_dataset_indices)):
            # 获取样本信息而不加载实际数据
            sample_idx = self.base_dataset_indices[i]
            sample_info = self.base_dataset.get_sample_info(sample_idx)
            seq_len = sample_info['sequence_length']
            
            if seq_len is None:
                # 如果没有序列数据，计为1个窗口
                window_count = 1
                self.window_params.append((self.ori_window_size, self.ori_window_stride))
                print(f"样本 {i+1}/{len(self.base_dataset_indices)} (索引: {sample_idx}): 无序列长度数据，计为1个窗口")
            else:
                # 为每个长度比例计算窗口数量
                sample_window_count = 0
                ratio_window_counts = {}  # 记录每个比例的窗口数
                for ratio in self.multi_length_training:
                    if split == "test":
                        # 测试时使用整个序列作为单个窗口
                        cut_length = seq_len
                        stride = seq_len
                    else:
                        # 训练时使用比例调整窗口大小和步长
                        cut_length = int(self.ori_window_size * ratio)
                        stride = int(self.ori_window_stride * ratio)
                    
                    # 计算窗口数量，使用BEAT数据集的计算方式
                    # 确保至少有一个窗口，即使序列长度小于窗口大小
                    if seq_len <= cut_length:
                        num_subdivision = 1
                    else:
                        num_subdivision = (seq_len - cut_length) // stride + 1
                    sample_window_count += num_subdivision
                    ratio_window_counts[ratio] = num_subdivision
                    
                    # 保存每个窗口的参数
                    for _ in range(num_subdivision):
                        self.window_params.append((cut_length, stride))
                
                window_count = sample_window_count
                # 打印详细信息，包括每个比例的窗口数
                ratio_str = ", ".join([f"{ratio}:{count}" for ratio, count in ratio_window_counts.items()])
                print(f"样本 {i+1}/{len(self.base_dataset_indices)} (索引: {sample_idx}): 序列长度={seq_len}, 各比例窗口数[{ratio_str}], 总窗口数={window_count}")
            
            self.window_counts.append(window_count)
            self.cumulative_windows.append(self.cumulative_windows[-1] + window_count)
            
            # 每处理100个样本输出一次进度
            if (i + 1) % 100 == 0 or i == len(self.base_dataset_indices) - 1:
                print(f"已处理 {i+1}/{len(self.base_dataset_indices)} 个样本，当前总窗口数: {self.cumulative_windows[-1]}")
        
        self.total_windows = self.cumulative_windows[-1]
        print(f"预计算完成，总共有 {self.total_windows} 个时间窗口")
    
    def _recompute_windows_for_max_samples(self, max_samples):
        """为指定的max_samples重新计算窗口数量"""
        new_window_counts = self.window_counts[:max_samples]
        new_cumulative_windows = [0]
        for count in new_window_counts:
            new_cumulative_windows.append(new_cumulative_windows[-1] + count)
        
        # 计算新的窗口参数范围
        new_total_windows = new_cumulative_windows[-1]
        new_window_params = self.window_params[:new_total_windows]
        
        self.window_counts = new_window_counts
        self.cumulative_windows = new_cumulative_windows
        self.window_params = new_window_params
        self.total_windows = new_total_windows
        
        print(f"为max_samples={max_samples}重新计算窗口数量，总共有 {self.total_windows} 个时间窗口")
    
    def save_window_params(self, save_path: Optional[str] = None):
        """
        保存当前窗口参数到文件
        
        Args:
            save_path: 保存路径，如果为None则自动生成
        """
        # 准备保存的数据
        window_data = {
            'window_counts': self.window_counts,
            'cumulative_windows': self.cumulative_windows,
            'window_params': self.window_params,
            'total_windows': self.total_windows,
            'base_dataset_indices': self.base_dataset_indices,
            'window_size': self.ori_window_size,
            'window_stride': self.ori_window_stride,
            'multi_length_training': self.multi_length_training,
            'pose_fps': self.pose_fps,
            'audio_fps': self.audio_fps,
            'split': self.base_dataset.split,
            'data_path': self.base_dataset.data_path
        }
        
        # 确定保存路径
        if save_path is None:
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(self.base_dataset.data_path), 'window_params')
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            filename = f"window_params_{self.base_dataset.split}_ws{self.ori_window_size}_ws{self.ori_window_stride}"
            if len(self.base_dataset_indices) < len(self.base_dataset):
                filename += f"_max{len(self.base_dataset_indices)}"
            filename += ".pkl"
            save_path = os.path.join(save_dir, filename)
        
        # 保存数据
        with open(save_path, 'wb') as f:
            pickle.dump(window_data, f)
        
        print(f"窗口参数已保存到: {save_path}")
        return save_path


def main():
    """
    主函数，用于预计算并保存窗口参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='预计算并保存窗口参数')
    parser.add_argument('--data_path', type=str, default='data/seamless_motions',
                        help='数据集路径')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集分割 (train, val, test)')
    parser.add_argument('--window_size', type=int, default=64,
                        help='窗口大小(帧数)')
    parser.add_argument('--window_stride', type=int, default=20,
                        help='窗口步长(帧数)')
    parser.add_argument('--multi_length_training', action='store_true',
                        help='是否使用多长度训练')
    parser.add_argument('--pose_fps', type=int, default=30,
                        help='姿态帧率')
    parser.add_argument('--audio_fps', type=int, default=16000,
                        help='音频帧率')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存路径')
    
    args = parser.parse_args()
    
    # 处理多长度训练参数
    if args.multi_length_training:
        multi_length_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
    else:
        multi_length_ratios = [1.0]
    
    # 保存窗口参数
    save_path = save_window_params(
        data_path=args.data_path,
        split=args.split,
        window_size=args.window_size,
        window_stride=args.window_stride,
        multi_length_training=multi_length_ratios,
        pose_fps=args.pose_fps,
        audio_fps=args.audio_fps,
        max_samples=args.max_samples,
        save_path=args.save_path
    )
    
    print(f"窗口参数已保存到: {save_path}")


if __name__ == "__main__":
    main()