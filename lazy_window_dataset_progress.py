import sys
import os
import time
import pickle
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataloaders.seamless_interaction import SeamlessInteractionDataset


class CachedLazySeamlessInteractionWindowDatasetWithProgress(Dataset):
    """
    带进度显示的缓存数据集版本
    在数据加载过程中显示详细的进度信息
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        sample_rate: int = 16000,
        pose_fps: int = 30,
        audio_fps: int = 16000,
        window_size: int = 64,
        window_stride: int = 20,
        multi_length_training: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        transform=None,
        load_video: bool = False,
        load_audio: bool = False,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        cache_path: Optional[str] = None,
        show_progress: bool = True,
        progress_interval: int = 100  # 每隔多少个样本显示一次进度
    ):
        """
        初始化带进度显示的缓存数据集

        Args:
            data_path: 数据集根目录路径
            split: 数据集分割 ('train', 'val', 'test')
            sample_rate: 音频采样率
            pose_fps: 姿态帧率
            audio_fps: 音频帧率
            window_size: 时间窗口大小(帧数)
            window_stride: 时间窗口步长(帧数)
            multi_length_training: 多长度训练比例列表
            transform: 数据变换
            load_video: 是否加载视频
            load_audio: 是否加载音频
            normalize: 是否标准化数据
            max_samples: 最大样本数，用于限制数据集大小
            cache_path: 缓存文件路径
            show_progress: 是否显示进度条
            progress_interval: 进度显示间隔
        """
        # 保存参数
        self.data_path = data_path
        self.split = split
        self.sample_rate = sample_rate
        self.pose_fps = pose_fps
        self.audio_fps = audio_fps
        self.window_size = window_size
        self.window_stride = window_stride
        self.multi_length_training = multi_length_training
        self.transform = transform
        self.load_video = load_video
        self.load_audio = load_audio
        self.normalize = normalize
        self.max_samples = max_samples
        self.show_progress = show_progress
        self.progress_interval = progress_interval

        # 如果是测试集，只使用1.0比例
        if split == "test":
            self.multi_length_training = [1.0]

        # 进度跟踪
        self.access_count = 0
        self.start_time = None

        # 从缓存文件加载窗口参数
        if cache_path and os.path.exists(cache_path):
            print(f"📂 从缓存文件加载窗口参数: {cache_path}")
            load_start = time.time()

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # 验证缓存参数是否匹配（允许窗口参数不匹配，只给出警告）
            if cache_data.get('window_size') != window_size:
                print(f"⚠️  警告: 缓存文件窗口大小({cache_data.get('window_size')})与请求的窗口大小({window_size})不匹配，将使用缓存参数")
                window_size = cache_data.get('window_size')
            if cache_data.get('window_stride') != window_stride:
                print(f"⚠️  警告: 缓存文件窗口步长({cache_data.get('window_stride')})与请求的窗口步长({window_stride})不匹配，将使用缓存参数")
                window_stride = cache_data.get('window_stride')
            if cache_data.get('split') != split:
                raise ValueError(f"❌ 缓存文件数据集分割({cache_data.get('split')})与请求的分割({split})不匹配")

            # 加载缓存数据
            self.window_counts = cache_data['window_counts']
            self.cumulative_windows = cache_data['cumulative_windows']
            self.window_params = cache_data['window_params']
            self.total_windows = cache_data['total_windows']
            self.base_dataset_indices = cache_data['base_dataset_indices']

            load_time = time.time() - load_start
            print(f"✅ 缓存加载完成，耗时: {load_time:.2f}秒")

            # 如果需要限制样本数
            if max_samples is not None and max_samples < len(self.base_dataset_indices):
                print(f"🔧 限制基础数据集大小: {len(self.base_dataset_indices)} -> {max_samples}")

                self.base_dataset_indices = self.base_dataset_indices[:max_samples]
                # 重新计算窗口相关数据
                self.window_counts = self.window_counts[:max_samples]
                self.cumulative_windows = [0]
                for count in self.window_counts:
                    self.cumulative_windows.append(self.cumulative_windows[-1] + count)
                self.total_windows = self.cumulative_windows[-1]
                # 重新计算窗口参数
                new_window_params = []
                for i in range(max_samples):
                    start_idx = sum(self.window_counts[:i])
                    end_idx = sum(self.window_counts[:i+1])
                    new_window_params.extend(self.window_params[start_idx:end_idx])
                self.window_params = new_window_params

                print(f"📊 限制后总窗口数: {self.total_windows}")
        else:
            raise FileNotFoundError(f"❌ 缓存文件不存在: {cache_path}")

        # 创建基础数据集
        print(f"🏗️  创建基础数据集...")
        dataset_start = time.time()

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

        dataset_time = time.time() - dataset_start
        print(f"✅ 基础数据集创建完成，耗时: {dataset_time:.2f}秒")
        print(f"📈 总窗口数: {self.total_windows:,}")
        print(f"🎯 数据集初始化完成！")

        # 初始化进度条
        if self.show_progress:
            self.progress_bar = None
            self.last_progress_update = 0

    def __len__(self) -> int:
        """返回窗口数量"""
        return self.total_windows

    def _update_progress(self, idx: int):
        """更新进度显示"""
        if not self.show_progress:
            return

        # 初始化进度条
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.total_windows,
                desc=f"📊 {self.split} 数据加载",
                unit="窗口",
                unit_scale=True,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            self.start_time = time.time()

        # 更新进度条
        current_progress = idx + 1
        self.progress_bar.update(current_progress - self.progress_bar.n)

        # 显示详细信息（每隔一定间隔）
        if current_progress % self.progress_interval == 0 or current_progress == self.total_windows:
            elapsed = time.time() - self.start_time if self.start_time else 0
            if elapsed > 0:
                rate = current_progress / elapsed
                eta = (self.total_windows - current_progress) / rate if rate > 0 else 0
                self.progress_bar.set_postfix({
                    '速度': f'{rate:.1f} 窗口/秒',
                    '剩余': f'{eta/60:.1f}分钟' if eta > 60 else f'{eta:.0f}秒'
                })

        # 完成时关闭进度条
        if current_progress == self.total_windows:
            self.progress_bar.close()
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"\n🎉 数据加载完成！")
            print(f"⏱️  总耗时: {total_time:.2f}秒")
            print(f"📊 平均速度: {self.total_windows/total_time:.1f} 窗口/秒")

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """获取单个窗口数据（带进度显示）"""
        # 更新进度
        self._update_progress(idx)

        # 找到窗口对应的基础样本索引
        sample_idx = self._find_sample_index(idx)

        # 计算窗口在样本中的位置
        window_idx_in_sample = idx - self.cumulative_windows[sample_idx]

        # 加载基础样本数据
        sample = self.base_dataset[self.base_dataset_indices[sample_idx]]

        # 获取序列长度
        if 'pose' in sample:
            seq_len = len(sample['pose'])
        elif 'keypoints' in sample:
            seq_len = len(sample['keypoints'])
        elif 'emotion_scores' in sample:
            seq_len = len(sample['emotion_scores'])
        else:
            return sample  # 如果没有序列数据，返回整个样本

        # 直接使用缓存的窗口参数，不需要重新计算
        cut_length, stride = self.window_params[idx]

        # 对于缓存的窗口参数，我们需要计算实际的start和end位置
        sample_start_window_idx = self.cumulative_windows[sample_idx]
        relative_window_idx = idx - sample_start_window_idx

        # 计算起始位置
        start = relative_window_idx * stride

        # 确保start不超过序列长度
        start = min(start, seq_len - cut_length if seq_len > cut_length else 0)
        start = max(0, start)

        # 计算结束位置
        end = min(start + cut_length, seq_len)

        # 创建窗口数据
        window_data = {}

        # 复制非序列数据
        for key, value in sample.items():
            if key not in ['pose', 'keypoints', 'emotion_scores', 'expression', 'audio']:
                window_data[key] = value

        # 提取序列数据的窗口
        if 'pose' in sample:
            window_data['pose'] = sample['pose'][start:end]

        if 'keypoints' in sample:
            window_data['keypoints'] = sample['keypoints'][start:end]

        if 'emotion_scores' in sample:
            window_data['emotion_scores'] = sample['emotion_scores'][start:end]

        if 'expression' in sample:
            window_data['expression'] = sample['expression'][start:end]

        # 处理音频窗口（如果存在）
        if 'audio' in sample and self.load_audio:
            audio_start = int(start * self.audio_fps / self.pose_fps)
            audio_end = int(end * self.audio_fps / self.pose_fps)
            if audio_end < len(sample['audio']):
                window_data['audio'] = sample['audio'][audio_start:audio_end]
            else:
                window_data['audio'] = sample['audio'][audio_start:]

        # 处理translation数据（如果存在）
        if 'translation' in sample:
            translation_data = sample['translation']
            if len(translation_data.shape) == 2:  # (T, 3)
                window_data['translation'] = translation_data[start:end]
            else:  # 假设是(3,) 或其他格式
                window_data['translation'] = translation_data

        # 应用变换
        if self.transform:
            window_data = self.transform(window_data)

        return window_data

    def _find_sample_index(self, window_idx: int) -> int:
        """二分查找窗口对应的基础样本索引"""
        import bisect
        return bisect.bisect_right(self.cumulative_windows, window_idx) - 1


def create_progress_dataset(*args, **kwargs):
    """创建带进度显示的数据集的便捷函数"""
    return CachedLazySeamlessInteractionWindowDatasetWithProgress(*args, **kwargs)