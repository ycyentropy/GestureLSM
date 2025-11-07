import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Union, Optional, Tuple
import random
import pickle
import os
from dataloaders.seamless_interaction import SeamlessInteractionDataset


class LazySeamlessInteractionWindowDataset(Dataset):
    """
    Seamless_Interaction数据集的惰性时间窗口版本
    每个样本是一个固定长度的时间窗口，但窗口是在需要时才计算的
    仿照BEAT数据集的处理方式，使用帧数作为窗口参数单位
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
        max_samples: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据集根目录路径
            base_dataset: 已创建的SeamlessInteractionDataset实例
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
        
        # 限制基础数据集大小
        if max_samples is not None and max_samples < len(self.base_dataset):
            self.base_dataset_indices = list(range(max_samples))
            print(f"基础数据集大小限制为: {max_samples}")
        else:
            self.base_dataset_indices = list(range(len(self.base_dataset)))
        
        # 保存原始窗口参数
        self.ori_window_size = window_size
        self.ori_window_stride = window_stride
        self.multi_length_training = multi_length_training
        self.pose_fps = pose_fps
        self.audio_fps = audio_fps
        
        # 如果是测试集，只使用1.0比例
        if split == "test":
            self.multi_length_training = [1.0]
        
        # 预计算每个样本的窗口数量，但不实际创建窗口
        self.window_counts = []
        self.cumulative_windows = [0]  # 累计窗口数，用于索引映射
        self.window_params = []  # 保存每个窗口的参数(长度, 步长)
        
        print("预计算窗口数量...")
        print(f"总样本数: {len(self.base_dataset_indices)}")
        print(f"窗口大小: {window_size} 帧, 窗口步长: {window_stride} 帧")
        print(f"多长度训练比例: {self.multi_length_training}")
        
        for i in range(len(self.base_dataset_indices)):
            # 获取样本信息而不加载实际数据
            sample_idx = self.base_dataset_indices[i]
            sample_info = self.base_dataset.get_sample_info(sample_idx)
            seq_len = sample_info['sequence_length']
            
            if seq_len is None:
                # 如果没有序列数据，计为1个窗口
                window_count = 1
                self.window_params.append((window_size, window_stride))
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
    
    def __len__(self) -> int:
        """返回窗口数量"""
        return self.total_windows
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """获取单个窗口数据"""
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
        
        # 获取当前窗口的参数
        cut_length, stride = self.window_params[idx]
        
        # 计算窗口在样本中的实际位置
        # 需要考虑多长度训练的情况，计算当前窗口在所有比例窗口中的偏移
        start_offset = 0
        found = False
        for ratio in self.multi_length_training:
            if self.base_dataset.split == "test":
                # 测试时使用整个序列作为单个窗口
                cur_cut_length = seq_len
                cur_stride = seq_len
                cur_num_subdivision = 1
            else:
                # 训练时使用比例调整窗口大小和步长
                cur_cut_length = int(self.ori_window_size * ratio)
                cur_stride = int(self.ori_window_stride * ratio)
                # 确保至少有一个窗口，即使序列长度小于窗口大小
                if seq_len <= cur_cut_length:
                    cur_num_subdivision = 1
                else:
                    cur_num_subdivision = (seq_len - cur_cut_length) // cur_stride + 1
            
            if window_idx_in_sample < cur_num_subdivision:
                # 找到当前窗口所在的比例
                start = start_offset + window_idx_in_sample * cur_stride
                end = start + cur_cut_length
                found = True
                break
            
            start_offset += cur_num_subdivision * cur_stride
            window_idx_in_sample -= cur_num_subdivision
        
        if not found:
            # 如果没有找到，使用默认值
            start = 0
            end = min(cut_length, seq_len)
        
        # 确保窗口不超出序列长度
        if end > seq_len:
            end = seq_len
            start = max(0, end - cut_length)
        
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
        if 'audio' in sample and self.base_dataset.load_audio:
            audio_start = int(start * self.audio_fps / self.pose_fps)
            audio_end = int(end * self.audio_fps / self.pose_fps)
            if audio_end <= len(sample['audio']):
                window_data['audio'] = sample['audio'][audio_start:audio_end]
            else:
                window_data['audio'] = sample['audio'][audio_start:]
        
        return window_data
    
    def _find_sample_index(self, window_idx: int) -> int:
        """
        根据窗口索引找到对应的基础样本索引
        
        Args:
            window_idx: 窗口索引
            
        Returns:
            基础样本索引
        """
        # 使用二分查找找到对应的样本
        left, right = 0, len(self.cumulative_windows) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_windows[mid] <= window_idx < self.cumulative_windows[mid + 1]:
                return mid
            elif window_idx < self.cumulative_windows[mid]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def get_random_window(self) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """
        获取一个随机窗口
        
        Returns:
            随机窗口数据
        """
        random_idx = random.randint(0, self.__len__() - 1)
        return self.__getitem__(random_idx)
    
    def get_random_windows(self, num_windows: int) -> List[Dict[str, Union[torch.Tensor, np.ndarray, Dict]]]:
        """
        获取多个随机窗口
        
        Args:
            num_windows: 窗口数量
            
        Returns:
            随机窗口数据列表
        """
        windows = []
        for _ in range(num_windows):
            windows.append(self.get_random_window())
        return windows
    
    def get_window_ratio(self, idx: int) -> float:
        """
        获取指定窗口的长度比例
        
        Args:
            idx: 窗口索引
            
        Returns:
            窗口长度比例
        """
        # 找到窗口对应的基础样本索引
        sample_idx = self._find_sample_index(idx)
        
        # 计算窗口在样本中的位置
        window_idx_in_sample = idx - self.cumulative_windows[sample_idx]
        
        # 获取样本信息
        sample_idx_data = self.base_dataset_indices[sample_idx]
        sample_info = self.base_dataset.get_sample_info(sample_idx_data)
        seq_len = sample_info['sequence_length']
        
        # 获取当前窗口的参数
        cut_length, stride = self.window_params[idx]
        
        # 计算窗口在样本中的实际位置
        start_offset = 0
        for ratio in self.multi_length_training:
            if self.base_dataset.split == "test":
                # 测试时使用整个序列作为单个窗口
                cur_cut_length = seq_len
                cur_stride = seq_len
                cur_num_subdivision = 1
            else:
                # 训练时使用比例调整窗口大小和步长
                cur_cut_length = int(self.ori_window_size * ratio)
                cur_stride = int(self.ori_window_stride * ratio)
                cur_num_subdivision = max(0, (seq_len - cur_cut_length) // cur_stride + 1)
            
            if window_idx_in_sample < cur_num_subdivision:
                # 找到当前窗口所在的比例
                return ratio
            
            start_offset += cur_num_subdivision * cur_stride
            window_idx_in_sample -= cur_num_subdivision
        
        return 1.0  # 默认返回1.0
    
    def get_window_info(self, idx: int) -> Dict[str, Union[int, float, str]]:
        """
        获取指定窗口的详细参数信息
        
        Args:
            idx: 窗口索引
            
        Returns:
            包含窗口详细信息的字典
        """
        # 找到窗口对应的基础样本索引
        sample_idx = self._find_sample_index(idx)
        
        # 计算窗口在样本中的位置
        window_idx_in_sample = idx - self.cumulative_windows[sample_idx]
        
        # 获取样本信息
        sample_idx_data = self.base_dataset_indices[sample_idx]
        sample_info = self.base_dataset.get_sample_info(sample_idx_data)
        seq_len = sample_info['sequence_length']
        
        # 获取当前窗口的参数
        cut_length, stride = self.window_params[idx]
        
        # 计算窗口在样本中的实际位置
        start_offset = 0
        for ratio in self.multi_length_training:
            if self.base_dataset.split == "test":
                # 测试时使用整个序列作为单个窗口
                cur_cut_length = seq_len
                cur_stride = seq_len
                cur_num_subdivision = 1
            else:
                # 训练时使用比例调整窗口大小和步长
                cur_cut_length = int(self.ori_window_size * ratio)
                cur_stride = int(self.ori_window_stride * ratio)
                cur_num_subdivision = max(0, (seq_len - cur_cut_length) // cur_stride + 1)
            
            if window_idx_in_sample < cur_num_subdivision:
                # 找到当前窗口所在的比例
                return {
                    "window_index": idx,
                    "sample_index": sample_idx,
                    "window_in_sample": window_idx_in_sample,
                    "sequence_length": seq_len,
                    "cut_length": cut_length,
                    "stride": stride,
                    "ratio": ratio,
                    "split": self.base_dataset.split,
                    "start_frame": start_offset + window_idx_in_sample * stride,
                    "end_frame": start_offset + window_idx_in_sample * stride + cut_length
                }
            
            start_offset += cur_num_subdivision * cur_stride
            window_idx_in_sample -= cur_num_subdivision
        
        # 默认返回信息
        return {
            "window_index": idx,
            "sample_index": sample_idx,
            "window_in_sample": window_idx_in_sample,
            "sequence_length": seq_len,
            "cut_length": cut_length,
            "stride": stride,
            "ratio": 1.0,
            "split": self.base_dataset.split,
            "start_frame": start_offset + window_idx_in_sample * stride,
            "end_frame": start_offset + window_idx_in_sample * stride + cut_length
        }


class CachedLazySeamlessInteractionWindowDataset(Dataset):
    """
    使用预计算窗口参数的Seamless_Interaction数据集
    从缓存文件加载窗口参数，避免重复计算
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
        cache_path: Optional[str] = None
    ):
        """
        初始化缓存数据集
        
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
        
        # 如果是测试集，只使用1.0比例
        if split == "test":
            self.multi_length_training = [1.0]
        
        # 从缓存文件加载窗口参数
        if cache_path and os.path.exists(cache_path):
            print(f"从缓存文件加载窗口参数: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # 验证缓存参数是否匹配（允许窗口参数不匹配，只给出警告）
            if cache_data.get('window_size') != window_size:
                print(f"警告: 缓存文件窗口大小({cache_data.get('window_size')})与请求的窗口大小({window_size})不匹配，将使用缓存参数")
                window_size = cache_data.get('window_size')
            if cache_data.get('window_stride') != window_stride:
                print(f"警告: 缓存文件窗口步长({cache_data.get('window_stride')})与请求的窗口步长({window_stride})不匹配，将使用缓存参数")
                window_stride = cache_data.get('window_stride')
            if cache_data.get('split') != split:
                raise ValueError(f"缓存文件数据集分割({cache_data.get('split')})与请求的分割({split})不匹配")
            
            # 加载缓存数据
            self.window_counts = cache_data['window_counts']
            self.cumulative_windows = cache_data['cumulative_windows']
            self.window_params = cache_data['window_params']
            self.total_windows = cache_data['total_windows']
            self.base_dataset_indices = cache_data['base_dataset_indices']
            
            # 如果需要限制样本数
            if max_samples is not None and max_samples < len(self.base_dataset_indices):
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
                
                print(f"限制基础数据集大小为: {max_samples}")
        else:
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
        
        # 创建基础数据集
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
        
        print(f"从缓存加载完成，总共有 {self.total_windows} 个时间窗口")
    
    def __len__(self) -> int:
        """返回窗口数量"""
        return self.total_windows
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """获取单个窗口数据"""
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
        # 缓存文件已经存储了正确的窗口位置和大小
        cut_length, stride = self.window_params[idx]

        # 对于缓存的窗口参数，我们需要计算实际的start和end位置
        # 由于缓存文件存储的是最终的窗口参数，我们直接使用它们
        # 但需要计算在当前样本中的具体位置

        # 获取当前样本在窗口参数中的起始位置
        sample_start_window_idx = self.cumulative_windows[sample_idx]

        # 计算当前窗口在样本中的相对位置
        relative_window_idx = idx - sample_start_window_idx

        # 计算起始位置
        start = relative_window_idx * stride

        # 确保start不超过序列长度
        start = min(start, seq_len - cut_length if seq_len > cut_length else 0)
        start = max(0, start)

        # 计算结束位置
        end = min(start + cut_length, seq_len)

        # 确保窗口大小正确
        if end - start < cut_length and start > 0:
            # 如果窗口太小，向前扩展
            start = max(0, end - cut_length)

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
        if 'audio' in sample and self.base_dataset.load_audio:
            audio_start = int(start * self.audio_fps / self.pose_fps)
            audio_end = int(end * self.audio_fps / self.pose_fps)
            if audio_end <= len(sample['audio']):
                window_data['audio'] = sample['audio'][audio_start:audio_end]
            else:
                window_data['audio'] = sample['audio'][audio_start:]

        return window_data
    
    def _find_sample_index(self, window_idx: int) -> int:
        """
        根据窗口索引找到对应的基础样本索引
        
        Args:
            window_idx: 窗口索引
            
        Returns:
            基础样本索引
        """
        # 使用二分查找找到对应的样本
        left, right = 0, len(self.cumulative_windows) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_windows[mid] <= window_idx < self.cumulative_windows[mid + 1]:
                return mid
            elif window_idx < self.cumulative_windows[mid]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def get_random_window(self) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """
        获取一个随机窗口
        
        Returns:
            随机窗口数据
        """
        random_idx = random.randint(0, self.__len__() - 1)
        return self.__getitem__(random_idx)
    
    def get_random_windows(self, num_windows: int) -> List[Dict[str, Union[torch.Tensor, np.ndarray, Dict]]]:
        """
        获取多个随机窗口
        
        Args:
            num_windows: 窗口数量
            
        Returns:
            随机窗口数据列表
        """
        windows = []
        for _ in range(num_windows):
            windows.append(self.get_random_window())
        return windows
    
    def get_window_ratio(self, idx: int) -> float:
        """
        获取指定窗口的长度比例
        
        Args:
            idx: 窗口索引
            
        Returns:
            窗口长度比例
        """
        # 找到窗口对应的基础样本索引
        sample_idx = self._find_sample_index(idx)
        
        # 计算窗口在样本中的位置
        window_idx_in_sample = idx - self.cumulative_windows[sample_idx]
        
        # 获取样本信息
        sample_idx_data = self.base_dataset_indices[sample_idx]
        sample_info = self.base_dataset.get_sample_info(sample_idx_data)
        seq_len = sample_info['sequence_length']
        
        # 获取当前窗口的参数
        cut_length, stride = self.window_params[idx]
        
        # 计算窗口在样本中的实际位置
        start_offset = 0
        for ratio in self.multi_length_training:
            if self.split == "test":
                # 测试时使用整个序列作为单个窗口
                cur_cut_length = seq_len
                cur_stride = seq_len
                cur_num_subdivision = 1
            else:
                # 训练时使用比例调整窗口大小和步长
                cur_cut_length = int(self.window_size * ratio)
                cur_stride = int(self.window_stride * ratio)
                cur_num_subdivision = max(0, (seq_len - cur_cut_length) // cur_stride + 1)
            
            if window_idx_in_sample < cur_num_subdivision:
                # 找到当前窗口所在的比例
                return ratio
            
            start_offset += cur_num_subdivision * cur_stride
            window_idx_in_sample -= cur_num_subdivision
        
        return 1.0  # 默认返回1.0
    
    def get_window_info(self, idx: int) -> Dict[str, Union[int, float, str]]:
        """
        获取指定窗口的详细参数信息
        
        Args:
            idx: 窗口索引
            
        Returns:
            包含窗口详细信息的字典
        """
        # 找到窗口对应的基础样本索引
        sample_idx = self._find_sample_index(idx)
        
        # 计算窗口在样本中的位置
        window_idx_in_sample = idx - self.cumulative_windows[sample_idx]
        
        # 获取样本信息
        sample_idx_data = self.base_dataset_indices[sample_idx]
        sample_info = self.base_dataset.get_sample_info(sample_idx_data)
        seq_len = sample_info['sequence_length']
        
        # 获取当前窗口的参数
        cut_length, stride = self.window_params[idx]
        
        # 计算窗口在样本中的实际位置
        start_offset = 0
        for ratio in self.multi_length_training:
            if self.split == "test":
                # 测试时使用整个序列作为单个窗口
                cur_cut_length = seq_len
                cur_stride = seq_len
                cur_num_subdivision = 1
            else:
                # 训练时使用比例调整窗口大小和步长
                cur_cut_length = int(self.window_size * ratio)
                cur_stride = int(self.window_stride * ratio)
                cur_num_subdivision = max(0, (seq_len - cur_cut_length) // cur_stride + 1)
            
            if window_idx_in_sample < cur_num_subdivision:
                # 找到当前窗口所在的比例
                return {
                    "window_index": idx,
                    "sample_index": sample_idx,
                    "window_in_sample": window_idx_in_sample,
                    "sequence_length": seq_len,
                    "cut_length": cut_length,
                    "stride": stride,
                    "ratio": ratio,
                    "split": self.split,
                    "start_frame": start_offset + window_idx_in_sample * stride,
                    "end_frame": start_offset + window_idx_in_sample * stride + cut_length
                }
            
            start_offset += cur_num_subdivision * cur_stride
            window_idx_in_sample -= cur_num_subdivision
        
        # 默认返回信息
        return {
            "window_index": idx,
            "sample_index": sample_idx,
            "window_in_sample": window_idx_in_sample,
            "sequence_length": seq_len,
            "cut_length": cut_length,
            "stride": stride,
            "ratio": 1.0,
            "split": self.split,
            "start_frame": start_offset + window_idx_in_sample * stride,
            "end_frame": start_offset + window_idx_in_sample * stride + cut_length
        }