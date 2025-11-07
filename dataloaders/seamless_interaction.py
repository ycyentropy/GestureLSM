import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import librosa
from pathlib import Path


class SeamlessInteractionDataset(Dataset):
    """
    Seamless_Interaction数据集加载器
    
    该数据集包含多模态交互数据，包括：
    - 文本转录 (JSON文件)
    - 音频 (WAV文件)
    - 视频 (MP4文件)
    - 多模态特征 (NPZ文件)
    
    NPZ文件包含以下数据类型：
    - boxes_and_keypoints: 边界框和关键点数据
    - movement: 运动和情绪特征
    - smplh: 人体姿态参数
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        pose_fps: int = 30,
        audio_fps: int = 16000,
        window_size: float = 2.0,
        window_stride: float = 0.5,
        transform=None,
        load_video: bool = False,
        load_audio: bool = True,
        normalize: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据集根目录路径
            split: 数据集分割 ('train', 'val', 'test')
            sample_rate: 音频采样率
            pose_fps: 姿态帧率
            audio_fps: 音频帧率
            window_size: 时间窗口大小(秒)
            window_stride: 时间窗口步长(秒)
            transform: 数据变换
            load_video: 是否加载视频
            load_audio: 是否加载音频
            normalize: 是否标准化数据
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = sample_rate
        self.pose_fps = pose_fps
        self.audio_fps = audio_fps
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.load_video = load_video
        self.load_audio = load_audio
        self.normalize = normalize
        
        # 调试打印标志，默认关闭，避免多进程环境下重复打印日志
        self._debug_print = False
        
        # 查找所有数据文件
        self.data_files = self._find_data_files()
        print(f"找到 {len(self.data_files)} 个数据样本")
        
        # 预计算窗口信息
        self.pose_window_size = int(window_size * pose_fps)
        self.pose_window_stride = int(window_stride * pose_fps)
        self.audio_window_size = int(window_size * audio_fps)
        self.audio_window_stride = int(window_stride * audio_fps)
        
    def _find_data_files(self) -> List[Dict[str, str]]:
        """查找所有数据文件并组织成样本列表"""
        # 根据split参数选择对应的子目录
        # 数据集结构: /datasets/seamless_interaction/improvised/{split}/
        split_path = self.data_path / "improvised" / self.split
        if not split_path.exists():
            print(f"警告: split目录不存在: {split_path}")
            # 如果split目录不存在，回退到improvised目录
            split_path = self.data_path / "improvised"
            if not split_path.exists():
                # 如果improvised目录也不存在，回退到根目录
                split_path = self.data_path
        
        # 根据数据集结构查找文件
        json_files = list(split_path.glob("**/*.json"))
        
        data_samples = []
        for json_file in json_files:
            base_name = json_file.stem
            npz_file = json_file.with_suffix('.npz')
            wav_file = json_file.with_suffix('.wav')
            mp4_file = json_file.with_suffix('.mp4')
            
            # 检查必要文件是否存在
            if not npz_file.exists():
                continue
                
            sample = {
                'id': base_name,
                'json_path': str(json_file),
                'npz_path': str(npz_file),
                'wav_path': str(wav_file) if wav_file.exists() else None,
                'mp4_path': str(mp4_file) if mp4_file.exists() else None,
            }
            
            data_samples.append(sample)
            
        return data_samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含多模态数据的字典
        """
        sample_info = self.data_files[idx]

        # 加载NPZ文件（多模态特征）
        npz_data = np.load(sample_info['npz_path'])

        # 加载JSON文件（文本转录）
        with open(sample_info['json_path'], 'r') as f:
            json_data = json.load(f)
        
        # 提取关键数据
        result = {
            'id': sample_info['id'],
            'transcript': json_data.get('metadata:transcript', []),
            'vad': json_data.get('metadata:vad', []),
        }
        
        # 处理SMPL-H姿态数据
        if 'smplh:body_pose' in npz_data:
            body_pose = npz_data['smplh:body_pose']  # (T, 21, 3)
            global_orient = npz_data['smplh:global_orient']  # (T, 3)
            left_hand_pose = npz_data.get('smplh:left_hand_pose', np.zeros((len(body_pose), 15, 3)))  # (T, 15, 3)
            right_hand_pose = npz_data.get('smplh:right_hand_pose', np.zeros((len(body_pose), 15, 3)))  # (T, 15, 3)
            translation = npz_data.get('smplh:translation', np.zeros((len(body_pose), 3)))  # (T, 3)
            
            # 组合所有姿态参数
            pose_data = np.concatenate([
                global_orient.reshape(-1, 3),
                body_pose.reshape(-1, 63),  # 21 * 3
                left_hand_pose.reshape(-1, 45),  # 15 * 3
                right_hand_pose.reshape(-1, 45),  # 15 * 3
            ], axis=1)  # (T, 156)
            
            result['pose'] = torch.from_numpy(pose_data).float()
            result['translation'] = torch.from_numpy(translation).float()
        
        # 处理关键点数据
        if 'boxes_and_keypoints:keypoints' in npz_data:
            keypoints = npz_data['boxes_and_keypoints:keypoints']  # (T, 133, 3)
            result['keypoints'] = torch.from_numpy(keypoints).float()
        
        # 处理情绪和表情特征
        if 'movement:emotion_scores' in npz_data:
            emotion_scores = npz_data['movement:emotion_scores']  # (T, 8)
            result['emotion_scores'] = torch.from_numpy(emotion_scores).float()

        if 'movement:expression' in npz_data:
            expression = npz_data['movement:expression']  # (T, 128)
            result['expression'] = torch.from_numpy(expression).float()
        
        # 加载音频数据（如果需要）
        if self.load_audio and sample_info['wav_path']:
            try:
                audio, sr = librosa.load(sample_info['wav_path'], sr=self.sample_rate)
                result['audio'] = torch.from_numpy(audio).float()
                result['audio_sample_rate'] = sr
            except Exception as e:
                print(f"加载音频失败 {sample_info['wav_path']}: {e}")
                result['audio'] = torch.zeros(1)  # 空音频
                result['audio_sample_rate'] = self.sample_rate
        
        # 加载视频路径（如果需要）
        if self.load_video and sample_info['mp4_path']:
            # 只在主进程且第一次加载时打印日志
            if idx == 0 and hasattr(self, '_debug_print') and self._debug_print:
                print(f"[SeamlessInteractionDataset] 视频路径: {sample_info['mp4_path']}")
            result['video_path'] = sample_info['mp4_path']
        
        # 应用变换
        if self.transform:
            # 只在主进程且第一次加载时打印日志
            if idx == 0 and hasattr(self, '_debug_print') and self._debug_print:
                print(f"[SeamlessInteractionDataset] 开始应用变换")
            result = self.transform(result)
            if idx == 0 and hasattr(self, '_debug_print') and self._debug_print:
                print(f"[SeamlessInteractionDataset] 变换完成")
        
        # 只在主进程且第一次加载时打印日志
        if idx == 0 and hasattr(self, '_debug_print') and self._debug_print:
            print(f"[SeamlessInteractionDataset] 成功构建样本字典，键: {list(result.keys())}")
            
        return result
    
    def get_windows(self, idx: int) -> List[Dict[str, Union[torch.Tensor, np.ndarray]]]:
        """
        获取数据样本的所有时间窗口
        
        Args:
            idx: 样本索引
            
        Returns:
            时间窗口数据列表
        """
        sample = self.__getitem__(idx)
        
        # 获取序列长度
        if 'pose' in sample:
            seq_len = len(sample['pose'])
        elif 'keypoints' in sample:
            seq_len = len(sample['keypoints'])
        elif 'emotion_scores' in sample:
            seq_len = len(sample['emotion_scores'])
        else:
            return [sample]  # 如果没有序列数据，返回整个样本
        
        windows = []
        
        # 生成时间窗口
        for start in range(0, seq_len - self.pose_window_size + 1, self.pose_window_stride):
            end = start + self.pose_window_size
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
            
            windows.append(window_data)
            
        return windows
    
    def get_sample_info(self, idx: int) -> Dict[str, Union[str, int, float]]:
        """
        获取样本的基本信息
        
        Args:
            idx: 样本索引
            
        Returns:
            包含样本信息的字典
        """
        sample_info = self.data_files[idx]
        
        # 获取序列长度
        npz_data = np.load(sample_info['npz_path'])
        seq_len = None
        if 'smplh:body_pose' in npz_data:
            seq_len = npz_data['smplh:body_pose'].shape[0]
        elif 'boxes_and_keypoints:keypoints' in npz_data:
            seq_len = npz_data['boxes_and_keypoints:keypoints'].shape[0]
        
        # 获取音频长度
        audio_len = 0
        if sample_info['wav_path'] and os.path.exists(sample_info['wav_path']):
            try:
                audio, sr = librosa.load(sample_info['wav_path'], sr=None)
                audio_len = len(audio) / sr
            except:
                pass
        
        return {
            'id': sample_info['id'],
            'sequence_length': seq_len,
            'audio_length_seconds': audio_len,
            'has_video': sample_info['mp4_path'] is not None,
            'has_audio': sample_info['wav_path'] is not None,
        }


class SeamlessInteractionWindowDataset(Dataset):
    """
    Seamless_Interaction数据集的时间窗口版本
    每个样本是一个固定长度的时间窗口
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        pose_fps: int = 30,
        audio_fps: int = 16000,
        window_size: float = 2.0,
        window_stride: float = 0.5,
        transform=None,
        load_video: bool = False,
        load_audio: bool = True,
        normalize: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据集根目录路径
            split: 数据集分割 ('train', 'val', 'test')
            sample_rate: 音频采样率
            pose_fps: 姿态帧率
            audio_fps: 音频帧率
            window_size: 时间窗口大小(秒)
            window_stride: 时间窗口步长(秒)
            transform: 数据变换
            load_video: 是否加载视频
            load_audio: 是否加载音频
            normalize: 是否标准化数据
        """
        self.base_dataset = SeamlessInteractionDataset(
            data_path=data_path,
            split=split,
            sample_rate=sample_rate,
            pose_fps=pose_fps,
            audio_fps=audio_fps,
            window_size=window_size,
            window_stride=window_stride,
            transform=transform,
            load_video=load_video,
            load_audio=load_audio,
            normalize=normalize
        )
        
        # 预计算所有窗口
        self.windows = []
        for i in range(len(self.base_dataset)):
            sample_windows = self.base_dataset.get_windows(i)
            self.windows.extend([(i, w) for w in sample_windows])
        
        print(f"创建了 {len(self.windows)} 个时间窗口")
    
    def __len__(self) -> int:
        """返回窗口数量"""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
        """获取单个窗口数据"""
        sample_idx, window = self.windows[idx]
        return window