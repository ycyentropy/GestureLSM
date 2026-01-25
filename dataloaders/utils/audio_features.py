"""modified from https://github.com/yesheng-THU/GFGE/blob/main/data_processing/audio_features.py"""
import numpy as np
import librosa
import math
import os
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from typing import Optional, Tuple
from numpy.lib import stride_tricks
from loguru import logger



def process_audio_data(audio_file, args, data, f_name, selected_file):
    """Process audio data with support for different representations."""
    logger.info(f"# ---- Building cache for Audio {f_name} ---- #")
    
    if not os.path.exists(audio_file):
        logger.warning(f"# ---- file not found for Audio {f_name}, skip all files with the same id ---- #")
        selected_file.drop(selected_file[selected_file['id'] == f_name].index, inplace=True)
        return None

    if args.audio_rep == "onset+amplitude":
        # data['audio'] = calculate_onset_amplitude(audio_file, args.audio_sr, None)
        data['audio'] = calculate_onset_amplitude_new(audio_file, args.audio_sr)
        
    elif args.audio_rep == "mfcc":
        audio_data, _ = librosa.load(audio_file, sr=args.audio_sr)
        data['audio'] = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=args.audio_sr, 
            n_mels=128, 
            hop_length=int(args.audio_sr/args.audio_fps)
        ).transpose(1, 0)
    
    if args.audio_norm and args.audio_rep == "wave16k":
        data['audio'] = (data['audio'] - args.mean_audio) / args.std_audio
    
    return data

def calculate_onset_amplitude(audio_file, audio_sr, save_path):
    """Calculate onset and amplitude features from audio file.
    
    Args:
        audio_file: Path to audio file
        audio_sr: Target sample rate
        save_path: Path to save features (optional)
        normalize: Whether to normalize amplitude envelope to 0-1 range
    """
    audio_data, sr = librosa.load(audio_file)
    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=audio_sr)
    
    # Calculate amplitude envelope
    frame_length = 1024
    shape = (audio_data.shape[-1] - frame_length + 1, frame_length)
    strides = (audio_data.strides[-1], audio_data.strides[-1])
    rolling_view = stride_tricks.as_strided(audio_data, shape=shape, strides=strides)
    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
    
    # # Normalize amplitude envelope to 0-1 range if requested 新加入的归一化
    # if normalize:
    #     max_amplitude = np.max(amplitude_envelope)
    #     if max_amplitude > 0:
    #         amplitude_envelope = amplitude_envelope / max_amplitude
    
    # Calculate onset
    audio_onset_f = librosa.onset.onset_detect(y=audio_data, sr=audio_sr, units='frames')
    onset_array = np.zeros(len(audio_data), dtype=float)
    onset_array[audio_onset_f] = 1.0
    
    # Combine features
    features = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
    
    return features

def calculate_onset_amplitude_new(
    audio_file: str,
    target_sr: int = 16000,
    n_fft: int = 1024,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    从16000Hz WAV音频文件中提取样本级的振幅包络和起音强度特征（彻底修复参数报错）
    
    Args:
        audio_file: WAV音频文件路径
        target_sr: 目标采样率（默认16000Hz）
        n_fft: 帧长（用于计算起音强度，默认1024）
        save_path: 特征保存路径（None则不保存）
    
    Returns:
        features: 二维特征数组，shape=(样本数, 2)，列依次为：振幅包络（0~1）、起音强度（0~1）
    
    Raises:
        FileNotFoundError: 音频文件不存在
        ValueError: 音频数据无效（如长度过短、格式错误）
        Exception: 其他音频加载/处理异常
    """
    # ========== 1. 异常处理与音频加载 ==========
    try:
        audio_data, orig_sr = librosa.load(
            audio_file,
            sr=target_sr,
            mono=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"音频文件不存在：{audio_file}")
    except Exception as e:
        raise Exception(f"音频加载失败：{str(e)}")
    
    if len(audio_data) < n_fft:
        raise ValueError(f"音频过短（{len(audio_data)}样本），至少需要{n_fft}样本")
    
    # ========== 2. 音频预处理 ==========
    audio_data = audio_data - np.mean(audio_data)  # 去直流
    audio_data = librosa.util.normalize(audio_data)  # 归一化
    
    # ========== 3. 振幅包络（样本级） ==========
    frames = librosa.util.frame(
        audio_data,
        frame_length=n_fft,
        hop_length=1         # 逐样本滑动
    )
    amplitude_envelope = np.max(np.abs(frames), axis=0)
    # 归一化
    amp_max = np.max(amplitude_envelope)
    if amp_max > 0:
        amplitude_envelope = amplitude_envelope / amp_max
    # 补全长度
    if len(amplitude_envelope) < len(audio_data):
        pad_length = len(audio_data) - len(amplitude_envelope)
        amplitude_envelope = np.pad(
            amplitude_envelope,
            (0, pad_length),
            mode='constant',
            constant_values=amplitude_envelope[-1]
        )
    
    # ========== 4. 起音特征（彻底修复参数错误） ==========
    hop_length = 512  # 16000Hz下≈23ms
    # 步骤1：先计算起音强度包络（仅这里指定n_fft）
    onset_strength_frame = librosa.onset.onset_strength(
        y=audio_data,
        sr=target_sr,
        n_fft=n_fft,        # n_fft仅在这里指定（起音强度计算需要）
        hop_length=hop_length
    )
    # 归一化
    onset_max = np.max(onset_strength_frame)
    if onset_max > 0:
        onset_strength_frame = onset_strength_frame / onset_max
    
    # 步骤2：提取起音峰值（仅传onset_detect支持的参数）
    # 先获取帧级起音索引，再转换为样本级（替代直接传units='samples'）
    onset_frames_idx = librosa.onset.onset_detect(
        y=audio_data,
        sr=target_sr,
        hop_length=hop_length,  # 仅传支持的参数：y/sr/hop_length
        # 移除n_fft！该参数不被onset_detect支持
    )
    # 将帧级索引转换为样本级索引（核心修复：手动转换，避免传无效参数）
    onset_frames = librosa.frames_to_samples(onset_frames_idx, hop_length=hop_length)
    
    # 样本级起音强度填充
    onset_strength_sample = np.zeros(len(audio_data), dtype=float)
    for i, frame_idx in enumerate(onset_frames):
        if i < len(onset_strength_frame):
            start = frame_idx
            end = onset_frames[i+1] if i+1 < len(onset_frames) else len(audio_data)
            onset_strength_sample[start:end] = onset_strength_frame[i]
    
    # ========== 5. 特征合并 ==========
    assert len(amplitude_envelope) == len(onset_strength_sample) == len(audio_data), \
        "特征长度不匹配"
    
    features = np.concatenate([
        amplitude_envelope.reshape(-1, 1),
        onset_strength_sample.reshape(-1, 1)
    ], axis=1)
    
    # ========== 6. 保存特征 ==========
    if save_path is not None:
        np.savez(save_path, features=features, target_sr=target_sr)
    
    return features
