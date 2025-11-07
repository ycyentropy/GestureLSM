import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


def extract_text_features(transcript_data: List[Dict], 
                         vocab: Optional[Dict] = None, 
                         max_words: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    从转录数据中提取文本特征
    
    Args:
        transcript_data: 转录数据列表，每个元素包含words、start、end和transcript
        vocab: 词汇表字典，用于将单词转换为索引
        max_words: 最大单词数，用于截断或填充
        
    Returns:
        包含文本特征的字典
    """
    if not transcript_data:
        return {
            'word_ids': torch.zeros(0, dtype=torch.long),
            'word_times': torch.zeros(0, 2),
            'word_scores': torch.zeros(0),
            'transcripts': []
        }
    
    # 提取所有单词及其时间戳和分数
    all_words = []
    all_times = []
    all_scores = []
    all_transcripts = []
    
    for segment in transcript_data:
        words = segment.get('words', [])
        segment_transcript = segment.get('transcript', '')
        
        for word_info in words:
            word = word_info.get('word', '').lower()
            start = word_info.get('start', 0.0)
            end = word_info.get('end', 0.0)
            score = word_info.get('score', 1.0)
            
            all_words.append(word)
            all_times.append([start, end])
            all_scores.append(score)
        
        if segment_transcript:
            all_transcripts.append(segment_transcript)
    
    # 转换为张量
    word_times = torch.tensor(all_times, dtype=torch.float32)
    word_scores = torch.tensor(all_scores, dtype=torch.float32)
    
    # 转换单词为ID（如果提供了词汇表）
    if vocab:
        word_ids = [vocab.get(word, vocab.get('<unk>', 0)) for word in all_words]
        word_ids = torch.tensor(word_ids, dtype=torch.long)
    else:
        # 创建简单的词汇映射
        unique_words = list(set(all_words))
        word_to_id = {word: i for i, word in enumerate(unique_words)}
        word_ids = torch.tensor([word_to_id[word] for word in all_words], dtype=torch.long)
    
    # 截断或填充到最大长度
    if max_words is not None and len(word_ids) > max_words:
        word_ids = word_ids[:max_words]
        word_times = word_times[:max_words]
        word_scores = word_scores[:max_words]
    
    return {
        'word_ids': word_ids,
        'word_times': word_times,
        'word_scores': word_scores,
        'transcripts': all_transcripts
    }


def extract_vad_features(vad_data: List[Dict], 
                         sample_rate: float = 30.0, 
                         max_frames: Optional[int] = None) -> torch.Tensor:
    """
    从VAD数据中提取语音活动特征
    
    Args:
        vad_data: VAD数据列表，每个元素包含start和end
        sample_rate: 采样率（帧/秒）
        max_frames: 最大帧数，用于截断或填充
        
    Returns:
        语音活动张量，1表示语音活动，0表示静音
    """
    if not vad_data:
        return torch.zeros(0, dtype=torch.float32)
    
    # 确定总时长
    total_time = max(segment.get('end', 0) for segment in vad_data)
    total_frames = int(total_time * sample_rate)
    
    # 创建VAD张量
    vad_tensor = torch.zeros(total_frames, dtype=torch.float32)
    
    # 填充VAD数据
    for segment in vad_data:
        start = segment.get('start', 0.0)
        end = segment.get('end', 0.0)
        
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        
        # 确保索引在有效范围内
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))
        
        if start_frame < end_frame:
            vad_tensor[start_frame:end_frame] = 1.0
    
    # 截断或填充到最大长度
    if max_frames is not None:
        if len(vad_tensor) > max_frames:
            vad_tensor = vad_tensor[:max_frames]
        elif len(vad_tensor) < max_frames:
            vad_tensor = torch.cat([
                vad_tensor, 
                torch.zeros(max_frames - len(vad_tensor), dtype=torch.float32)
            ])
    
    return vad_tensor


def extract_pose_features(pose_data: torch.Tensor, 
                         joint_mask: Optional[torch.Tensor] = None,
                         normalize: bool = True) -> torch.Tensor:
    """
    从SMPL-H姿态数据中提取特征
    
    Args:
        pose_data: SMPL-H姿态参数 (T, 156)
        joint_mask: 关节掩码，用于选择特定关节
        normalize: 是否标准化数据
        
    Returns:
        处理后的姿态特征
    """
    # 确保输入是正确的形状
    if pose_data.dim() != 2 or pose_data.size(1) != 156:
        raise ValueError(f"姿态数据形状应为(T, 156)，但得到 {pose_data.shape}")
    
    # 分解姿态参数
    global_orient = pose_data[:, :3]  # 全局方向
    body_pose = pose_data[:, 3:66]  # 身体姿态 (21 * 3)
    left_hand_pose = pose_data[:, 66:111]  # 左手姿态 (15 * 3)
    right_hand_pose = pose_data[:, 111:156]  # 右手姿态 (15 * 3)
    
    # 应用关节掩码（如果提供）
    if joint_mask is not None:
        if joint_mask.dim() != 1 or joint_mask.size(0) != 156:
            raise ValueError(f"关节掩码形状应为(156,)，但得到 {joint_mask.shape}")
        
        # 重塑掩码以匹配各个部分
        global_orient_mask = joint_mask[:3]
        body_pose_mask = joint_mask[3:66]
        left_hand_mask = joint_mask[66:111]
        right_hand_mask = joint_mask[111:156]
        
        # 应用掩码
        global_orient = global_orient * global_orient_mask.unsqueeze(0)
        body_pose = body_pose * body_pose_mask.unsqueeze(0)
        left_hand_pose = left_hand_pose * left_hand_mask.unsqueeze(0)
        right_hand_pose = right_hand_pose * right_hand_mask.unsqueeze(0)
        
        # 重新组合
        processed_pose = torch.cat([
            global_orient, body_pose, left_hand_pose, right_hand_pose
        ], dim=1)
    else:
        processed_pose = pose_data
    
    # 标准化（如果需要）
    if normalize:
        # 计算均值和标准差（这里使用简单的标准化，实际应用中可能需要预计算的统计量）
        mean = processed_pose.mean(dim=0)
        std = processed_pose.std(dim=0) + 1e-8  # 避免除以零
        processed_pose = (processed_pose - mean) / std
    
    return processed_pose


def extract_keypoint_features(keypoints: torch.Tensor, 
                             normalize: bool = True,
                             confidence_threshold: float = 0.5) -> torch.Tensor:
    """
    从关键点数据中提取特征
    
    Args:
        keypoints: 关键点数据 (T, 133, 3)
        normalize: 是否标准化数据
        confidence_threshold: 置信度阈值，低于此值的关键点将被过滤
        
    Returns:
        处理后的关键点特征
    """
    # 确保输入是正确的形状
    if keypoints.dim() != 3 or keypoints.size(2) != 3:
        raise ValueError(f"关键点数据形状应为(T, 133, 3)，但得到 {keypoints.shape}")
    
    # 分离坐标和置信度
    coords = keypoints[:, :, :2]  # (T, 133, 2)
    confidence = keypoints[:, :, 2]  # (T, 133)
    
    # 应用置信度阈值
    mask = confidence > confidence_threshold
    mask = mask.unsqueeze(-1).expand_as(coords)
    coords = coords * mask.float()
    
    # 展平关键点
    flattened = coords.view(coords.size(0), -1)  # (T, 266)
    
    # 标准化（如果需要）
    if normalize:
        # 计算均值和标准差
        mean = flattened.mean(dim=0)
        std = flattened.std(dim=0) + 1e-8  # 避免除以零
        flattened = (flattened - mean) / std
    
    return flattened


def extract_emotion_features(emotion_scores: torch.Tensor, 
                           normalize: bool = True) -> torch.Tensor:
    """
    从情绪分数中提取特征
    
    Args:
        emotion_scores: 情绪分数 (T, 8)
        normalize: 是否标准化数据
        
    Returns:
        处理后的情绪特征
    """
    # 确保输入是正确的形状
    if emotion_scores.dim() != 2 or emotion_scores.size(1) != 8:
        raise ValueError(f"情绪分数形状应为(T, 8)，但得到 {emotion_scores.shape}")
    
    processed_emotions = emotion_scores
    
    # 标准化（如果需要）
    if normalize:
        # 计算均值和标准差
        mean = processed_emotions.mean(dim=0)
        std = processed_emotions.std(dim=0) + 1e-8  # 避免除以零
        processed_emotions = (processed_emotions - mean) / std
    
    return processed_emotions


def extract_expression_features(expression: torch.Tensor, 
                               normalize: bool = True) -> torch.Tensor:
    """
    从表情特征中提取特征
    
    Args:
        expression: 表情特征 (T, 128)
        normalize: 是否标准化数据
        
    Returns:
        处理后的表情特征
    """
    # 确保输入是正确的形状
    if expression.dim() != 2 or expression.size(1) != 128:
        raise ValueError(f"表情特征形状应为(T, 128)，但得到 {expression.shape}")
    
    processed_expression = expression
    
    # 标准化（如果需要）
    if normalize:
        # 计算均值和标准差
        mean = processed_expression.mean(dim=0)
        std = processed_expression.std(dim=0) + 1e-8  # 避免除以零
        processed_expression = (processed_expression - mean) / std
    
    return processed_expression


def align_audio_to_motion(audio: torch.Tensor, 
                          motion_length: int, 
                          audio_fps: int = 16000, 
                          motion_fps: int = 30) -> torch.Tensor:
    """
    将音频数据与运动数据对齐
    
    Args:
        audio: 音频数据
        motion_length: 运动数据长度（帧数）
        audio_fps: 音频采样率
        motion_fps: 运动帧率
        
    Returns:
        对齐后的音频数据
    """
    # 计算所需的音频长度
    target_audio_length = int(motion_length * audio_fps / motion_fps)
    
    # 如果音频太短，填充零
    if len(audio) < target_audio_length:
        padding = target_audio_length - len(audio)
        audio = torch.cat([audio, torch.zeros(padding, dtype=audio.dtype)])
    # 如果音频太长，截断
    elif len(audio) > target_audio_length:
        audio = audio[:target_audio_length]
    
    return audio


def create_joint_mask(joint_names: List[str], 
                     all_joint_names: List[str]) -> torch.Tensor:
    """
    创建关节掩码，用于选择特定关节
    
    Args:
        joint_names: 要选择的关节名称列表
        all_joint_names: 所有关节名称列表
        
    Returns:
        关节掩码张量
    """
    # 创建映射字典
    joint_to_idx = {name: idx for idx, name in enumerate(all_joint_names)}
    
    # 创建掩码
    mask = torch.zeros(len(all_joint_names), dtype=torch.float32)
    
    # 设置选中的关节
    for name in joint_names:
        if name in joint_to_idx:
            mask[joint_to_idx[name]] = 1.0
        else:
            print(f"警告: 关节 '{name}' 不在可用关节列表中")
    
    return mask


def extract_audio_features(audio: torch.Tensor, 
                          sample_rate: int = 16000, 
                          n_mels: int = 80,
                          n_fft: int = 1024,
                          hop_length: int = 256) -> torch.Tensor:
    """
    从音频数据中提取梅尔频谱图特征
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        n_mels: 梅尔滤波器数量
        n_fft: FFT窗口大小
        hop_length: 跳跃长度
        
    Returns:
        梅尔频谱图特征
    """
    import torch.nn.functional as F
    
    # 确保输入是一维的
    if audio.dim() > 1:
        audio = audio.squeeze()
    
    # 创建梅尔频谱图
    # 这里使用简单的实现，实际应用中可能需要更复杂的特征提取
    # 例如使用librosa或torchaudio
    
    # 计算短时傅里叶变换
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitude = torch.abs(stft)
    
    # 创建梅尔滤波器（简化版）
    # 实际应用中应使用标准实现
    mel_spec = magnitude[:n_mels, :]
    
    # 转换为对数刻度
    log_mel_spec = torch.log1p(mel_spec)
    
    return log_mel_spec