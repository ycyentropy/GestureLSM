import os
import json
import pickle
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from loguru import logger
from collections import defaultdict
import lmdb
import shutil
from tqdm import tqdm
import time
import librosa
import math

# Import Vocab class for text encoding
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.build_vocab import Vocab


class BidirectionalDyadicDataset(Dataset):
    """
    双向生成的Dyadic数据集
    同时返回Speaker和Listener的数据，用于同时生成两者的手势
    支持LMDB缓存和完整的数据过滤功能
    从WAV文件实时提取音频特征
    优化：参考seamless_interaction_dataset的数据处理流程
    """
    
    def __init__(self, args, split='train', build_cache=True):
        """
        Args:
            args: 配置参数
            split: 'train' 或 'val'
            build_cache: 是否构建缓存
        """
        self.args = args
        self.split = split
        self.data_path = args.data_path
        
        # 时间窗口参数
        self.horizon = getattr(args, 'horizon', 144)  # h
        self.stride = getattr(args, 'stride', 60)
        self.min_speak_duration = getattr(args, 'min_speak_duration', 5.0)  # 判定Speaker的最小说话时长
        
        # 姿态参数
        self.pose_fps = getattr(args, 'pose_fps', 30)
        self.audio_fps = getattr(args, 'audio_fps', 30)
        self.audio_sr = getattr(args, 'audio_sr', 48000)
        
        # 音频特征类型
        self.audio_rep = getattr(args, 'audio_rep', 'mfcc')  # mfcc, onset+amplitude
        
        # 音频归一化
        self.mean_audio = getattr(args, 'mean_audio', 0.0)
        self.std_audio = getattr(args, 'std_audio', 1.0)
        self.audio_norm = getattr(args, 'audio_norm', False)
        
        # 数据过滤参数
        self.training_speakers = getattr(args, 'training_speakers', None)
        self.disable_filtering = getattr(args, 'disable_filtering', False)
        self.clean_first_seconds = getattr(args, 'clean_first_seconds', 0)
        self.clean_final_seconds = getattr(args, 'clean_final_seconds', 0)
        self.skip_segmentation = getattr(args, 'skip_segmentation', False)
        
        # LMDB缓存路径
        self.cache_dir = getattr(args, 'cache_dir', 'cache/bidirectional_dyadic')
        self.lmdb_path = os.path.join(self.cache_dir, split)
        self.use_lmdb = getattr(args, 'use_lmdb', True)
        self.lmdb_env = None
        
        # 加载词汇表用于文本编码
        self.vocab_path = getattr(args, 'vocab_path', 'datasets/unified_vocab.pkl')
        self.lang_model = None
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, 'rb') as f:
                    self.lang_model = pickle.load(f)
                logger.info(f"✓ Language model loaded from {self.vocab_path}")
            except Exception as e:
                logger.warning(f"Failed to load language model: {e}")
        else:
            logger.warning(f"Vocab file not found: {self.vocab_path}")
        
        # 1. 扫描并配对文件（带ID过滤）
        logger.info(f"[{split}] Scanning files from {self.data_path}...")
        self.pairs = self._scan_and_pair_files()
        logger.info(f"[{split}] Found {len(self.pairs)} interaction pairs after filtering")
        
        # 2. 构建或加载LMDB缓存
        if self.use_lmdb:
            self._init_cache(build_cache)
        else:
            # 不使用时，预计算有效样本
            logger.info(f"[{split}] Building valid segment index...")
            self.valid_segments = self._build_valid_segments()
            logger.info(f"[{split}] Indexed {len(self.valid_segments)} valid segments")
    
    def _init_cache(self, build_cache):
        """初始化或构建LMDB缓存"""
        self.mapping_data = None
        
        if build_cache:
            self._build_lmdb_cache()
        
        # 加载缓存映射
        self._load_lmdb_mapping()
    
    def _build_lmdb_cache(self):
        """
        构建LMDB缓存 - 参考seamless_interaction_dataset
        存储完整序列，不预切片，不存储seed
        """
        logger.info(f"Building LMDB cache at {self.lmdb_path}...")
        
        # 清理旧缓存（如果需要）
        if getattr(self.args, 'new_cache', False) and os.path.exists(self.lmdb_path):
            shutil.rmtree(self.lmdb_path)
            logger.info(f"Removed old cache: {self.lmdb_path}")
        
        # 如果缓存已存在，跳过构建
        if os.path.exists(self.lmdb_path) and os.path.exists(os.path.join(self.lmdb_path, "mapping.pkl")):
            logger.info(f"Found existing cache at {self.lmdb_path}")
            return
        
        # 创建缓存目录
        os.makedirs(self.lmdb_path, exist_ok=True)
        
        # 构建有效片段索引（带分段过滤）
        logger.info("Building valid segment index with filtering...")
        valid_segments = self._build_valid_segments()
        logger.info(f"Found {len(valid_segments)} valid segments after filtering")
        
        # 创建LMDB环境
        map_size = 100 * 1024 * 1024 * 1024  # 100GB
        lmdb_env = lmdb.open(self.lmdb_path, map_size=map_size)
        
        # 构建缓存
        logger.info("Writing segments to LMDB...")
        mapping = []
        sample_idx = 0
        n_skipped = 0
        
        with lmdb_env.begin(write=True) as txn:
            for segment_info in tqdm(valid_segments, desc="Processing segments"):
                segment_data = self._load_segment_data(segment_info)
                
                if segment_data is None:
                    n_skipped += 1
                    continue
                
                # 序列化并存储
                key = f"{sample_idx:08d}".encode('ascii')
                value = pickle.dumps(segment_data)
                txn.put(key, value)
                
                mapping.append(sample_idx)
                sample_idx += 1
        
        lmdb_env.close()
        
        # 保存映射文件
        mapping_data = {
            'mapping': mapping,
            'n_samples': len(mapping),
            'pairs': self.pairs,
            'valid_segments': valid_segments
        }
        
        with open(os.path.join(self.lmdb_path, "mapping.pkl"), 'wb') as f:
            pickle.dump(mapping_data, f)
        
        logger.info(f"Cache built successfully: {len(mapping)} segments ({n_skipped} skipped)")
    
    def _load_lmdb_mapping(self):
        """加载LMDB映射"""
        mapping_file = os.path.join(self.lmdb_path, "mapping.pkl")
        
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        with open(mapping_file, 'rb') as f:
            self.mapping_data = pickle.load(f)
        
        self.n_samples = self.mapping_data['n_samples']
        self.valid_segments = self.mapping_data['valid_segments']
        
        logger.info(f"Loaded LMDB mapping: {self.n_samples} segments")
    
    def _get_lmdb_env(self):
        """获取LMDB环境（延迟加载）"""
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        return self.lmdb_env
    
    def _scan_and_pair_files(self):
        """
        扫描文件并按(V, S, I)三元组配对（带ID过滤）
        
        Returns:
            pairs: List of [(p1_path, p1_json), (p2_path, p2_json)]
        """
        # 递归扫描所有npz文件
        pattern = os.path.join(self.data_path, "**", "*.npz")
        all_npz_files = sorted(glob(pattern, recursive=True))
        
        logger.info(f"Total NPZ files found: {len(all_npz_files)}")
        
        # ID过滤
        if not self.disable_filtering and self.training_speakers:
            filtered_files = []
            for npz_file in all_npz_files:
                # 从文件名提取ID: V00_S0039_I00000581_P0061A.npz -> 61
                speaker_id_match = re.search(r'P(\d+)[A-Za-z]*\.npz$', npz_file)
                if speaker_id_match:
                    file_id = int(speaker_id_match.group(1))
                    if file_id in self.training_speakers:
                        filtered_files.append(npz_file)
            
            logger.info(f"Filtered {len(filtered_files)} files based on training_speakers: {self.training_speakers}")
            
            if not filtered_files:
                logger.warning(f"No files match training_speakers, using all files")
                filtered_files = all_npz_files
            
            all_npz_files = filtered_files
        else:
            logger.info("ID filtering disabled or no training_speakers specified")
        
        # 按(V, S, I)分组
        groups = defaultdict(list)
        for npz_path in all_npz_files:
            fname = os.path.basename(npz_path)
            # 解析文件名: V00_S0039_I00000581_P0061A.npz
            parts = fname.replace('.npz', '').split('_')
            if len(parts) >= 4:
                key = "_".join(parts[:3])  # V00_S0039_I00000581
                json_path = npz_path.replace('.npz', '.json')
                if os.path.exists(json_path):
                    groups[key].append((npz_path, json_path))
        
        # 构建配对 (只保留有2个参与者的组)
        pairs = []
        for key, files in groups.items():
            if len(files) == 2:
                # 按participant_id排序保证顺序一致
                files.sort(key=lambda x: os.path.basename(x[0]).split('_')[-1])
                pairs.append(tuple(files))
            elif len(files) > 2:
                logger.warning(f"Interaction {key} has {len(files)} participants, taking first 2")
                files.sort(key=lambda x: os.path.basename(x[0]).split('_')[-1])
                pairs.append(tuple(files[:2]))
        
        return pairs
    
    def _build_valid_segments(self):
        """
        构建有效片段索引（参考seamless_interaction_dataset）
        返回完整片段，不预切片
        
        Returns:
            valid_segments: List of (pair_idx, start_frame, end_frame, speaker_idx, p1_word_data, p2_word_data)
        """
        valid_segments = []
        n_filtered = defaultdict(int)
        
        for pair_idx, ((p1_npz, p1_json), (p2_npz, p2_json)) in enumerate(tqdm(self.pairs, desc="Processing pairs")):
            # 加载并处理JSON数据（包含文本编码）
            p1_processed = self._load_and_process_json(p1_json)
            p2_processed = self._load_and_process_json(p2_json)
            
            if p1_processed is None or p2_processed is None:
                n_filtered['json_load_error'] += 1
                continue
            
            p1_utterances, p1_word_data = p1_processed
            p2_utterances, p2_word_data = p2_processed
            
            # 加载姿态数据
            p1_pose_data = self._load_seamless_motion_data(p1_npz)
            p2_pose_data = self._load_seamless_motion_data(p2_npz)
            
            if p1_pose_data is None or p2_pose_data is None:
                n_filtered['load_error'] += 1
                continue
            
            p1_pose = p1_pose_data['pose']
            p2_pose = p2_pose_data['pose']
            p1_motion_valid = p1_pose_data['motion_valid']
            p2_motion_valid = p2_pose_data['motion_valid']
            
            # 确保word数据长度与姿态一致
            if len(p1_word_data) != len(p1_pose):
                p1_word_data = self._align_word_data(p1_word_data, len(p1_pose))
            if len(p2_word_data) != len(p2_pose):
                p2_word_data = self._align_word_data(p2_word_data, len(p2_pose))
            
            # 计算有效性交集
            p1_final_valid = p1_motion_valid & (np.ones(len(p1_word_data), dtype=bool) if p1_word_data is not None else p1_motion_valid)
            p2_final_valid = p2_motion_valid & (np.ones(len(p2_word_data), dtype=bool) if p2_word_data is not None else p2_motion_valid)
            
            # 计算清理后的时间边界
            total_seconds = len(p1_pose) / self.pose_fps
            clip_start_t = self.clean_first_seconds
            clip_end_t = total_seconds - self.clean_final_seconds
            
            if clip_start_t >= clip_end_t:
                n_filtered['too_short_after_cleaning'] += 1
                continue
            
            # 转换为帧索引
            clip_start_frame = int(clip_start_t * self.pose_fps)
            clip_end_frame = int(clip_end_t * self.pose_fps)
            
            # 确保片段足够长
            min_segment_length = self.horizon * 2  # 至少2个horizon长度
            if clip_end_frame - clip_start_frame < min_segment_length:
                n_filtered['segment_too_short'] += 1
                continue
            
            # 查找连续有效片段
            p1_valid_segments = self.find_continuous_valid_segments(p1_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
            p2_valid_segments = self.find_continuous_valid_segments(p2_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
            
            # 为每个满足条件的区间创建独立的数据副本
            for p1_seg in p1_valid_segments:
                p1_seg_start, p1_seg_end = p1_seg
                # 转换为绝对帧索引
                p1_abs_start = clip_start_frame + p1_seg_start
                p1_abs_end = clip_start_frame + p1_seg_end
                
                # 检查整个片段的Speaker/Listener角色
                # 使用片段中间点判断角色
                mid_frame = (p1_abs_start + p1_abs_end) // 2
                mid_time = mid_frame / self.pose_fps
                window_duration = 10.0  # 用10秒窗口判断角色
                
                p1_is_speaker = self._check_is_speaker(p1_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
                p2_is_speaker = self._check_is_speaker(p2_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
                
                # 只有一个是Speaker时才是有效片段
                if p1_is_speaker and not p2_is_speaker:
                    valid_segments.append((pair_idx, p1_abs_start, p1_abs_end, 0, p1_word_data[p1_abs_start:p1_abs_end], p2_word_data[p1_abs_start:p1_abs_end]))
                elif p2_is_speaker and not p1_is_speaker:
                    valid_segments.append((pair_idx, p1_abs_start, p1_abs_end, 1, p2_word_data[p1_abs_start:p1_abs_end], p1_word_data[p1_abs_start:p1_abs_end]))
                else:
                    n_filtered['no_clear_speaker'] += 1
        
        # 打印过滤统计
        logger.info("Filtering statistics:")
        for key, count in n_filtered.items():
            logger.info(f"  {key}: {count}")
        
        return valid_segments
    
    def _load_and_process_json(self, json_path):
        """
        加载JSON文件并处理文本编码
        
        Returns:
            (utterances, word_data): 说话信息和文本编码结果
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 提取utterances（用于Speaker判定）
            if 'metadata:transcript' in json_data:
                utterances = json_data['metadata:transcript']
            elif 'utterances' in json_data:
                utterances = json_data['utterances']
            else:
                utterances = []
            
            # 提取word数据和transcript时间范围（用于文本编码）
            all_words = []
            transcript_ranges = []
            for transcript in json_data.get('metadata:transcript', []):
                transcript_start = transcript.get('start')
                transcript_end = transcript.get('end')
                if transcript_start is not None and transcript_end is not None:
                    transcript_ranges.append((transcript_start, transcript_end))
                all_words.extend(transcript.get('words', []))
            
            # 按开始时间排序
            all_words.sort(key=lambda x: x.get('start') if x.get('start') is not None else 0)
            transcript_ranges.sort(key=lambda x: x[0])
            
            # 处理文本编码
            # 先估计帧数（后面会对齐）
            if transcript_ranges:
                max_time = max(tr[1] for tr in transcript_ranges)
                estimated_frames = int(max_time * self.pose_fps) + 1
            else:
                estimated_frames = 1000  # 默认值
            
            word_data = self._process_word_encoding(all_words, transcript_ranges, estimated_frames)
            
            return utterances, np.array(word_data, dtype=np.int64)
            
        except Exception as e:
            logger.warning(f"Failed to load and process {json_path}: {e}")
            return None
    
    def _align_word_data(self, word_data, target_frames):
        """对齐word数据到目标帧数"""
        current_len = len(word_data)
        if current_len == target_frames:
            return word_data
        elif current_len < target_frames:
            # 填充PAD token
            pad_length = target_frames - current_len
            return np.concatenate([word_data, np.full(pad_length, self.lang_model.PAD_token if self.lang_model else 0, dtype=np.int64)])
        else:
            # 截断
            return word_data[:target_frames]
    
    def _check_is_speaker(self, utterances, window_start, window_end):
        """
        检查在指定时间窗口内是否为Speaker
        规则：必须有单次说话时长 > min_speak_duration (5秒)
        
        Args:
            utterances: List of {'start': float, 'end': float, ...}
            window_start: 窗口开始时间(秒)
            window_end: 窗口结束时间(秒)
        
        Returns:
            bool: 是否为Speaker
        """
        for utt in utterances:
            utt_start = utt.get('start', 0)
            utt_end = utt.get('end', 0)
            
            # 检查utterance是否与窗口重叠
            overlap_start = max(window_start, utt_start)
            overlap_end = min(window_end, utt_end)
            
            if overlap_start < overlap_end:
                # 计算utterance持续时间
                duration = utt_end - utt_start
                if duration > self.min_speak_duration:
                    return True
        
        return False
    
    def _process_word_encoding(self, word_list, transcript_ranges, num_frames):
        """
        处理文本编码，将单词映射到词汇表索引
        参考: seamless_text_features.py 中的 process_basic_encoding
        """
        word_data = []
        
        # 如果没有语言模型，使用PAD token填充
        if self.lang_model is None:
            return [0] * num_frames
        
        # 双指针技术
        transcript_idx = 0
        word_idx = 0
        num_transcripts = len(transcript_ranges)
        num_words = len(word_list)
        
        # 时间扩展参数（覆盖小间隙）
        time_extension = 0.1  # 100ms
        
        # 遍历每个帧
        for frame_idx in range(num_frames):
            # 计算当前时间（秒）
            current_time = frame_idx / self.pose_fps
            
            # 检查当前时间是否在transcript范围内
            in_transcript = False
            while transcript_idx < num_transcripts and current_time > transcript_ranges[transcript_idx][1]:
                transcript_idx += 1
            
            if transcript_idx < num_transcripts and transcript_ranges[transcript_idx][0] <= current_time <= transcript_ranges[transcript_idx][1]:
                in_transcript = True
            
            # 查找对应的单词
            found_word = False
            while word_idx < num_words:
                start_time = word_list[word_idx].get('start')
                end_time = word_list[word_idx].get('end')
                
                # 跳过无效时间
                if start_time is None or end_time is None:
                    word_idx += 1
                    continue
                
                # 如果当前时间在此单词之前，停止查找
                if current_time < start_time - time_extension:
                    break
                
                # 检查当前时间是否在单词时间范围内（带扩展）
                if start_time - time_extension <= current_time <= end_time + time_extension:
                    word = word_list[word_idx].get('word', '')
                    if word.strip():
                        word_data.append(self.lang_model.get_word_index(word))
                    else:
                        word_data.append(self.lang_model.PAD_token)
                    found_word = True
                    break
                else:
                    # 当前时间在此单词之后，移动到下一个单词
                    word_idx += 1
            
            # 如果没有找到单词，使用PAD token
            if not found_word:
                word_data.append(self.lang_model.PAD_token)
        
        return word_data
    
    def _extract_audio_features(self, wav_path, target_length=None):
        """
        从WAV文件提取音频特征
        
        Args:
            wav_path: WAV文件路径
            target_length: 目标长度（帧数），如果指定则进行填充或截断
        
        Returns:
            audio_features: [N, audio_dim] 音频特征
        """
        if not os.path.exists(wav_path):
            logger.warning(f"WAV file not found: {wav_path}")
            return None
        
        try:
            if self.audio_rep == "mfcc":
                # 提取MFCC特征
                audio_data, _ = librosa.load(wav_path, sr=self.audio_sr)
                
                # 提取梅尔频谱并转换为对数刻度 (log-mel spectrogram)
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=self.audio_sr,
                    n_mels=128,
                    hop_length=int(self.audio_sr / self.audio_fps)
                )
                
                # 转换为对数刻度，降低数值范围
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                log_mel_spec = log_mel_spec.transpose(1, 0)  # [time, n_mels]
                
                # 如果指定了目标长度，进行填充或截断
                if target_length is not None and len(log_mel_spec) < target_length:
                    # 用最后一帧填充
                    pad_length = target_length - len(log_mel_spec)
                    last_frame = log_mel_spec[-1:]  # [1, n_mels]
                    padding = np.tile(last_frame, (pad_length, 1))  # [pad_length, n_mels]
                    log_mel_spec = np.concatenate([log_mel_spec, padding], axis=0)
                elif target_length is not None and len(log_mel_spec) > target_length:
                    # 截断到目标长度
                    log_mel_spec = log_mel_spec[:target_length]
                
                # 归一化到 [-1, 1] 范围
                # log_mel_spec 的范围通常是 [-80, 0] dB
                log_mel_spec = (log_mel_spec + 80.0) / 80.0 * 2.0 - 1.0  # 映射到 [-1, 1]
                
                # 可选的额外归一化
                if self.audio_norm:
                    log_mel_spec = (log_mel_spec - self.mean_audio) / (self.std_audio + 1e-8)
                
                return log_mel_spec.astype(np.float32)
            
            elif self.audio_rep == "onset+amplitude":
                # 提取起音和振幅特征
                audio_data, sr = librosa.load(wav_path)
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.audio_sr)
                
                # 计算振幅包络
                frame_length = 1024
                amplitude_envelope = self._calculate_amplitude_envelope(audio_data, frame_length)
                
                # 计算起音
                onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.audio_sr, units='frames')
                onset_array = np.zeros(len(audio_data), dtype=float)
                onset_array[onset_frames] = 1.0
                
                # 组合特征
                features = np.concatenate([
                    amplitude_envelope.reshape(-1, 1),
                    onset_array.reshape(-1, 1)
                ], axis=1).astype(np.float32)
                
                # 如果指定了目标长度，进行填充或截断
                if target_length is not None and len(features) < target_length:
                    # 用最后一行填充
                    pad_length = target_length - len(features)
                    last_row = features[-1:]  # [1, 2]
                    padding = np.tile(last_row, (pad_length, 1))  # [pad_length, 2]
                    features = np.concatenate([features, padding], axis=0)
                elif target_length is not None and len(features) > target_length:
                    # 截断到目标长度
                    features = features[:target_length]
                
                return features
            
            else:
                logger.warning(f"Unsupported audio_rep: {self.audio_rep}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio features from {wav_path}: {e}")
            return None
    
    def _calculate_amplitude_envelope(self, audio_data, frame_length):
        """计算振幅包络"""
        from numpy.lib import stride_tricks
        
        shape = (audio_data.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_data.strides[-1], audio_data.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_data, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = np.pad(
            amplitude_envelope,
            (0, frame_length - 1),
            mode='constant',
            constant_values=amplitude_envelope[-1]
        )
        return amplitude_envelope
    
    def _load_seamless_motion_data(self, pose_file):
        """
        加载并处理姿态数据（参考seamless_interaction_dataset）
        """
        try:
            # 加载npz文件
            pose_data = np.load(pose_file, allow_pickle=True)
            
            # 提取姿态数据
            trans = pose_data['smplh:translation']
            
            # 将平移数据从厘米转换为米
            trans = trans / 100.0
            
            global_orient = pose_data['smplh:global_orient']
            body_pose = pose_data['smplh:body_pose']
            left_hand_pose = pose_data['smplh:left_hand_pose']
            right_hand_pose = pose_data['smplh:right_hand_pose']
            
            # 初始化有效性标记
            N = trans.shape[0]
            is_valid = np.ones(N, dtype=bool)
            
            # 如果存在smplh:is_valid键，使用该键的标记
            if "smplh:is_valid" in pose_data:
                is_valid &= pose_data["smplh:is_valid"]
            
            # 计算帧数
            num_frames = trans.shape[0]
            
            # 创建完整姿态数组，165维标准格式
            full_pose = np.zeros((num_frames, 55 * 3), dtype=np.float32)
            
            # Reshape组件为2D数组
            global_orient_2d = global_orient.reshape(num_frames, 3)
            body_pose_2d = body_pose.reshape(num_frames, 21*3)
            left_hand_pose_2d = left_hand_pose.reshape(num_frames, 15*3)
            right_hand_pose_2d = right_hand_pose.reshape(num_frames, 15*3)
            
            # 填充可用数据
            full_pose[:, 0:3] = global_orient_2d  # global_orient (1 joint)
            full_pose[:, 3:3+21*3] = body_pose_2d  # body_pose (21 joints)
            # Jaw, leye, reye保持为0
            full_pose[:, 3+21*3+3+3+3:3+21*3+3+3+3+15*3] = left_hand_pose_2d  # left_hand_pose (15 joints)
            full_pose[:, 3+21*3+3+3+3+15*3:] = right_hand_pose_2d  # right_hand_pose (15 joints)
            
            # 处理平移数据
            processed_trans = np.zeros_like(trans)
            
            # 计算平移速度
            trans_v = np.zeros_like(processed_trans)
            if num_frames > 1:
                trans_v[1:, 0] = processed_trans[1:, 0] - processed_trans[:-1, 0]
                trans_v[0, 0] = trans_v[1, 0]
                trans_v[1:, 2] = processed_trans[1:, 2] - processed_trans[:-1, 2]
                trans_v[0, 2] = trans_v[1, 2]
            trans_v[:, 1] = processed_trans[:, 1]
            
            # 提取ID
            fname = os.path.basename(pose_file)
            pid_str = fname.split('_')[-1].replace('.npz', '').replace('P', '').replace('A', '')
            try:
                pid = int(pid_str)
            except:
                pid = 0
            
            return {
                'pose': full_pose,
                'trans': processed_trans,
                'trans_v': trans_v,
                'motion_valid': is_valid,
                'id': pid
            }
            
        except Exception as e:
            logger.error(f"Error processing motion data from {pose_file}: {e}")
            return None
    
    def find_continuous_valid_segments(self, is_valid, min_duration_threshold=None):
        """
        找出时间连续的valid区间，并可选地筛选出满足最小时长要求的区间
        参考seamless_interaction_dataset
        
        Args:
            is_valid: 有效性标记，形状为 (N,)
            min_duration_threshold: 最小时长阈值（秒），如果为None则不进行时长筛选
        
        Returns:
            valid_segments: 连续valid区间的列表，每个元素为 (start_idx, end_idx)
        """
        valid_segments = []
        N = len(is_valid)
        
        i = 0
        while i < N:
            if is_valid[i]:
                # 找到一个valid区间的开始
                start_idx = i
                # 找到这个valid区间的结束
                while i < N and is_valid[i]:
                    i += 1
                end_idx = i - 1
                
                # 如果需要时长筛选且满足条件，则添加到结果中
                if min_duration_threshold is None:
                    valid_segments.append((start_idx, end_idx))
                else:
                    segment_duration = (end_idx - start_idx + 1) / self.pose_fps
                    if segment_duration >= min_duration_threshold:
                        valid_segments.append((start_idx, end_idx))
            else:
                i += 1
        
        return valid_segments
    
    def smooth_outliers(self, data, is_valid):
        """
        对异常值区间进行平滑处理，使用区间两端的正常值进行插值
        参考seamless_interaction_dataset
        
        Args:
            data: 原始数据，形状为 (N, D) 或可能的其他形状
            is_valid: 有效性标记，形状为 (N,)
        
        Returns:
            平滑后的数据，形状为 (N, D)
        """
        # 保存原始形状，以便最后恢复
        original_shape = data.shape
        
        # 确保data是2维数据
        if len(data.shape) > 2:
            # 如果是3维或更高维，将其转换为2维
            N, D = data.shape[0], -1
            data = data.reshape(N, D)
        else:
            N, D = data.shape
            
        smoothed_data = data.copy()
        
        # 找到所有异常帧的索引
        invalid_indices = np.where(~is_valid)[0]
        
        if len(invalid_indices) == 0:
            return smoothed_data
        
        # 将异常帧分组为连续区间
        invalid_groups = []
        current_group = [invalid_indices[0]]
        
        for i in range(1, len(invalid_indices)):
            if invalid_indices[i] == current_group[-1] + 1:
                current_group.append(invalid_indices[i])
            else:
                invalid_groups.append(current_group)
                current_group = [invalid_indices[i]]
        invalid_groups.append(current_group)
        
        # 对每个异常区间进行平滑处理
        for group in invalid_groups:
            start_idx = group[0]
            end_idx = group[-1]
            
            # 查找前一个有效帧
            prev_valid = None
            for idx in range(start_idx - 1, -1, -1):
                if is_valid[idx]:
                    prev_valid = idx
                    break
            
            # 查找后一个有效帧
            next_valid = None
            for idx in range(end_idx + 1, N):
                if is_valid[idx]:
                    next_valid = idx
                    break
            
            # 判断无有效插值基准（前后均无有效帧）
            if prev_valid is None and next_valid is None:
                # 无有效帧可供插值，跳过当前异常区间
                continue
            # 单端有有效帧时，用该端数据填充（而非插值）
            elif prev_valid is None:
                # 只有后有效帧，用后有效帧数据填充异常区间
                smoothed_data[start_idx:end_idx+1] = data[next_valid]
            elif next_valid is None:
                # 只有前有效帧，用前有效帧数据填充异常区间
                smoothed_data[start_idx:end_idx+1] = data[prev_valid]
            else:
                # 前后均有有效帧，执行线性插值
                prev_data = data[prev_valid]
                next_data = data[next_valid]
                
                group_length = end_idx - start_idx + 1
                for i, idx in enumerate(group):
                    alpha = (i + 1) / (group_length + 1)
                    smoothed_data[idx] = (1 - alpha) * prev_data + alpha * next_data
        
        # 恢复原始形状
        if len(original_shape) > 2:
            smoothed_data = smoothed_data.reshape(original_shape)
        
        return smoothed_data
    
    def _load_segment_data(self, segment_info):
        """
        加载完整片段数据（参考seamless_interaction_dataset风格）
        
        Args:
            segment_info: (pair_idx, start_frame, end_frame, speaker_idx, speaker_word_data, listener_word_data)
        
        Returns:
            dict: 完整片段数据
        """
        pair_idx, start_frame, end_frame, speaker_idx, speaker_word_data, listener_word_data = segment_info
        
        # 获取文件路径
        (p1_npz, p1_json), (p2_npz, p2_json) = self.pairs[pair_idx]
        
        # 根据speaker_idx确定角色
        if speaker_idx == 0:
            speaker_npz = p1_npz
            listener_npz = p2_npz
        else:
            speaker_npz = p2_npz
            listener_npz = p1_npz
        
        # 加载两人姿态数据
        speaker_data = self._load_seamless_motion_data(speaker_npz)
        listener_data = self._load_seamless_motion_data(listener_npz)
        
        if speaker_data is None or listener_data is None:
            return None
        
        # 确定目标长度（使用姿态序列的长度）
        target_length = min(len(speaker_data['pose']), len(listener_data['pose']))
        
        # 提取音频特征（仅Speaker需要），传入目标长度以确保长度匹配
        speaker_wav_path = speaker_npz.replace('.npz', '.wav')
        speaker_audio = self._extract_audio_features(speaker_wav_path, target_length=target_length)
        
        if speaker_audio is None:
            logger.warning(f"Failed to extract audio for {speaker_npz}")
            return None
        
        # 切分片段数据
        # 注意：speaker_word_data 和 listener_word_data 已经是切片后的数据
        # 它们的长度应该是 end_frame - start_frame
        # 所以不需要对 word 数据进行边界检查，也不需要再切片
        
        # 确保不越界（只检查 pose 和 audio）
        original_end_frame = end_frame
        end_frame = min(end_frame, len(speaker_data['pose']))
        end_frame = min(end_frame, len(listener_data['pose']))
        end_frame = min(end_frame, len(speaker_audio))
        
        # word 数据已经是切片后的，但长度可能需要调整
        # 如果 end_frame 被缩短了，需要相应缩短 word 数据
        word_length = len(speaker_word_data)
        expected_word_length = end_frame - start_frame
        
        if word_length > expected_word_length:
            # word 数据太长，截断
            speaker_word_data = speaker_word_data[:expected_word_length]
            listener_word_data = listener_word_data[:expected_word_length]
        elif word_length < expected_word_length:
            # word 数据太短，这是异常情况
            logger.warning(f"Word data too short: {word_length} < {expected_word_length}")
            return None
        
        # 调试信息
        if end_frame != original_end_frame:
            logger.debug(f"Boundary adjustment: {original_end_frame} -> {end_frame} "
                       f"(pose: {min(len(speaker_data['pose']), len(listener_data['pose']))}, "
                       f"audio: {len(speaker_audio)})")
        
        if end_frame <= start_frame + self.horizon:
            logger.warning(f"Segment too short after boundary check: "
                        f"end_frame={end_frame}, start_frame={start_frame}, "
                        f"required={start_frame + self.horizon}, "
                        f"segment_length={end_frame - start_frame}")
            return None
        
        # 构建样本 - 不存储seed，只存储完整序列
        # word 数据已经是切片后的，直接使用
        speaker_sample = {
            'audio': speaker_audio[start_frame:end_frame],
            'word': speaker_word_data,  # 已经是切片后的数据
            'pose': speaker_data['pose'][start_frame:end_frame],
            'id': speaker_data['id'],
            'audio_name': speaker_wav_path,
        }
        
        listener_sample = {
            'pose': listener_data['pose'][start_frame:end_frame],
            'id': listener_data['id'],
        }
        
        return {
            'speaker': speaker_sample,
            'listener': listener_sample,
            'metadata': {
                'speaker_file': os.path.basename(speaker_npz),
                'listener_file': os.path.basename(listener_npz),
                'start_frame': start_frame,
                'end_frame': end_frame,
            }
        }
    
    def __len__(self):
        """返回片段数量"""
        if self.use_lmdb and self.mapping_data is not None:
            return self.n_samples
        return len(self.valid_segments)
    
    def __getitem__(self, idx):
        """
        获取一个片段
        
        Returns:
            dict with keys:
                'speaker': dict with 'audio', 'word', 'pose', 'id', 'audio_name'
                'listener': dict with 'pose', 'id'
                'metadata': dict with file info
        """
        if self.use_lmdb:
            # 从LMDB加载
            lmdb_env = self._get_lmdb_env()
            with lmdb_env.begin(write=False) as txn:
                key = f"{idx:08d}".encode('ascii')
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Sample {idx} not found in LMDB")
                sample_data = pickle.loads(value)
                return sample_data
        else:
            # 直接从文件加载
            segment_info = self.valid_segments[idx]
            return self._load_segment_data(segment_info)


def collate_fn(batch):
    """自定义批处理函数 - 用于完整序列"""
    # 过滤None值
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    collated = {'speaker': {}, 'listener': {}, 'metadata': []}
    
    # 处理speaker数据
    speaker_keys = batch[0]['speaker'].keys()
    for key in speaker_keys:
        values = [item['speaker'][key] for item in batch]
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            # 对于变长序列，使用列表存储而不是stack
            collated['speaker'][key] = [torch.as_tensor(v) for v in values]
        elif isinstance(values[0], (int, float)):
            collated['speaker'][key] = torch.tensor(values)
        else:
            collated['speaker'][key] = values
    
    # 处理listener数据
    listener_keys = batch[0]['listener'].keys()
    for key in listener_keys:
        values = [item['listener'][key] for item in batch]
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            # 对于变长序列，使用列表存储而不是stack
            collated['listener'][key] = [torch.as_tensor(v) for v in values]
        elif isinstance(values[0], (int, float)):
            collated['listener'][key] = torch.tensor(values)
        else:
            collated['listener'][key] = values
    
    # 处理metadata
    collated['metadata'] = [item['metadata'] for item in batch]
    
    return collated
