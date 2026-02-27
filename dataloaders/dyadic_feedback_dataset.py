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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.build_vocab import Vocab


class DyadicFeedbackDataset(Dataset):
    def __init__(self, args, split='train', build_cache=True):
        self.args = args
        self.split = split
        
        data_cfg = args.get('data', args)
        base_data_path = data_cfg.get('data_path', args.get('data_path', './datasets/seamless_interaction/improvised/'))
        
        # 根据split使用不同的数据目录
        if split == 'train':
            self.data_path = os.path.join(base_data_path, 'train')
        elif split == 'val':
            self.data_path = os.path.join(base_data_path, 'val')
        else:
            self.data_path = os.path.join(base_data_path, 'test')
        
        logger.info(f"[{split}] Using data path: {self.data_path}")
        
        self.stride = data_cfg.get('stride', getattr(args, 'stride', 60))
        self.min_speak_duration = data_cfg.get('min_speak_duration', getattr(args, 'min_speak_duration', 5.0))
        
        self.pose_fps = data_cfg.get('pose_fps', getattr(args, 'pose_fps', 30))
        self.audio_fps = data_cfg.get('audio_fps', getattr(args, 'audio_fps', 100))
        self.audio_sr = data_cfg.get('audio_sr', getattr(args, 'audio_sr', 48000))
        
        self.audio_rep = data_cfg.get('audio_rep', getattr(args, 'audio_rep', 'mfcc'))
        
        self.mean_audio = getattr(args, 'mean_audio', 0.0)
        self.std_audio = getattr(args, 'std_audio', 1.0)
        self.audio_norm = data_cfg.get('audio_norm', getattr(args, 'audio_norm', False))
        
        self.training_speakers = getattr(args, 'training_speakers', None)
        self.disable_filtering = data_cfg.get('disable_filtering', getattr(args, 'disable_filtering', False))
        self.clean_first_seconds = data_cfg.get('clean_first_seconds', getattr(args, 'clean_first_seconds', 0))
        self.clean_final_seconds = data_cfg.get('clean_final_seconds', getattr(args, 'clean_final_seconds', 0))
        self.skip_segmentation = getattr(args, 'skip_segmentation', False)
        
        self.cache_dir = data_cfg.get('cache_dir', data_cfg.get('cache_path', getattr(args, 'cache_dir', 'cache/dyadic_feedback')))
        self.lmdb_path = os.path.join(self.cache_dir, split)
        self.use_lmdb = getattr(args, 'use_lmdb', True)
        self.lmdb_env = None
        
        self.vocab_path = getattr(args, 'vocab_path', 'datasets/unified_vocab.pkl')
        model_cfg = args.get('model', {})
        modality_cfg = model_cfg.get('modality_encoder', {}).get('params', {})
        if 'vocab_path' in modality_cfg:
            self.vocab_path = modality_cfg['vocab_path']
        
        self.lang_model = None
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, 'rb') as f:
                    self.lang_model = pickle.load(f)
                logger.info(f"Loaded language model from {self.vocab_path}")
            except Exception as e:
                logger.warning(f"Failed to load language model: {e}")
        else:
            logger.warning(f"Vocab file not found: {self.vocab_path}")
        
        self.multi_length_training = data_cfg.get('multi_length_training', getattr(args, 'multi_length_training', [1.0]))
        self.pose_length = data_cfg.get('pose_length', getattr(args, 'pose_length', 128))
        
        if split == 'train':
            self.use_slicing = True
        else:
            self.use_slicing = False
            self.multi_length_training = [1.0]
        
        logger.info(f"[{split}] Scanning files from {self.data_path}...")
        self.pairs = self._scan_and_pair_files()
        logger.info(f"[{split}] Found {len(self.pairs)} interaction pairs after filtering")
        
        if self.use_lmdb:
            self._init_cache(build_cache)
        else:
            self.valid_segments = []
    
    def _init_cache(self, build_cache):
        self.mapping_data = None
        
        if build_cache:
            self._build_lmdb_cache()
        
        self._load_lmdb_mapping()
    
    def _build_lmdb_cache(self):
        logger.info(f"Building LMDB cache at {self.lmdb_path}...")
        
        data_cfg = self.args.get('data', self.args)
        new_cache = data_cfg.get('new_cache', getattr(self.args, 'new_cache', False))
        
        if new_cache:
            mapping_file = os.path.join(self.lmdb_path, "mapping.pkl")
            if os.path.exists(mapping_file):
                os.remove(mapping_file)
                logger.info(f"Removed mapping file to force rebuild: {mapping_file}")
            if os.path.exists(self.lmdb_path):
                try:
                    shutil.rmtree(self.lmdb_path)
                    logger.info(f"Removed old cache: {self.lmdb_path}")
                except Exception as e:
                    logger.warning(f"Could not remove cache dir: {e}, will overwrite")
        
        if os.path.exists(self.lmdb_path) and os.path.exists(os.path.join(self.lmdb_path, "mapping.pkl")):
            logger.info(f"Found existing cache at {self.lmdb_path}")
            return
        
        os.makedirs(self.lmdb_path, exist_ok=True)
        
        map_size = 100 * 1024 * 1024 * 1024
        lmdb_env = lmdb.open(self.lmdb_path, map_size=map_size)
        
        sample_idx = 0
        n_filtered = defaultdict(int)
        n_segments = 0
        n_pairs_processed = 0
        n_pairs_with_segments = 0
        
        with lmdb_env.begin(write=True) as txn:
            for pair_idx, ((p1_npz, p1_json), (p2_npz, p2_json)) in enumerate(tqdm(self.pairs, desc="Processing pairs")):
                pair_data = self._load_pair_data_once(p1_npz, p1_json, p2_npz, p2_json)
                
                if pair_data is None:
                    n_filtered['load_error'] += 1
                    continue
                
                n_pairs_processed += 1
                
                if self.use_slicing:
                    segments = self._generate_segments_from_pair(pair_data, pair_idx)
                else:
                    segments = self._generate_full_pair_segment(pair_data, pair_idx)
                
                if len(segments) == 0:
                    n_filtered['no_segments'] += 1
                    continue
                
                n_pairs_with_segments += 1
                
                for segment in segments:
                    key = f"{sample_idx:08d}".encode('ascii')
                    value = pickle.dumps(segment)
                    txn.put(key, value)
                    sample_idx += 1
                    n_segments += 1
        
        lmdb_env.close()
        
        mapping_data = {
            'n_samples': sample_idx,
            'n_pairs': len(self.pairs),
        }
        
        with open(os.path.join(self.lmdb_path, "mapping.pkl"), 'wb') as f:
            pickle.dump(mapping_data, f)
        
        logger.info(f"Cache built: {sample_idx} segments from {len(self.pairs)} pairs")
        logger.info(f"Pairs processed: {n_pairs_processed}, with segments: {n_pairs_with_segments}")
        logger.info(f"Filtered: {dict(n_filtered)}")
    
    def _load_pair_data_once(self, p1_npz, p1_json, p2_npz, p2_json):
        p1_pose_data = self._load_seamless_motion_data(p1_npz)
        p2_pose_data = self._load_seamless_motion_data(p2_npz)
        
        if p1_pose_data is None or p2_pose_data is None:
            return None
        
        p1_json_data = self._load_and_process_json(p1_json)
        p2_json_data = self._load_and_process_json(p2_json)
        
        if p1_json_data is None or p2_json_data is None:
            return None
        
        p1_wav_path = p1_npz.replace('.npz', '.wav')
        p2_wav_path = p2_npz.replace('.npz', '.wav')
        
        # 音频帧率和姿态帧率不同，不需要target_length
        p1_audio = self._extract_audio_features(p1_wav_path)
        p2_audio = self._extract_audio_features(p2_wav_path)
        
        return {
            'p1': {
                'pose_data': p1_pose_data,
                'json_data': p1_json_data,
                'audio': p1_audio,
                'npz_path': p1_npz,
            },
            'p2': {
                'pose_data': p2_pose_data,
                'json_data': p2_json_data,
                'audio': p2_audio,
                'npz_path': p2_npz,
            }
        }
    
    def _generate_segments_from_pair(self, pair_data, pair_idx):
        segments = []
        
        p1_pose = pair_data['p1']['pose_data']['pose']
        p2_pose = pair_data['p2']['pose_data']['pose']
        p1_motion_valid = pair_data['p1']['pose_data'].get('motion_valid', np.ones(len(p1_pose), dtype=bool))
        p2_motion_valid = pair_data['p2']['pose_data'].get('motion_valid', np.ones(len(p2_pose), dtype=bool))
        p1_utterances = pair_data['p1']['json_data'][0]
        p2_utterances = pair_data['p2']['json_data'][0]
        p1_word_data = pair_data['p1']['json_data'][1]
        p2_word_data = pair_data['p2']['json_data'][1]
        
        if len(p1_word_data) != len(p1_pose):
            p1_word_data = self._align_word_data(p1_word_data, len(p1_pose))
        if len(p2_word_data) != len(p2_pose):
            p2_word_data = self._align_word_data(p2_word_data, len(p2_pose))
        
        total_frames = min(len(p1_pose), len(p2_pose))
        
        p1_final_valid = p1_motion_valid[:total_frames] & (np.ones(len(p1_word_data[:total_frames]), dtype=bool) if p1_word_data is not None else p1_motion_valid[:total_frames])
        p2_final_valid = p2_motion_valid[:total_frames] & (np.ones(len(p2_word_data[:total_frames]), dtype=bool) if p2_word_data is not None else p2_motion_valid[:total_frames])
        
        total_seconds = total_frames / self.pose_fps
        clip_start_t = self.clean_first_seconds
        clip_end_t = total_seconds - self.clean_final_seconds
        
        if clip_start_t >= clip_end_t:
            return segments
        
        clip_start_frame = int(clip_start_t * self.pose_fps)
        clip_end_frame = int(clip_end_t * self.pose_fps)
        
        min_segment_length = self.pose_length * 2
        
        if clip_end_frame - clip_start_frame < min_segment_length:
            return segments
        
        p1_valid_segments = self.find_continuous_valid_segments(p1_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
        p2_valid_segments = self.find_continuous_valid_segments(p2_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
        
        n_p1_segments = 0
        n_p2_segments = 0
        n_filtered_duration = 0
        n_filtered_speaker = 0
        
        window_duration = 10.0
        window_size = self.pose_length
        
        for p1_seg in p1_valid_segments:
            p1_seg_start, p1_seg_end = p1_seg
            p1_abs_start = clip_start_frame + p1_seg_start
            p1_abs_end = clip_start_frame + p1_seg_end
            
            for window_start in range(p1_abs_start, p1_abs_end - window_size + 1, self.stride):
                window_end = window_start + window_size
                
                mid_frame = (window_start + window_end) // 2
                mid_time = mid_frame / self.pose_fps
                
                p1_is_speaker = self._check_is_speaker(p1_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
                p2_is_speaker = self._check_is_speaker(p2_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
                
                if p1_is_speaker or p2_is_speaker:
                    if p1_is_speaker and p2_is_speaker:
                        p1_speak_time = 0.0
                        p2_speak_time = 0.0
                        for utt in p1_utterances:
                            utt_start = utt.get('start', 0)
                            utt_end = utt.get('end', 0)
                            overlap_start = max(window_start / self.pose_fps, utt_start)
                            overlap_end = min(window_end / self.pose_fps, utt_end)
                            if overlap_start < overlap_end:
                                p1_speak_time += (overlap_end - overlap_start)
                        for utt in p2_utterances:
                            utt_start = utt.get('start', 0)
                            utt_end = utt.get('end', 0)
                            overlap_start = max(window_start / self.pose_fps, utt_start)
                            overlap_end = min(window_end / self.pose_fps, utt_end)
                            if overlap_start < overlap_end:
                                p2_speak_time += (overlap_end - overlap_start)
                        
                        if p1_speak_time >= p2_speak_time:
                            segment = self._create_segment(pair_data, window_start, window_end, speaker_idx=0)
                        else:
                            segment = self._create_segment(pair_data, window_start, window_end, speaker_idx=1)
                    elif p1_is_speaker:
                        segment = self._create_segment(pair_data, window_start, window_end, speaker_idx=0)
                    elif p2_is_speaker:
                        segment = self._create_segment(pair_data, window_start, window_end, speaker_idx=1)
                    
                    if segment is not None:
                        segments.append(segment)
                        if p1_is_speaker:
                            n_p1_segments += 1
                        else:
                            n_p2_segments += 1
                else:
                    n_filtered_speaker += 1
        
        return segments
    
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
    
    def _generate_full_pair_segment(self, pair_data, pair_idx):
        segments = []
        
        p1_pose = pair_data['p1']['pose_data']['pose']
        p2_pose = pair_data['p2']['pose_data']['pose']
        p1_motion_valid = pair_data['p1']['pose_data'].get('motion_valid', np.ones(len(p1_pose), dtype=bool))
        p2_motion_valid = pair_data['p2']['pose_data'].get('motion_valid', np.ones(len(p2_pose), dtype=bool))
        p1_utterances = pair_data['p1']['json_data'][0]
        p2_utterances = pair_data['p2']['json_data'][0]
        p1_word_data = pair_data['p1']['json_data'][1]
        p2_word_data = pair_data['p2']['json_data'][1]
        
        if len(p1_word_data) != len(p1_pose):
            p1_word_data = self._align_word_data(p1_word_data, len(p1_pose))
        if len(p2_word_data) != len(p2_pose):
            p2_word_data = self._align_word_data(p2_word_data, len(p2_pose))
        
        total_frames = min(len(p1_pose), len(p2_pose))
        
        p1_final_valid = p1_motion_valid[:total_frames] & (np.ones(len(p1_word_data[:total_frames]), dtype=bool) if p1_word_data is not None else p1_motion_valid[:total_frames])
        p2_final_valid = p2_motion_valid[:total_frames] & (np.ones(len(p2_word_data[:total_frames]), dtype=bool) if p2_word_data is not None else p2_motion_valid[:total_frames])
        
        total_seconds = total_frames / self.pose_fps
        clip_start_t = self.clean_first_seconds
        clip_end_t = total_seconds - self.clean_final_seconds
        
        if clip_start_t >= clip_end_t:
            return segments
        
        clip_start_frame = int(clip_start_t * self.pose_fps)
        clip_end_frame = int(clip_end_t * self.pose_fps)
        
        min_segment_length = self.pose_length * 2
        if clip_end_frame - clip_start_frame < min_segment_length:
            return segments
        
        p1_valid_segments = self.find_continuous_valid_segments(p1_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
        p2_valid_segments = self.find_continuous_valid_segments(p2_final_valid[clip_start_frame:clip_end_frame], min_duration_threshold=5.0)
        
        for p1_seg in p1_valid_segments:
            p1_seg_start, p1_seg_end = p1_seg
            p1_abs_start = clip_start_frame + p1_seg_start
            p1_abs_end = clip_start_frame + p1_seg_end
            
            mid_frame = (p1_abs_start + p1_abs_end) // 2
            mid_time = mid_frame / self.pose_fps
            window_duration = 10.0
            
            p1_is_speaker = self._check_is_speaker(p1_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
            p2_is_speaker = self._check_is_speaker(p2_utterances, mid_time - window_duration/2, mid_time + window_duration/2)
            
            if p1_is_speaker and not p2_is_speaker:
                segment = self._create_full_segment(pair_data, p1_abs_start, p1_abs_end, speaker_idx=0)
                if segment is not None:
                    segments.append(segment)
            elif p2_is_speaker and not p1_is_speaker:
                segment = self._create_full_segment(pair_data, p1_abs_start, p1_abs_end, speaker_idx=1)
                if segment is not None:
                    segments.append(segment)
        
        return segments
    
    def _create_full_segment(self, pair_data, start_frame, end_frame, speaker_idx):
        if speaker_idx == 0:
            speaker_data = pair_data['p1']
            listener_data = pair_data['p2']
        else:
            speaker_data = pair_data['p2']
            listener_data = pair_data['p1']
        
        speaker_pose_data = speaker_data['pose_data']
        listener_pose_data = listener_data['pose_data']
        speaker_audio = speaker_data['audio']
        speaker_word_data = speaker_data['json_data'][1]
        
        target_length = end_frame - start_frame
        
        if end_frame > len(speaker_pose_data['pose']):
            end_frame = len(speaker_pose_data['pose'])
            target_length = end_frame - start_frame
        if end_frame > len(listener_pose_data['pose']):
            end_frame = len(listener_pose_data['pose'])
            target_length = end_frame - start_frame
        
        speaker_sample = {
            'pose': speaker_pose_data['pose'][start_frame:end_frame],
            'trans_v': speaker_pose_data['trans_v'][start_frame:end_frame],
            'id': speaker_pose_data['id'],
        }
        
        if speaker_audio is not None:
            audio_start = int(start_frame * self.audio_fps / self.pose_fps)
            audio_end = int(end_frame * self.audio_fps / self.pose_fps)
            audio_end = min(audio_end, len(speaker_audio))
            audio_data = speaker_audio[audio_start:audio_end]
            
            target_audio_len = int(target_length * self.audio_fps / self.pose_fps)
            if len(audio_data) < target_audio_len:
                pad_length = target_audio_len - len(audio_data)
                audio_data = np.concatenate([
                    audio_data,
                    np.zeros((pad_length,) + audio_data.shape[1:], dtype=audio_data.dtype)
                ])
            elif len(audio_data) > target_audio_len:
                audio_data = audio_data[:target_audio_len]
            speaker_sample['audio'] = audio_data
            
            word_data = speaker_word_data[start_frame:end_frame]
            if len(word_data) < target_length:
                pad_length = target_length - len(word_data)
                word_data = np.concatenate([
                    word_data, 
                    np.full(pad_length, self.lang_model.PAD_token if self.lang_model else 0, dtype=np.int64)
                ])
            speaker_sample['word'] = word_data
            speaker_sample['audio_name'] = speaker_data['npz_path'].replace('.npz', '.wav')
        
        listener_sample = {
            'pose': listener_pose_data['pose'][start_frame:end_frame],
            'trans_v': listener_pose_data['trans_v'][start_frame:end_frame],
            'id': listener_pose_data['id'],
        }
        
        return {
            'speaker': speaker_sample,
            'listener': listener_sample,
            'metadata': {
                'speaker_file': os.path.basename(speaker_data['npz_path']),
                'listener_file': os.path.basename(listener_data['npz_path']),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'speaker_idx': speaker_idx,
            }
        }
    
    def _create_segment(self, pair_data, start_frame, end_frame, speaker_idx):
        if speaker_idx == 0:
            speaker_data = pair_data['p1']
            listener_data = pair_data['p2']
        else:
            speaker_data = pair_data['p2']
            listener_data = pair_data['p1']
        
        speaker_pose_data = speaker_data['pose_data']
        listener_pose_data = listener_data['pose_data']
        speaker_audio = speaker_data['audio']
        speaker_word_data = speaker_data['json_data'][1]
        
        # 固定长度为 pose_length
        target_length = self.pose_length
        end_frame = start_frame + target_length
        
        # 检查是否超出范围
        if end_frame > len(speaker_pose_data['pose']):
            return None
        if end_frame > len(listener_pose_data['pose']):
            return None
        
        speaker_sample = {
            'pose': speaker_pose_data['pose'][start_frame:end_frame],
            'trans_v': speaker_pose_data['trans_v'][start_frame:end_frame],
            'id': speaker_pose_data['id'],
        }
        
        if speaker_audio is not None:
            audio_start = int(start_frame * self.audio_fps / self.pose_fps)
            audio_end = int(end_frame * self.audio_fps / self.pose_fps)
            audio_end = min(audio_end, len(speaker_audio))
            audio_data = speaker_audio[audio_start:audio_end]
            
            target_audio_len = int(target_length * self.audio_fps / self.pose_fps)
            if len(audio_data) < target_audio_len:
                pad_length = target_audio_len - len(audio_data)
                audio_data = np.concatenate([
                    audio_data,
                    np.zeros((pad_length,) + audio_data.shape[1:], dtype=audio_data.dtype)
                ])
            elif len(audio_data) > target_audio_len:
                audio_data = audio_data[:target_audio_len]
            speaker_sample['audio'] = audio_data
            
            word_data = speaker_word_data[start_frame:end_frame]
            if len(word_data) < target_length:
                pad_length = target_length - len(word_data)
                word_data = np.concatenate([
                    word_data, 
                    np.full(pad_length, self.lang_model.PAD_token if self.lang_model else 0, dtype=np.int64)
                ])
            speaker_sample['word'] = word_data
            speaker_sample['audio_name'] = speaker_data['npz_path'].replace('.npz', '.wav')
        
        listener_sample = {
            'pose': listener_pose_data['pose'][start_frame:end_frame],
            'trans_v': listener_pose_data['trans_v'][start_frame:end_frame],
            'id': listener_pose_data['id'],
        }
        
        return {
            'speaker': speaker_sample,
            'listener': listener_sample,
            'metadata': {
                'speaker_file': os.path.basename(speaker_data['npz_path']),
                'listener_file': os.path.basename(listener_data['npz_path']),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'speaker_idx': speaker_idx,
            }
        }
    
    def _load_lmdb_mapping(self):
        mapping_file = os.path.join(self.lmdb_path, "mapping.pkl")
        
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        with open(mapping_file, 'rb') as f:
            self.mapping_data = pickle.load(f)
        
        self.n_samples = self.mapping_data['n_samples']
        logger.info(f"Loaded LMDB mapping: {self.n_samples} segments")
    
    def _get_lmdb_env(self):
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        return self.lmdb_env
    
    def _scan_and_pair_files(self):
        pattern = os.path.join(self.data_path, "**", "*.npz")
        all_npz_files = sorted(glob(pattern, recursive=True))
        
        logger.info(f"Total NPZ files found: {len(all_npz_files)}")
        
        if not self.disable_filtering and self.training_speakers:
            filtered_files = []
            for npz_file in all_npz_files:
                speaker_id_match = re.search(r'P(\d+)[A-Za-z]*\.npz$', npz_file)
                if speaker_id_match:
                    file_id = int(speaker_id_match.group(1))
                    if file_id in self.training_speakers:
                        filtered_files.append(npz_file)
            
            logger.info(f"Filtered {len(filtered_files)} files based on training_speakers")
            
            if not filtered_files:
                logger.warning(f"No files match training_speakers, using all files")
                filtered_files = all_npz_files
            
            all_npz_files = filtered_files
        else:
            logger.info("ID filtering disabled or no training_speakers specified")
        
        groups = defaultdict(list)
        for npz_path in all_npz_files:
            fname = os.path.basename(npz_path)
            parts = fname.replace('.npz', '').split('_')
            if len(parts) >= 4:
                key = "_".join(parts[:3])
                json_path = npz_path.replace('.npz', '.json')
                if os.path.exists(json_path):
                    groups[key].append((npz_path, json_path))
        
        pairs = []
        for key, files in groups.items():
            if len(files) == 2:
                files.sort(key=lambda x: os.path.basename(x[0]).split('_')[-1])
                pairs.append(tuple(files))
            elif len(files) > 2:
                files.sort(key=lambda x: os.path.basename(x[0]).split('_')[-1])
                pairs.append(tuple(files[:2]))
        
        return pairs
    
    def _get_speaking_ranges(self, utterances, clip_start, clip_end):
        ranges = []
        for utt in utterances:
            start = utt.get('start')
            end = utt.get('end')
            
            if start is None or end is None:
                continue
            
            if start < clip_end and end > clip_start:
                overlap_start = max(start, clip_start)
                overlap_end = min(end, clip_end)
                if overlap_end - overlap_start > 0:
                    ranges.append((overlap_start, overlap_end))
        
        return ranges
    
    def _load_and_process_json(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if 'metadata:transcript' in json_data:
                utterances = json_data['metadata:transcript']
            elif 'utterances' in json_data:
                utterances = json_data['utterances']
            else:
                utterances = []
            
            all_words = []
            transcript_ranges = []
            for transcript in json_data.get('metadata:transcript', []):
                transcript_start = transcript.get('start')
                transcript_end = transcript.get('end')
                if transcript_start is not None and transcript_end is not None:
                    transcript_ranges.append((transcript_start, transcript_end))
                all_words.extend(transcript.get('words', []))
            
            all_words.sort(key=lambda x: x.get('start') if x.get('start') is not None else 0)
            transcript_ranges.sort(key=lambda x: x[0])
            
            if transcript_ranges:
                max_time = max(tr[1] for tr in transcript_ranges)
                estimated_frames = int(max_time * self.pose_fps) + 1
            else:
                estimated_frames = 1000
            
            word_data = self._process_word_encoding(all_words, transcript_ranges, estimated_frames)
            
            return utterances, np.array(word_data, dtype=np.int64)
            
        except Exception as e:
            logger.warning(f"Failed to load and process {json_path}: {e}")
            return None
    
    def _align_word_data(self, word_data, target_frames):
        current_len = len(word_data)
        if current_len == target_frames:
            return word_data
        elif current_len < target_frames:
            pad_length = target_frames - current_len
            return np.concatenate([word_data, np.full(pad_length, self.lang_model.PAD_token if self.lang_model else 0, dtype=np.int64)])
        else:
            return word_data[:target_frames]
    
    def _process_word_encoding(self, word_list, transcript_ranges, num_frames):
        word_data = []
        
        if self.lang_model is None:
            return [0] * num_frames
        
        transcript_idx = 0
        word_idx = 0
        num_transcripts = len(transcript_ranges)
        num_words = len(word_list)
        
        time_extension = 0.1
        
        for frame_idx in range(num_frames):
            current_time = frame_idx / self.pose_fps
            
            while transcript_idx < num_transcripts and current_time > transcript_ranges[transcript_idx][1]:
                transcript_idx += 1
            
            found_word = False
            while word_idx < num_words:
                start_time = word_list[word_idx].get('start')
                end_time = word_list[word_idx].get('end')
                
                if start_time is None or end_time is None:
                    word_idx += 1
                    continue
                
                if current_time < start_time - time_extension:
                    break
                
                if start_time - time_extension <= current_time <= end_time + time_extension:
                    word = word_list[word_idx].get('word', '')
                    if word.strip():
                        word_data.append(self.lang_model.get_word_index(word))
                    else:
                        word_data.append(self.lang_model.PAD_token)
                    found_word = True
                    break
                else:
                    word_idx += 1
            
            if not found_word:
                word_data.append(self.lang_model.PAD_token)
        
        return word_data
    
    def _extract_audio_features(self, wav_path, target_length=None):
        if not os.path.exists(wav_path):
            return None
        
        try:
            if self.audio_rep == "mfcc":
                audio_data, _ = librosa.load(wav_path, sr=self.audio_sr)
                
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=self.audio_sr,
                    n_mels=128,
                    hop_length=int(self.audio_sr / self.audio_fps)
                )
                
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                log_mel_spec = log_mel_spec.transpose(1, 0)
                
                if target_length is not None and len(log_mel_spec) < target_length:
                    pad_length = target_length - len(log_mel_spec)
                    last_frame = log_mel_spec[-1:]
                    padding = np.tile(last_frame, (pad_length, 1))
                    log_mel_spec = np.concatenate([log_mel_spec, padding], axis=0)
                elif target_length is not None and len(log_mel_spec) > target_length:
                    log_mel_spec = log_mel_spec[:target_length]
                
                log_mel_spec = (log_mel_spec + 80.0) / 80.0 * 2.0 - 1.0
                
                if self.audio_norm:
                    log_mel_spec = (log_mel_spec - self.mean_audio) / (self.std_audio + 1e-8)
                
                return log_mel_spec.astype(np.float32)
            
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting audio from {wav_path}: {e}")
            return None
    
    def _load_seamless_motion_data(self, pose_file):
        try:
            pose_data = np.load(pose_file, allow_pickle=True)
            
            trans = pose_data['smplh:translation']
            trans = trans / 100.0
            
            global_orient = pose_data['smplh:global_orient']
            body_pose = pose_data['smplh:body_pose']
            left_hand_pose = pose_data['smplh:left_hand_pose']
            right_hand_pose = pose_data['smplh:right_hand_pose']
            
            N = trans.shape[0]
            is_valid = np.ones(N, dtype=bool)
            
            if "smplh:is_valid" in pose_data:
                is_valid &= pose_data["smplh:is_valid"]
            
            num_frames = trans.shape[0]
            
            full_pose = np.zeros((num_frames, 55 * 3), dtype=np.float32)
            
            global_orient_2d = global_orient.reshape(num_frames, 3)
            body_pose_2d = body_pose.reshape(num_frames, 21*3)
            left_hand_pose_2d = left_hand_pose.reshape(num_frames, 15*3)
            right_hand_pose_2d = right_hand_pose.reshape(num_frames, 15*3)
            
            full_pose[:, 0:3] = global_orient_2d
            full_pose[:, 3:3+21*3] = body_pose_2d
            full_pose[:, 3+21*3+3+3+3:3+21*3+3+3+3+15*3] = left_hand_pose_2d
            full_pose[:, 3+21*3+3+3+3+15*3:] = right_hand_pose_2d
            
            processed_trans = np.zeros_like(trans)
            
            trans_v = np.zeros_like(processed_trans)
            if num_frames > 1:
                trans_v[1:, 0] = processed_trans[1:, 0] - processed_trans[:-1, 0]
                trans_v[0, 0] = trans_v[1, 0]
                trans_v[1:, 2] = processed_trans[1:, 2] - processed_trans[:-1, 2]
                trans_v[0, 2] = trans_v[1, 2]
            trans_v[:, 1] = processed_trans[:, 1]
            
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
                start_idx = i
                while i < N and is_valid[i]:
                    i += 1
                end_idx = i - 1
                
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
        
        Args:
            data: 原始数据，形状为 (N, D) 或可能的其他形状
            is_valid: 有效性标记，形状为 (N,)
        
        Returns:
            平滑后的数据，形状为 (N, D)
        """
        original_shape = data.shape
        
        if len(data.shape) > 2:
            N, D = data.shape[0], -1
            data = data.reshape(N, D)
        else:
            N, D = data.shape
            
        smoothed_data = data.copy()
        
        invalid_indices = np.where(~is_valid)[0]
        
        if len(invalid_indices) == 0:
            return smoothed_data
        
        invalid_groups = []
        current_group = [invalid_indices[0]]
        
        for i in range(1, len(invalid_indices)):
            if invalid_indices[i] == current_group[-1] + 1:
                current_group.append(invalid_indices[i])
            else:
                invalid_groups.append(current_group)
                current_group = [invalid_indices[i]]
        invalid_groups.append(current_group)
        
        for group in invalid_groups:
            start_idx = group[0]
            end_idx = group[-1]
            
            prev_valid = None
            for idx in range(start_idx - 1, -1, -1):
                if is_valid[idx]:
                    prev_valid = idx
                    break
            
            next_valid = None
            for idx in range(end_idx + 1, N):
                if is_valid[idx]:
                    next_valid = idx
                    break
            
            if prev_valid is None and next_valid is None:
                continue
            elif prev_valid is None:
                smoothed_data[start_idx:end_idx+1] = data[next_valid]
            elif next_valid is None:
                smoothed_data[start_idx:end_idx+1] = data[prev_valid]
            else:
                prev_data = data[prev_valid]
                next_data = data[next_valid]
                
                group_length = end_idx - start_idx + 1
                for i, idx in enumerate(group):
                    alpha = (i + 1) / (group_length + 1)
                    smoothed_data[idx] = (1 - alpha) * prev_data + alpha * next_data
        
        if len(original_shape) > 2:
            smoothed_data = smoothed_data.reshape(original_shape)
        
        return smoothed_data
    
    def __len__(self):
        if self.use_lmdb and self.mapping_data is not None:
            return self.n_samples
        return 0
    
    def __getitem__(self, idx):
        if self.use_lmdb:
            lmdb_env = self._get_lmdb_env()
            with lmdb_env.begin(write=False) as txn:
                key = f"{idx:08d}".encode('ascii')
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Sample {idx} not found in LMDB")
                sample_data = pickle.loads(value)
                return sample_data
        else:
            raise NotImplementedError("Non-LMDB mode not supported")


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    collated = {'speaker': {}, 'listener': {}, 'metadata': []}
    
    speaker_keys = batch[0]['speaker'].keys()
    for key in speaker_keys:
        values = [item['speaker'][key] for item in batch]
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            tensors = [torch.as_tensor(v) for v in values]
            shapes = [t.shape for t in tensors]
            if all(s == shapes[0] for s in shapes):
                collated['speaker'][key] = torch.stack(tensors, dim=0)
            else:
                collated['speaker'][key] = tensors
        elif isinstance(values[0], (int, float)):
            collated['speaker'][key] = torch.tensor(values)
        else:
            collated['speaker'][key] = values
    
    listener_keys = batch[0]['listener'].keys()
    for key in listener_keys:
        values = [item['listener'][key] for item in batch]
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            tensors = [torch.as_tensor(v) for v in values]
            shapes = [t.shape for t in tensors]
            if all(s == shapes[0] for s in shapes):
                collated['listener'][key] = torch.stack(tensors, dim=0)
            else:
                collated['listener'][key] = tensors
        elif isinstance(values[0], (int, float)):
            collated['listener'][key] = torch.tensor(values)
        else:
            collated['listener'][key] = values
    
    collated['metadata'] = [item['metadata'] for item in batch]
    
    return collated


def collate_fn_single(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    item = batch[0]
    
    collated = {'speaker': {}, 'listener': {}, 'metadata': item['metadata']}
    
    for key in item['speaker'].keys():
        value = item['speaker'][key]
        if isinstance(value, (torch.Tensor, np.ndarray)):
            collated['speaker'][key] = torch.as_tensor(value).unsqueeze(0)
        else:
            collated['speaker'][key] = value
    
    for key in item['listener'].keys():
        value = item['listener'][key]
        if isinstance(value, (torch.Tensor, np.ndarray)):
            collated['listener'][key] = torch.as_tensor(value).unsqueeze(0)
        else:
            collated['listener'][key] = value
    
    return collated
