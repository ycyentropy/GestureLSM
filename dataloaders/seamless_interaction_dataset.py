import os
import pickle
import math
import re
import shutil
import numpy as np
import lmdb as lmdb
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle
import smplx
from tqdm import tqdm
# Import Vocab class first to ensure it's available when unpickling
from dataloaders.build_vocab import Vocab
from .utils.audio_features import process_audio_data
from .data_tools import joints_list
from .utils.other_tools import MultiLMDBManager
from .utils.motion_rep_transfer import process_smplx_motion
from .utils.mis_features import process_semantic_data, process_emotion_data
from .utils.seamless_text_features import process_word_data
from .utils.data_sample import sample_from_clip
import time


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        
        # Set rank safely - handle cases where distributed training is not yet initialized
        try:
            if torch.distributed.is_initialized():
                self.rank = torch.distributed.get_rank()
            else:
                self.rank = 0
        except:
            self.rank = 0
        
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        
        # Initialize basic parameters
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0]  # for trinity
        
        """Initialize SMPLX model."""
        # Try to load SMPLX model, but skip if not found (for testing purposes)
        self.smplx = None
        try:
            # Use the correct model path
            smplx_model_path = os.path.join(self.args.root_path, "datasets/hub/smplx_models/")
            self.smplx = smplx.create(
                smplx_model_path,
                model_type='smplx',
                gender='NEUTRAL_2020', 
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100, 
                ext='npz',
                use_pca=False,
            )
            # Only move to CUDA if available
            if torch.cuda.is_available():
                self.smplx = self.smplx.cuda().eval()
            else:
                self.smplx = self.smplx.eval()
            logger.info("SMPLX model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load SMPLX model: {e}")
            logger.warning("Continuing without SMPLX model.")
        
        # Load language model if needed and available
        self.lang_model = None
        if self.args.word_rep is not None:
            vocab_path = "/home/embodied/yangchenyu/GestureLSM/datasets/unified_vocab.pkl"
            logger.info(f"Attempting to load language model from: {vocab_path}")
            if os.path.exists(vocab_path):
                try:
                    with open(vocab_path, 'rb') as f:
                        self.lang_model = pickle.load(f)
                    logger.info(f"✓ Language model loaded successfully")
                except (AttributeError, ImportError) as e:
                    logger.warning(f"Failed to load language model: {e}")
                    logger.warning("Continuing without language model. Text processing may be disabled.")
            else:
                logger.warning(f"Vocab file not found: {vocab_path}")
                logger.warning("Continuing without language model. Text processing may be disabled.")
        logger.info(f"Word representation: {self.args.word_rep}, Language model: {'Available' if self.lang_model is not None else 'Not available'}")
        
        # Initialize data directories and lengths
        self._init_data_paths()

        # Process split rules for ID filtering
        self._process_split_rules()

        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        
        # Build or load cache
        self._init_cache(build_cache)
    
    def _init_data_paths(self):
        """Initialize data directories and lengths for seamless_interaction dataset."""
        self.data_dir = self.args.data_path
        
        if self.loader_type == "test" or self.loader_type == "val":
            self.args.multi_length_training = [1.0]
            
        self.max_length = int(self.args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(self.args.pose_length / self.args.pose_fps * self.args.audio_sr)
        
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr
        
        if self.args.test_clip and self.loader_type == "test":
            self.preloaded_dir = self.args.root_path + self.args.cache_path + self.loader_type + "_clip" + f"/{self.args.pose_rep}_cache"
        else:
            self.preloaded_dir = self.args.root_path + self.args.cache_path + self.loader_type + f"/{self.args.pose_rep}_cache"
    
    def _init_cache(self, build_cache):
        """Initialize or build cache."""
        self.lmdb_envs = {}
        self.mapping_data = None
        
        if build_cache and self.rank == 0:
            self.build_cache(self.preloaded_dir)
        
        # In DDP mode, ensure all processes wait for cache building to complete
        if build_cache and torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        
        # Try to regenerate cache if corrupted (only on rank 0 to avoid race conditions)
        if self.rank == 0:
            self.regenerate_cache_if_corrupted()
        
        # Wait for cache regeneration to complete
        if build_cache and torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        
        self.load_db_mapping()
    
    def _process_split_rules(self):
        """Process dataset split rules based on file names."""
        # Get all npz files in the dataset directory
        data_path = os.path.join(self.data_dir, self.loader_type)
        all_npz_files = glob.glob(os.path.join(data_path, "**/*.npz"), recursive=True)
        
        # Skip filtering if training_speakers is not provided or is empty
        if not hasattr(self.args, 'training_speakers') or not self.args.training_speakers:
            logger.info(f"No training_speakers provided, using all {len(all_npz_files)} files")
            self.filtered_files = all_npz_files
            return
        
        # Extract IDs from filenames and filter based on training_speakers
        self.filtered_files = []
        for npz_file in all_npz_files:
            # Extract ID from filename using correct regex
            # Format: V00_S0039_I00000581_P0061A.npz -> speaker ID: 61
            speaker_id_match = re.search(r'P(\d+)[A-Za-z]*\.npz$', npz_file)
            if speaker_id_match:
                file_id = int(speaker_id_match.group(1))
                # Check if ID is in the allowed list
                if file_id in self.args.training_speakers:
                    self.filtered_files.append(npz_file)
        
        logger.info(f"Filtered {len(self.filtered_files)} files from {len(all_npz_files)} total files based on ID")
        
        # Handle empty selection
        if not self.filtered_files:
            logger.warning(f"{self.loader_type} is empty for speakers {self.args.training_speakers}, using all files instead")
            self.filtered_files = all_npz_files
    
    def build_cache(self, preloaded_dir):
        """Build the dataset cache."""
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        
        if self.args.new_cache and os.path.exists(preloaded_dir):
            shutil.rmtree(preloaded_dir)
            
        if os.path.exists(preloaded_dir):
            # if the dir is empty, that means we still need to build the cache
            if not os.listdir(preloaded_dir):
                self.cache_generation(
                    preloaded_dir, 
                    self.args.disable_filtering,
                    self.args.clean_first_seconds,
                    self.args.clean_final_seconds,
                    is_test=False
                )
            else:
                logger.info("Found the cache {}".format(preloaded_dir))

        elif self.loader_type == "test" or self.loader_type == "val":
            self.cache_generation(preloaded_dir, True, 0, 0, is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, 
                self.args.disable_filtering,
                self.args.clean_first_seconds,
                self.args.clean_final_seconds,
                is_test=False
            )
    
    def regenerate_cache_if_corrupted(self):
        """Regenerate cache if the pickle file is corrupted."""
        mapping_path = os.path.join(self.preloaded_dir, "sample_db_mapping.pkl")
        
        if os.path.exists(mapping_path):
            try:
                # Try to load the file to check if it's corrupted
                with open(mapping_path, 'rb') as f:
                    test_data = pickle.load(f)
                return False  # File is not corrupted
            except (EOFError, pickle.UnpicklingError):
                print(f"Detected corrupted pickle file: {mapping_path}")
                print("Regenerating cache...")
                
                # Remove corrupted file
                os.remove(mapping_path)
                
                # Regenerate cache
                self.build_cache(self.preloaded_dir)
                return True
        
        return False
    
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds, clean_final_seconds, is_test=False):
        """Generate cache for the dataset."""
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        
        # Initialize the multi-LMDB manager
        lmdb_manager = MultiLMDBManager(out_lmdb_dir, max_db_size=100*1024*1024*1024)
        
        self.n_out_samples = 0
        file_num = 0
        n_filtered_out = defaultdict(int)
        
        # Performance tracking
        total_start_time = time.time()
        file_times = []
        
        # Get npz files based on filtering
        data_path = os.path.join(self.data_dir, self.loader_type)
        if hasattr(self, 'filtered_files') and self.filtered_files:
            npz_files = self.filtered_files
        else:
            npz_files = glob.glob(os.path.join(data_path, "**/*.npz"), recursive=True)
        
        # For test or val data, only process first 10 samples
        if self.loader_type == "test" or self.loader_type == "val":
            npz_files = npz_files[:10]
            logger.info(f"Processing first {len(npz_files)} files for {self.loader_type} split")
        else:
            logger.info(f"Found {len(npz_files)} npz files in {data_path}")
        
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        
        # Add progress bar
        pbar = tqdm(npz_files, desc=f"Processing {self.loader_type} files", unit="file")
        
        for pose_file in pbar:
            file_start_time = time.time()
            file_num += 1
            f_name = os.path.splitext(os.path.basename(pose_file))[0]
            
            # Update progress bar description with current file
            pbar.set_postfix({"current": f_name[:20]})
            
            # Process data - now returns list of segment data
            all_segment_data = self._process_file_data(f_name, pose_file, ".npz")
            if all_segment_data is None or len(all_segment_data) == 0:
                file_times.append(time.time() - file_start_time)
                continue
            
            # Process each valid segment separately
            for segment_idx, data in enumerate(all_segment_data):
                # Sample from clip for current segment
                filtered_result, self.n_out_samples = sample_from_clip(
                    lmdb_manager=lmdb_manager,
                    audio_file=pose_file.replace(".npz", ".wav"),
                    audio_each_file=data['audio'],
                    pose_each_file=data['pose'],
                    trans_each_file=data['trans'],
                    trans_v_each_file=data['trans_v'],
                    shape_each_file=data['shape'],
                    facial_each_file=data['facial'],
                    word_each_file=data['word'],
                    vid_each_file=data['vid'],
                    emo_each_file=data['emo'],
                    sem_each_file=data['sem'],
                    args=self.args,
                    ori_stride=self.ori_stride,
                    ori_length=self.ori_length,
                    disable_filtering=disable_filtering,
                    clean_first_seconds=clean_first_seconds,
                    clean_final_seconds=clean_final_seconds,
                    is_test=is_test,
                    n_out_samples=self.n_out_samples
                )
                
                for type_key in filtered_result:
                    n_filtered_out[type_key] += filtered_result[type_key]
            
            file_times.append(time.time() - file_start_time)
            
            # Update progress bar with average processing time
            if len(file_times) > 0:
                avg_time = sum(file_times) / len(file_times)
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s", "current": f_name[:20]})
        
        pbar.close()
        
        # Print performance statistics
        total_time = time.time() - total_start_time
        avg_file_time = sum(file_times) / len(file_times) if file_times else 0
        logger.info(f"Cache generation completed for {self.loader_type}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.info(f"Average time per file: {avg_file_time:.2f}s")
        logger.info(f"Files processed: {len(file_times)}/{len(npz_files)}")
        logger.info(f"Total filtered out: {self.n_out_samples}")
        
        lmdb_manager.close()
    
    def _process_file_data(self, f_name, pose_file, ext):
        """Process all data for a single file."""
        start_time = time.time()
        
        data = {
            'pose': None, 'trans': None, 'trans_v': None, 'shape': None,
            'audio': None, 'facial': None, 'word': None, 'word_valid': None,
            'emo': None, 'sem': None, 'vid': None
        }
        
        # Load motion data from npz file (returns complete data with motion_valid)
        motion_data = self._load_seamless_motion_data(pose_file)
        if motion_data is None:
            return None
        
        data.update(motion_data)
        
        # Process speaker ID
        has_id_rep = hasattr(self.args, 'id_rep')
        if has_id_rep and self.args.id_rep is not None:
            # Extract speaker ID from pose_file filename
            # Format: V00_S0039_I00000581_P0061A.npz -> speaker ID: 0061
            speaker_id_match = re.search(r'P(\d+)[A-Za-z]*\.npz$', pose_file)
            if speaker_id_match:
                speaker_id = int(speaker_id_match.group(1))
                # Ensure ID is within valid range (0-4999 for max_id=5000)
                speaker_id = min(speaker_id, 4999)
                # Save integer ID directly, no one-hot encoding
                data['vid'] = np.array([speaker_id])
            else:
                # No speaker ID found, use 0 as general speaker
                data['vid'] = np.array([0])
        else:
            # No ID representation specified, use -1 as placeholder
            data['vid'] = np.array([-1])
        
        # Process audio if needed (load complete audio data)
        if self.args.audio_rep is not None:
            audio_file = pose_file.replace(".npz", ".wav")
            data = process_audio_data(audio_file, self.args, data, f_name, None)
            if data is None:
                return None
        
        # Process text data if needed and language model is available
        if self.args.word_rep is not None:
            if self.lang_model is not None:
                word_file = pose_file.replace(".npz", ".json")
                data = process_word_data(self.args.data_path, word_file, self.args, data, f_name, None, self.lang_model)
                if data is None:
                    return None
            else:
                # If text processing is not possible, set word to default value
                if 'pose' in data and data['pose'] is not None:
                    num_frames = data['pose'].shape[0]
                    data['word'] = np.array([-1] * num_frames)
                    data['word_valid'] = np.array([False] * num_frames)
        
        # Calculate intersection of motion_valid and word_valid
        motion_valid = data.get('motion_valid', np.ones(data['pose'].shape[0], dtype=bool))
        word_valid = data.get('word_valid', np.ones(data['pose'].shape[0], dtype=bool))
        
        # In test mode, set all valid masks to True to skip filtering
        if self.loader_type == "test" or self.loader_type == "val":
            motion_valid = np.ones(data['pose'].shape[0], dtype=bool)
            word_valid = np.ones(data['pose'].shape[0], dtype=bool)
        
        # Calculate final valid mask as intersection
        final_valid = motion_valid & word_valid
        
        # In test mode, skip segmentation and use complete data
        # Controlled by skip_segmentation arg (defaults to True in test/val)
        skip_segmentation = getattr(self.args, 'skip_segmentation', True)
        if (self.loader_type == "test" or self.loader_type == "val") and skip_segmentation:
            all_segment_data = []
            segment_data = {}
            
            # Use complete data without filtering
            segment_data['pose'] = data['pose']
            segment_data['trans'] = data['trans']
            segment_data['trans_v'] = data['trans_v']
            segment_data['motion_valid'] = np.ones(data['pose'].shape[0], dtype=bool)
            
            # Use complete audio data
            if data['audio'] is not None and data['audio'].size > 1:
                segment_data['audio'] = data['audio']
            else:
                segment_data['audio'] = None
            
            # Use complete word data
            if data['word'] is not None and data['word'].size > 1:
                segment_data['word'] = data['word']
                segment_data['word_valid'] = np.ones(data['word'].shape[0], dtype=bool)
            else:
                segment_data['word'] = None
                segment_data['word_valid'] = None
            
            # Process other data
            segment_data['emo'] = np.array([-1])
            segment_data['sem'] = np.array([-1])
            segment_data['vid'] = data['vid']
            segment_data['shape'] = data['shape']
            segment_data['facial'] = data['facial']
            
            all_segment_data.append(segment_data)
            
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                logger.debug(f"Processed {f_name} in {elapsed_time:.2f}s, {len(all_segment_data)} segments")
            
            return all_segment_data
        
        # Find continuous valid segments from the intersection and filter by duration
        min_duration_threshold = 5.0
        valid_segments = self.find_continuous_valid_segments(final_valid, min_duration_threshold)
        
        if len(valid_segments) == 0:
            return None
        
        # 为每个满足条件的区间创建独立的数据副本
        all_segment_data = []
        
        for selected_segment in valid_segments:
            final_start_idx, final_end_idx = selected_segment
            
            # 创建当前区间的数据副本
            segment_data = {}
            
            # Filter motion data based on current valid segment
            segment_data['pose'] = data['pose'][final_start_idx:final_end_idx+1]
            segment_data['trans'] = data['trans'][final_start_idx:final_end_idx+1]
            segment_data['trans_v'] = data['trans_v'][final_start_idx:final_end_idx+1]
            segment_data['motion_valid'] = data['motion_valid'][final_start_idx:final_end_idx+1]
            
            # Filter audio data based on current valid segment
            if data['audio'] is not None and data['audio'].size > 1:
                audio_start_idx = int(final_start_idx * self.args.audio_fps / self.args.pose_fps)
                audio_end_idx = int(final_end_idx * self.args.audio_fps / self.args.pose_fps)
                
                audio_start_idx = max(0, audio_start_idx)
                audio_end_idx = min(data['audio'].shape[0], audio_end_idx)
                
                expected_audio_length = int((final_end_idx - final_start_idx + 1) * self.args.audio_fps / self.args.pose_fps)
                actual_audio_length = audio_end_idx - audio_start_idx
                
                min_audio_frames = int(0.5 * self.args.audio_fps)
                if actual_audio_length >= min_audio_frames:
                    segment_data['audio'] = data['audio'][audio_start_idx:audio_end_idx+1]
                else:
                    continue
            else:
                segment_data['audio'] = None
            
            # Filter word and word_valid data based on current valid segment
            if data['word'] is not None and data['word'].size > 1:
                segment_data['word'] = data['word'][final_start_idx:final_end_idx+1]
                
                if data['word_valid'] is not None:
                    segment_data['word_valid'] = data['word_valid'][final_start_idx:final_end_idx+1]
                else:
                    segment_data['word_valid'] = None
            else:
                segment_data['word'] = None
                segment_data['word_valid'] = None
            
            # Process emotion data (not used for seamless dataset)
            segment_data['emo'] = np.array([-1])
            
            # Process semantic data (not used for seamless dataset)
            segment_data['sem'] = np.array([-1])
            
            # Process speaker ID
            segment_data['vid'] = data['vid'] #[final_start_idx:final_end_idx+1]
            
            # Copy shape and facial data
            segment_data['shape'] = data['shape']
            segment_data['facial'] = data['facial']
            
            all_segment_data.append(segment_data)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            logger.debug(f"Processed {f_name} in {elapsed_time:.2f}s, {len(all_segment_data)} segments")
        
        return all_segment_data
    
    def _load_seamless_motion_data(self, pose_file):
        """Load and process motion data from seamless interaction dataset npz files."""
        try:
            # Load the npz file
            pose_data = np.load(pose_file, allow_pickle=True)
            
            # Extract motion data from the npz file
            # For seamless dataset, the structure is:
            # - smplh:translation: global translation
            # - smplh:global_orient: global orientation (root rotation)
            # - smplh:body_pose: body joint angles (21 joints, 3 angles each)
            # - smplh:left_hand_pose: left hand joint angles (15 joints, 3 angles each)
            # - smplh:right_hand_pose: right hand joint angles (15 joints, 3 angles each)
            
            # Extract individual components
            trans = pose_data['smplh:translation']
            
            # 将平移数据从厘米转换为米
            trans = trans / 100.0
            
            global_orient = pose_data['smplh:global_orient']
            body_pose = pose_data['smplh:body_pose']
            left_hand_pose = pose_data['smplh:left_hand_pose']
            right_hand_pose = pose_data['smplh:right_hand_pose']
            
            # Initialize validity标记
            N = trans.shape[0]
            is_valid = np.ones(N, dtype=bool)
            
            # 如果存在smplh:is_valid键，使用该键的标记
            if "smplh:is_valid" in pose_data:
                is_valid &= pose_data["smplh:is_valid"]
            
            # 必须对平移数据进行异常值检测
            # 使用新的阈值检测方法，范围在-100到100之间
            # trans_is_valid = self.detect_translation_outliers_threshold(trans) 暂时不用
            # is_valid &= trans_is_valid
            
            # Calculate number of frames (all frames, not just valid segments)
            num_frames = trans.shape[0]
            
            # Create full pose array with 55 joints (3 angles each = 165 values per frame)
            # SMPL-X full pose structure:
            # - 0-2: global_orient (1 joint)
            # - 3-65: body_pose (21 joints)
            # - 66-68: jaw_pose (1 joint) - MISSING in seamless dataset
            # - 69-71: leye_pose (1 joint) - MISSING in seamless dataset
            # - 72-74: reye_pose (1 joint) - MISSING in seamless dataset
            # - 75-119: left_hand_pose (15 joints)
            # - 120-164: right_hand_pose (15 joints)
            
            # Initialize full pose array with zeros
            full_pose = np.zeros((num_frames, 55 * 3), dtype=np.float32)
            
            # Reshape the components to 2D arrays
            global_orient_2d = global_orient.reshape(num_frames, 3)
            body_pose_2d = body_pose.reshape(num_frames, 21*3)
            left_hand_pose_2d = left_hand_pose.reshape(num_frames, 15*3)
            right_hand_pose_2d = right_hand_pose.reshape(num_frames, 15*3)
            
            # Fill in the available data
            full_pose[:, 0:3] = global_orient_2d  # global_orient (1 joint)
            full_pose[:, 3:3+21*3] = body_pose_2d  # body_pose (21 joints)
            # Jaw, leye, reye are set to zero (missing in seamless dataset)
            full_pose[:, 3+21*3+3+3+3:3+21*3+3+3+3+15*3] = left_hand_pose_2d  # left_hand_pose (15 joints)
            full_pose[:, 3+21*3+3+3+3+15*3:] = right_hand_pose_2d  # right_hand_pose (15 joints)
            
            # Process translations
            # processed_trans = trans.copy() #暂时不用
            # processed_trans[:, 0] = processed_trans[:, 0] - processed_trans[0, 0]
            # processed_trans[:, 2] = processed_trans[:, 2] - processed_trans[0, 2]
            processed_trans = np.zeros_like(trans)
            
            # Calculate translation velocities
            trans_v = np.zeros_like(processed_trans)
            # trans_v[1:, 0] = processed_trans[1:, 0] - processed_trans[:-1, 0] #暂时不用
            # trans_v[0, 0] = trans_v[1, 0]
            # trans_v[1:, 2] = processed_trans[1:, 2] - processed_trans[:-1, 2]
            # trans_v[0, 2] = trans_v[1, 2]
            # trans_v[:, 1] = processed_trans[:, 1]
            
            # Process shape data (use placeholder for seamless dataset, will be filled with zeros when needed)
            shape = np.array([-1])
            
            # Process facial data (not used for seamless dataset, will be filled with zeros when needed)
            facial = np.array([-1])
            
            return {
                'pose': full_pose,
                'trans': processed_trans,
                'trans_v': trans_v,
                'shape': shape,
                'facial': facial,
                'motion_valid': is_valid
            }
        
        except Exception as e:
            logger.error(f"Error processing motion data from {pose_file}: {e}")
            return None
    
    def detect_translation_outliers_threshold(self, translation, lower_bound=-1.0, upper_bound=1.0):
        """
        基于简单阈值检测平移数据中的异常值
        
        Args:
            translation: 平移数据，形状为 (N, 3) 或可能的其他形状（单位：米）
            lower_bound: 正常值下限（默认-1.0米）
            upper_bound: 正常值上限（默认1.0米）
            
        Returns:
            is_valid: 有效性标记，形状为 (N,)
        """
        # 确保translation是2维数据
        if len(translation.shape) > 2:
            # 如果是3维或更高维，将其转换为2维
            translation = translation.reshape(-1, translation.shape[-1])
        
        N, _ = translation.shape
        is_valid = np.ones(N, dtype=bool)
        
        # 对每个坐标轴分别检测异常值
        for axis in range(3):
            data = translation[:, axis]
            
            # 检测异常值，必须在-1.0到1.0米之间
            axis_is_valid = (data >= lower_bound) & (data <= upper_bound)
            is_valid &= axis_is_valid
        
        return is_valid
    
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
                    segment_duration = (end_idx - start_idx + 1) / self.args.pose_fps
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
            
            # 修正：准确查找前一个有效帧（从start_idx往前找第一个有效帧）
            prev_valid = None
            for idx in range(start_idx - 1, -1, -1):
                if is_valid[idx]:
                    prev_valid = idx
                    break
            
            # 修正：准确查找后一个有效帧（从end_idx往后找第一个有效帧）
            next_valid = None
            for idx in range(end_idx + 1, N):
                if is_valid[idx]:
                    next_valid = idx
                    break
            
            # 修正：严谨判断无有效插值基准（前后均无有效帧）
            if prev_valid is None and next_valid is None:
                # 无有效帧可供插值，跳过当前异常区间
                continue
            # 修正：单端有有效帧时，用该端数据填充（而非插值）
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
    
    def load_db_mapping(self):
        """Load database mapping from file."""
        mapping_path = os.path.join(self.preloaded_dir, "sample_db_mapping.pkl")
        backup_path = os.path.join(self.preloaded_dir, "sample_db_mapping_backup.pkl")
        
        # Check if file exists and is readable
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        
        # Check file size to ensure it's not empty
        file_size = os.path.getsize(mapping_path)
        if file_size == 0:
            raise ValueError(f"Mapping file is empty: {mapping_path}")
        
        print(f"Loading mapping file: {mapping_path} (size: {file_size} bytes)")
        
        # Add error handling and retry logic for pickle loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(mapping_path, 'rb') as f:
                    self.mapping_data = pickle.load(f)
                print(f"Successfully loaded mapping data with {len(self.mapping_data.get('mapping', []))} samples")
                break
            except (EOFError, pickle.UnpicklingError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Failed to load pickle file (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"File path: {mapping_path}")
                    
                    # Try backup file if main file is corrupted
                    if os.path.exists(backup_path) and os.path.getsize(backup_path) > 0:
                        print("Trying backup file...")
                        try:
                            with open(backup_path, 'rb') as f:
                                self.mapping_data = pickle.load(f)
                            print(f"Successfully loaded mapping data from backup with {len(self.mapping_data.get('mapping', []))} samples")
                            break
                        except Exception as backup_e:
                            print(f"Backup file also failed: {backup_e}")
                    
                    print("Retrying...")
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    print(f"Error: Failed to load pickle file after {max_retries} attempts: {e}")
                    print(f"File path: {mapping_path}")
                    print("Please check if the file is corrupted or incomplete.")
                    print("You may need to regenerate the cache files.")
                    raise
        
        # Update paths from test to test_clip if needed
        if self.loader_type == "test" and self.args.test_clip:
            updated_paths = []
            for path in self.mapping_data['db_paths']:
                updated_path = path.replace("test/", "test_clip/")
                updated_paths.append(updated_path)
            self.mapping_data['db_paths'] = updated_paths
            
            # In DDP mode, avoid modifying shared files to prevent race conditions
            # Instead, just update the in-memory data
            print(f"Updated test paths for test_clip mode (avoiding file modification in DDP)")
        
        self.n_samples = len(self.mapping_data['mapping'])
    
    def get_lmdb_env(self, db_idx):
        """Get LMDB environment for given database index."""
        if db_idx not in self.lmdb_envs:
            db_path = self.mapping_data['db_paths'][db_idx]
            self.lmdb_envs[db_idx] = lmdb.open(db_path, readonly=True, lock=False)
        return self.lmdb_envs[db_idx]
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        db_idx = self.mapping_data['mapping'][idx]
        lmdb_env = self.get_lmdb_env(db_idx)
        
        with lmdb_env.begin(write=False) as txn:
            key = "{:008d}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            
            tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans, trans_v, audio_name = sample
            
            # Convert data to tensors with appropriate types
            processed_data = self._convert_to_tensors(
                tar_pose, in_audio, in_facial, in_shape, in_word,
                emo, sem, vid, trans, trans_v
            )
            
            processed_data['audio_name'] = audio_name
            return processed_data
    
    def _convert_to_tensors(self, tar_pose, in_audio, in_facial, in_shape, in_word,
                           emo, sem, vid, trans, trans_v):
        """Convert numpy arrays to tensors with appropriate types."""
        data = {
            'emo': torch.from_numpy(emo).int(),
            'sem': torch.from_numpy(sem).float(),
            'audio_onset': torch.from_numpy(in_audio).float(),
            'word': torch.from_numpy(in_word).int()
        }
        
        if self.loader_type == "test":
            # Handle ID tensor consistently with training mode
            if vid.size == 0:
                # Create a placeholder tensor with integer ID 0
                id_tensor = torch.zeros((tar_pose.shape[0],), dtype=torch.long)
            elif vid.size == 1 and vid[0] == -1:
                # Handle placeholder [-1] - use 0 as general speaker
                id_tensor = torch.zeros((tar_pose.shape[0],), dtype=torch.long)
            elif len(vid.shape) == 1:
                # Single integer ID, expand to sequence length
                id_tensor = torch.full((tar_pose.shape[0],), int(vid[0]), dtype=torch.long)
            else:
                # Multiple IDs, use directly as sequence
                id_tensor = torch.from_numpy(vid).long()
                # If sequence length doesn't match, repeat or truncate
                if id_tensor.shape[0] != tar_pose.shape[0]:
                    id_tensor = id_tensor.repeat(tar_pose.shape[0])[:tar_pose.shape[0]]
            
            # Handle shape tensor (use zeros if placeholder)
            if in_shape.size == 1 and in_shape[0] == -1:
                # Create a placeholder tensor with shape (tar_pose.shape[0], 300)
                shape_tensor = torch.zeros((tar_pose.shape[0], 300), dtype=torch.float32)
            else:
                shape_tensor = torch.from_numpy(in_shape).float()
            
            # Handle facial tensor (use zeros if placeholder)
            if in_facial.size == 1 and in_facial[0] == -1:
                # Create a placeholder tensor with shape (tar_pose.shape[0], 100)
                facial_tensor = torch.zeros((tar_pose.shape[0], 100), dtype=torch.float32)
            else:
                facial_tensor = torch.from_numpy(in_facial).float()
            
            data.update({
                'pose': torch.from_numpy(tar_pose).float(),
                'trans': torch.from_numpy(trans).float(),
                'trans_v': torch.from_numpy(trans_v).float(),
                'facial': facial_tensor,
                'id': id_tensor,
                'beta': shape_tensor
            })
        else:
            # Training mode ID handling
            if vid.size == 0:
                # Create a placeholder tensor with integer ID 0
                id_tensor = torch.zeros((tar_pose.shape[0],), dtype=torch.long)
            elif vid.size == 1 and vid[0] == -1:
                # Handle placeholder [-1] - use 0 as general speaker
                id_tensor = torch.zeros((tar_pose.shape[0],), dtype=torch.long)
            elif len(vid.shape) == 1:
                # Single integer ID, expand to sequence length
                id_tensor = torch.full((tar_pose.shape[0],), int(vid[0]), dtype=torch.long)
            else:
                # Multiple IDs, use directly as sequence
                id_tensor = torch.from_numpy(vid).long()
                # If sequence length doesn't match, repeat or truncate
                if id_tensor.shape[0] != tar_pose.shape[0]:
                    id_tensor = id_tensor.repeat(tar_pose.shape[0])[:tar_pose.shape[0]]
            
            # Handle shape tensor (use zeros if placeholder)
            if in_shape.size == 0 or (in_shape.size == 1 and in_shape[0] == -1):
                # Create a placeholder tensor with shape (tar_pose.shape[0], 300)
                shape_tensor = torch.zeros((tar_pose.shape[0], 300), dtype=torch.float32)
            else:
                shape_tensor = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
            
            # Handle facial tensor (use zeros if placeholder)
            if in_facial.size == 0 or (in_facial.size == 1 and in_facial[0] == -1):
                # Create a placeholder tensor with shape (tar_pose.shape[0], 100)
                facial_tensor = torch.zeros((tar_pose.shape[0], 100), dtype=torch.float32)
            else:
                facial_tensor = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            
            data.update({
                'pose': torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float(),
                'trans': torch.from_numpy(trans).reshape((trans.shape[0], -1)).float(),
                'trans_v': torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float(),
                'facial': facial_tensor,
                'id': id_tensor,
                'beta': shape_tensor
            })
        
        return data
