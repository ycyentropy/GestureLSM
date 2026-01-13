import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import librosa
import smplx

# from .build_vocab import Vocab
# from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
# from .utils import other_tools

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type

        self.rank = 0
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplh' in self.args.pose_rep:
            # Seamless使用52个关节点，6D表示 = 312维
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*6)  # 6D表示
            self.joints = len(list(self.tar_joint_list.keys()))
            for joint_name in self.tar_joint_list:
                start_idx = (self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]) * 2  # 6D = 2*3D
                end_idx = self.ori_joint_list[joint_name][1] * 2
                self.joint_mask[start_idx:end_idx] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        # Seamless数据集没有train_test_split.csv，直接遍历目录结构
        self.split_dir = os.path.join(args.data_path, loader_type)
        self.selected_files = []
        self._scan_seamless_directory()

        if len(self.selected_files) == 0:
            logger.warning(f"{loader_type} is empty for directory {self.split_dir}")

        self.data_dir = args.data_path
        self.beatx_during_time = 0

        if loader_type == "test" or loader_type == "val":
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"

        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")

        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)

        # 检查缓存是否存在，不存在则设置为0样本
        if os.path.exists(preloaded_dir):
            self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
            with self.lmdb_env.begin() as txn:
                self.n_samples = txn.stat()["entries"]
        else:
            logger.warning(f"Cache directory {preloaded_dir} does not exist. Please run with build_cache=True first.")
            self.n_samples = 0
            self.lmdb_env = None

        self.norm = True
        # 使用seamless的归一化文件
        self.mean = np.load('./mean_std_seamless/mean_pose.npy')
        self.std = np.load('./mean_std_seamless/std_pose.npy')

        self.trans_mean = np.load('./mean_std_seamless/mean_trans.npy')
        self.trans_std = np.load('./mean_std_seamless/std_trans.npy')



    def _scan_seamless_directory(self):
        """扫描seamless数据集的三层目录结构"""
        if not os.path.exists(self.split_dir):
            logger.error(f"Directory does not exist: {self.split_dir}")
            return

        logger.info(f"Scanning seamless directory: {self.split_dir}")
        for session in sorted(os.listdir(self.split_dir)):
            session_path = os.path.join(self.split_dir, session)
            if os.path.isdir(session_path):
                for gesture in sorted(os.listdir(session_path)):
                    gesture_path = os.path.join(session_path, gesture)
                    if os.path.isdir(gesture_path):
                        for npz_file in sorted(os.listdir(gesture_path)):
                            if npz_file.endswith('.npz'):
                                full_path = os.path.join(gesture_path, npz_file)
                                self.selected_files.append(full_path)

        logger.info(f"Found {len(self.selected_files)} NPZ files in {self.split_dir}")

    def calculate_mean_velocity(self, save_path):
        """计算seamless数据集的52个关节点的平均速度"""
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/",
            model_type='smplh',        # 使用SMPLH模型
            gender='neutral',
            use_face_contour=False,    # 关闭面部轮廓
            num_betas=300,             # 10维形状参数 
            num_expression_coeffs=100, #10 
            ext='pkl',                # 使用PKL格式
            use_pca=False,
        ).cuda().eval()

        all_list = []
        from tqdm import tqdm
        for npz_file in tqdm(self.selected_files, desc="Calculating mean velocity"):
            try:
                m_data = np.load(npz_file, allow_pickle=True)

                # 从seamless数据格式中提取姿态参数
                global_orient = m_data["smplh:global_orient"]  # [N, 3]
                body_pose = m_data["smplh:body_pose"].reshape(-1, 63)          # [N, 21, 3] -> [N, 63]
                left_hand_pose = m_data["smplh:left_hand_pose"].reshape(-1, 45)  # [N, 15, 3] -> [N, 45]
                right_hand_pose = m_data["smplh:right_hand_pose"].reshape(-1, 45) # [N, 15, 3] -> [N, 45]
                trans = m_data["smplh:translation"]           # [N, 3]
                
                # 将平移数据从厘米转换为米
                trans = trans / 100.0
                
                # 初始化有效性标记
                N = global_orient.shape[0]
                is_valid = np.ones(N, dtype=bool)
                
                # 如果存在smplh:is_valid键，使用该键的标记
                if "smplh:is_valid" in m_data:
                    is_valid &= m_data["smplh:is_valid"]
                
                # 必须对平移数据进行异常值检测
                # 使用新的阈值检测方法，范围在-1.0到1.0米之间
                trans_is_valid = self.detect_translation_outliers_threshold(trans)
                is_valid &= trans_is_valid
                
                n_frames = global_orient.shape[0]

                # 组装完整的姿态向量 (156维)
                poses = np.concatenate([
                    global_orient,      # 3维 (全局方向)
                    body_pose,         # 63维 (21身体关节 × 3)
                    left_hand_pose,    # 45维 (15左手关节 × 3)
                    right_hand_pose    # 45维 (15右手关节 × 3)
                ], axis=1)             # 总计156维

                # 扩展betas到所有帧
                betas = np.tile(betas, (n_frames, 1))

                # 转换为torch tensor
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses).cuda().float()
                trans = torch.from_numpy(trans).cuda().float()

                max_length = 128
                s, r = n_frames//max_length, n_frames%max_length

                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        # 使用SMPLH模型计算关节点位置
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length],
                            transl=trans[i*max_length:(i+1)*max_length],
                            body_pose=poses[i*max_length:(i+1)*max_length, 3:66],
                            global_orient=poses[i*max_length:(i+1)*max_length,:3],
                            left_hand_pose=poses[i*max_length:(i+1)*max_length, 66:111],
                            right_hand_pose=poses[i*max_length:(i+1)*max_length, 111:156],
                            return_verts=True,
                            return_joints=True,
                        )['joints'][:, :52, :].reshape(max_length, 52*3)  # 只要52个关节点
                    all_tensor.append(joints)

                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r],
                            transl=trans[s*max_length:s*max_length+r],
                            body_pose=poses[s*max_length:s*max_length+r, 3:66],
                            global_orient=poses[s*max_length:s*max_length+r,:3],
                            left_hand_pose=poses[s*max_length:s*max_length+r, 66:111],
                            right_hand_pose=poses[s*max_length:s*max_length+r, 111:156],
                            return_verts=True,
                            return_joints=True,
                        )['joints'][:, :52, :].reshape(r, 52*3)
                    all_tensor.append(joints)

                if all_tensor:
                    joints = torch.cat(all_tensor, axis=0)
                    joints = joints.permute(1, 0)
                    dt = 1/30

                    # 计算速度
                    init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                    middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n_frames, 52, 3)

                    vel_seq_np = vel_seq.cpu().numpy()
                    vel_joints_np = np.linalg.norm(vel_seq_np, axis=2) # n * 52
                    all_list.append(vel_joints_np)

            except Exception as e:
                logger.warning(f"Error processing {npz_file}: {e}")
                continue

        if all_list:
            avg_vel = np.mean(np.concatenate(all_list, axis=0), axis=0) # 52
            np.save(save_path, avg_vel)
            logger.info(f"Saved mean velocity for 52 joints to {save_path}")
        else:
            logger.error("No valid velocity data computed")

    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        cache_needs_rebuild = False
        if os.path.exists(preloaded_dir):
            # Check if cache is empty
            import lmdb
            try:
                env = lmdb.open(preloaded_dir, readonly=True, lock=False)
                with env.begin() as txn:
                    entries = txn.stat()["entries"]
                env.close()
                if entries == 0:
                    logger.info(f"Cache exists but empty (0 entries), rebuilding: {preloaded_dir}")
                    shutil.rmtree(preloaded_dir)
                    cache_needs_rebuild = True
                else:
                    logger.info("Found the cache {}".format(preloaded_dir))
            except Exception as e:
                logger.warning(f"Error reading cache, rebuilding: {e}")
                shutil.rmtree(preloaded_dir)
                cache_needs_rebuild = True

        # Build cache if needed (not exists or needs rebuild)
        if not os.path.exists(preloaded_dir) or cache_needs_rebuild:
            if self.loader_type == "test" or self.loader_type == "val":
                self.cache_generation(
                    preloaded_dir, True,
                    0, 0,
                    is_test=True)
            else:
                self.cache_generation(
                    preloaded_dir, self.args.disable_filtering,
                    self.args.clean_first_seconds, self.args.clean_final_seconds,
                    is_test=False)
        logger.info(f"BEATX during time is {self.beatx_during_time}s !")

    def __len__(self):
        return self.n_samples

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        # 为seamless数据集分配500GB空间
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 500))  # 500GB
        n_filtered_out = defaultdict(int)

        for npz_file in self.selected_files:
            try:
                pose_each_file = []
                trans_each_file = []
                trans_v_each_file = []
                shape_each_file = []
                facial_each_file = []
                vid_each_file = []

                # 从文件路径提取ID
                relative_path = os.path.relpath(npz_file, self.data_dir)
                id_pose = relative_path.replace('/', '_').replace('.npz', '')

                logger.info(colored(f"# ---- Building cache for {id_pose} ---- #", "blue"))

                # 加载seamless数据格式
                pose_data = np.load(npz_file, allow_pickle=True)

                # 确保数据是30FPS的倍数
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)

                # 提取姿态参数
                global_orient_full = pose_data["smplh:global_orient"]  # [N, 3]
                body_pose_full = pose_data["smplh:body_pose"]  # [N, 21, 3]
                left_hand_pose_full = pose_data["smplh:left_hand_pose"]  # [N, 15, 3]
                right_hand_pose_full = pose_data["smplh:right_hand_pose"]  # [N, 15, 3]
                # translation_full = pose_data["smplh:translation"]  # [N, 3] 暂时不用
                translation_full = np.zeros_like(pose_data["smplh:translation"])  # [N, 3]
                
                # # 将平移数据从厘米转换为米
                # translation_full = translation_full / 100.0 暂时不用
                
                # 初始化有效性标记
                N = len(translation_full)
                is_valid = np.ones(N, dtype=bool)
                
                # 如果存在smplh:is_valid键，使用该键的标记
                if "smplh:is_valid" in pose_data:
                    is_valid &= pose_data["smplh:is_valid"]
                
                # # 必须对平移数据进行异常值检测
                # # 使用新的阈值检测方法，范围在-1.0到1.0米之间
                # trans_is_valid = self.detect_translation_outliers_threshold(translation_full) 暂时不用
                # is_valid &= trans_is_valid
                
                # 采样处理
                global_orient = global_orient_full[::stride]
                body_pose = body_pose_full[::stride].reshape(-1, 63)
                left_hand_pose = left_hand_pose_full[::stride].reshape(-1, 45)
                right_hand_pose = right_hand_pose_full[::stride].reshape(-1, 45)
                translation = translation_full[::stride]
                # Seamless数据集没有betas字段，使用零向量
                betas = np.zeros((1, 300), dtype=np.float32)                   # [1, 10] 改成300

                # 组装完整的姿态向量 (156维)
                poses = np.concatenate([
                    global_orient,      # 3维
                    body_pose,         # 63维
                    left_hand_pose,    # 45维
                    right_hand_pose    # 45维
                ], axis=1)             # 总计156维

                # 应用关节掩码
                masked_poses = poses * self.joint_mask[:156]  # 只取前156维的掩码
                pose_each_file = masked_poses[:, self.joint_mask[:156].astype(bool)]

                # 转换为6D表示
                pose_each_file_reshaped = pose_each_file.reshape(-1, 52, 3)  # 52个关节点
                pose_tensor = torch.from_numpy(pose_each_file_reshaped).float()
                pose_matrix = rc.axis_angle_to_matrix(pose_tensor)
                pose_6d = rc.matrix_to_rotation_6d(pose_matrix)
                pose_each_file_6d = pose_6d.reshape(-1, 52*6).numpy()  # 312维

                pose_each_file_6d_padded = np.zeros((pose_each_file_6d.shape[0], 330))  # 330维
                pose_each_file_6d_padded[:, :22*6] = pose_each_file_6d[:, :22*6]  # 前22个关节点
                pose_each_file_6d_padded[:, 25*6:] = pose_each_file_6d[:, 22*6:]  # 后30个关节点
                pose_each_file_6d = pose_each_file_6d_padded  # 330维

                self.beatx_during_time += pose_each_file_6d.shape[0]/30

                # 处理平移数据
                trans_each_file = translation.copy()
                # trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0] #暂时不用
                # trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
                trans_v_each_file = np.zeros_like(trans_each_file)
                # trans_v_each_file[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0] 暂时不用
                # trans_v_each_file[0,0] = trans_v_each_file[1,0]
                # trans_v_each_file[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
                # trans_v_each_file[0,2] = trans_v_each_file[1,2]
                # trans_v_each_file[:,1] = trans_each_file[:,1]

                # 形状参数
                shape_each_file = np.repeat(betas, pose_each_file_6d.shape[0], axis=0)

                # 面部参数 (seamless数据可能有面部表情，设为空)
                facial_each_file = np.zeros((pose_each_file_6d.shape[0], 100))

                # 视频ID (从路径中提取)
                session_id = int(os.path.basename(os.path.dirname(npz_file)))
                vid_each_file = np.repeat(np.array(session_id).reshape(1, 1), pose_each_file_6d.shape[0], axis=0)

                filtered_result = self._sample_from_clip(
                    dst_lmdb_env,
                    pose_each_file_6d, trans_each_file, trans_v_each_file, shape_each_file, facial_each_file,
                    vid_each_file,
                    disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                    )
                for type in filtered_result.keys():
                    n_filtered_out[type] += filtered_result[type]

            except Exception as e:
                logger.warning(f"Error processing {npz_file}: {e}")
                continue

        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def _sample_from_clip(
        self, dst_lmdb_env, pose_each_file, trans_each_file, trans_v_each_file, shape_each_file, facial_each_file,
        vid_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """为多长度训练采样数据片段"""

        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps

        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps

        n_filtered_out = defaultdict(int)

        # 多长度训练核心逻辑
        for ratio in self.args.multi_length_training:
            if is_test:
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)

            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length} (ratio {ratio})")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")

            sample_pose_list = []
            sample_face_list = []
            sample_shape_list = []
            sample_vid_list = []
            sample_trans_list = []
            sample_trans_v_list = []

            for i in range(num_subdivision):
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length
                sample_pose = pose_each_file[start_idx:fin_idx]
                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_trans_v = trans_v_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                sample_face = facial_each_file[start_idx:fin_idx]
                sample_vid = vid_each_file[start_idx:fin_idx]

                if sample_pose.any() != None:
                    # 过滤运动骨架数据
                    # sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    # is_correct_motion = (sample_pose is not None)
                    is_correct_motion = True
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_shape_list.append(sample_shape)
                        sample_vid_list.append(sample_vid)
                        sample_face_list.append(sample_face)
                        sample_trans_list.append(sample_trans)
                        sample_trans_v_list.append(sample_trans_v)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, shape, face, vid, trans, trans_v in zip(
                        sample_pose_list,
                        sample_shape_list,
                        sample_face_list,
                        sample_vid_list,
                        sample_trans_list,
                        sample_trans_v_list,
                        ):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose, shape, face, vid, trans, trans_v]
                        v = pickle.dumps(v, 5)
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        if self.lmdb_env is None:
            raise RuntimeError("LMDB cache not available. Please build cache first.")

        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            tar_pose, in_shape, tar_face, vid, trans, trans_v = sample
            tar_pose = torch.from_numpy(tar_pose).float()
            tar_face = torch.from_numpy(tar_face).float()

            if self.norm:
                tar_pose = (tar_pose - self.mean) / self.std
                # trans_v = (trans_v - self.trans_mean) / self.trans_std 暂时不用

            if self.loader_type == "test" or self.loader_type == "val":
                tar_pose = tar_pose.float()
                trans = torch.from_numpy(trans).float()
                trans_v = torch.from_numpy(trans_v).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
                tar_pose = torch.cat([tar_pose, trans_v], dim=1)
                tar_pose = torch.cat([tar_pose, tar_face], dim=1)
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                trans_v = torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = tar_pose.reshape((tar_pose.shape[0], -1)).float()
                tar_pose = torch.cat([tar_pose, trans_v], dim=1)
                tar_pose = torch.cat([tar_pose, tar_face], dim=1)
            return tar_pose
    
    def detect_translation_outliers(self, translation, iqr_factor=1.0):
        """
        基于平移数据本身使用IQR方法检测异常值
        
        Args:
            translation: 平移数据，形状为 (N, 3)
            iqr_factor: IQR倍数，值越小检测越严格（默认1.0，标准值为1.5）
            
        Returns:
            is_valid: 有效性标记，形状为 (N,)
        """
        N, _ = translation.shape
        is_valid = np.ones(N, dtype=bool)
        
        # 对每个坐标轴分别检测异常值
        for axis in range(3):
            data = translation[:, axis]
            
            # 计算IQR
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            # 定义异常值阈值（更严格的检测，使用更小的IQR倍数）
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            # 检测异常值
            axis_is_valid = (data >= lower_bound) & (data <= upper_bound)
            is_valid &= axis_is_valid
        
        return is_valid
    
    def detect_translation_outliers_threshold(self, translation, lower_bound=-1.0, upper_bound=1.0):
        """
        基于简单阈值检测平移数据中的异常值
        
        Args:
            translation: 平移数据，形状为 (N, 3)（单位：米）
            lower_bound: 正常值下限（默认-1.0米）
            upper_bound: 正常值上限（默认1.0米）
            
        Returns:
            is_valid: 有效性标记，形状为 (N,)
        """
        N, _ = translation.shape
        is_valid = np.ones(N, dtype=bool)
        
        # 对每个坐标轴分别检测异常值
        for axis in range(3):
            data = translation[:, axis]
            
            # 检测异常值，必须在-100到100之间
            axis_is_valid = (data >= lower_bound) & (data <= upper_bound)
            is_valid &= axis_is_valid
        
        return is_valid
    
    # def smooth_outliers(self, data, is_valid):
    #     """
    #     对异常值区间进行平滑处理，使用区间两端的正常值进行插值
        
    #     Args:
    #         data: 原始数据，形状为 (N, D)
    #         is_valid: 有效性标记，形状为 (N,)
            
    #     Returns:
    #         平滑后的数据，形状为 (N, D)
    #     """
    #     N, D = data.shape
    #     smoothed_data = data.copy()
        
    #     # 找到所有异常帧的索引
    #     invalid_indices = np.where(~is_valid)[0]
        
    #     if len(invalid_indices) == 0:
    #         return smoothed_data
        
    #     # 将异常帧分组为连续区间
    #     invalid_groups = []
    #     current_group = [invalid_indices[0]]
        
    #     for i in range(1, len(invalid_indices)):
    #         if invalid_indices[i] == current_group[-1] + 1:
    #             current_group.append(invalid_indices[i])
    #         else:
    #             invalid_groups.append(current_group)
    #             current_group = [invalid_indices[i]]
    #     invalid_groups.append(current_group)
        
    #     # 对每个异常区间进行平滑处理
    #     for group in invalid_groups:
    #         start_idx = group[0]
    #         end_idx = group[-1]
            
    #         # 找到区间前的最后一个正常帧和区间后的第一个正常帧
    #         prev_valid = start_idx - 1 if start_idx > 0 else 0
    #         next_valid = end_idx + 1 if end_idx < N - 1 else N - 1
            
    #         # 如果整个序列都是异常的，跳过
    #         if prev_valid == next_valid:
    #             continue
            
    #         # 获取前后正常帧的数据
    #         prev_data = data[prev_valid]
    #         next_data = data[next_valid]
            
    #         # 计算插值因子
    #         group_length = end_idx - start_idx + 1
    #         for i, idx in enumerate(group):
    #             # 线性插值
    #             alpha = (i + 1) / (group_length + 1)
    #             smoothed_data[idx] = (1 - alpha) * prev_data + alpha * next_data
        
    #     return smoothed_data

    def smooth_outliers(self, data, is_valid):
        """
        对异常值区间进行平滑处理，使用区间两端的正常值进行插值
        
        Args:
            data: 原始数据，形状为 (N, D)
            is_valid: 有效性标记，形状为 (N,)
            
        Returns:
            平滑后的数据，形状为 (N, D)
        """
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
        
        return smoothed_data

class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)
        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx*6:(joint_idx+1)*6]  # 6D表示
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        # 手腕关节索引 (需要根据seamless关节配置调整)
        left_arm_var = get_variance(self.skeletons, 20)  # 左手腕 这里不对
        right_arm_var = get_variance(self.skeletons, 41)  # 右手腕

        th = 0.0014
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False

    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        # 从6D表示恢复3D位置来计算脊柱角度 (简化版本)
        angles = []
        for i in range(self.skeletons.shape[0]):
            # 假设脊柱是前几个关节点
            if len(self.skeletons[i]) >= 12:  # 至少有2个关节点的6D表示
                spine_vec = self.skeletons[i, 6:12] - self.skeletons[i, 0:6]
                angle = angle_between(spine_vec, [0, -1, 0, 0, -1, 0])  # 6D版本
                angles.append(angle)

        if angles and (np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20):
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles) if angles else 0))
            return False