import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loguru import logger
from tqdm import tqdm
import numpy as np
import smplx
import librosa

from models.vq.model import RVQVAE
from utils import rotation_conversions as rc
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders.data_tools import joints_list
from dataloaders import data_tools
from utils import metric


class BidirectionalDyadicTrainer:
    """
    双向生成的Dyadic训练器
    同时训练生成Speaker和Listener的手势
    """
    
    def __init__(self, cfg, model, train_dataset, val_dataset=None):
        """
        Args:
            cfg: 配置对象
            model: BidirectionalDyadicLSM模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集(可选)
        """
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 初始化VQ-VAE模型
        self._init_vqvae_models()
        
        # 初始化归一化参数
        self._init_normalization()
        
        # 初始化关节掩码
        self._init_joint_masks()
        
        # 初始化评估工具
        self._init_evaluation_tools()
        
        # 创建DataLoader
        from dataloaders.bidirectional_dyadic_dataset import collate_fn
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.trainer.num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.trainer.get('val_batch_size', 1),
                shuffle=False,
                num_workers=cfg.trainer.num_workers,
                collate_fn=collate_fn
            )
        
        # 梯度更新模式配置
        self.separate_gradient_update = cfg.trainer.get('separate_gradient_update', False)
        self.gradient_update_mode = cfg.trainer.get('gradient_update_mode', 'alternate')  # 'alternate' 或 'separate_optimizers'

        if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
            # 为Speaker和Listener分别创建优化器
            # 获取共享参数（denoiser和modality_encoder）
            # [修改] 去掉输出头，与单人模型保持一致
            shared_params = list(model.denoiser.parameters()) + list(model.modality_encoder.parameters())
            if model.use_id:
                shared_params += list(model.id_embeddings.parameters())
            
            # Speaker和Listener使用相同的共享参数（去掉输出头）
            speaker_params = shared_params
            listener_params = shared_params
            
            # Speaker优化器
            self.speaker_optimizer = AdamW(
                speaker_params,
                lr=cfg.trainer.lr,
                weight_decay=cfg.trainer.weight_decay
            )

            # Listener优化器
            self.listener_optimizer = AdamW(
                listener_params,
                lr=cfg.trainer.lr,
                weight_decay=cfg.trainer.weight_decay
            )

            self.optimizer = None  # 不使用统一优化器
            logger.info("Using separate optimizers for Speaker and Listener")
        else:
            # 统一优化器（联合训练或交替更新模式）
            self.optimizer = AdamW(
                model.parameters(),
                lr=cfg.trainer.lr,
                weight_decay=cfg.trainer.weight_decay
            )
            self.speaker_optimizer = None
            self.listener_optimizer = None
            if self.separate_gradient_update:
                logger.info(f"Using alternate gradient update mode")

        # 训练参数
        self.max_epochs = cfg.trainer.max_epochs
        self.save_interval = cfg.trainer.save_interval
        self.log_interval = cfg.trainer.log_interval
        self.grad_norm = cfg.trainer.get('grad_norm', 1.0)
        self.val_interval = cfg.trainer.get('val_interval', 5)
        self.max_val_iterations = cfg.trainer.get('max_val_iterations', 50)

        # 早停机制参数
        self.early_stopping_patience = cfg.trainer.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = cfg.trainer.get('early_stopping_min_delta', 0.01)

        # 早停状态
        self.best_fgd = float('inf')
        self.patience_counter = 0
        self.early_stopped = False

        # 学习率调度
        self.lr_scheduler_type = cfg.trainer.get('lr_scheduler', 'cosine')
        self.warmup_epochs = cfg.trainer.get('warmup_epochs', 5)
        self.current_epoch = 0

        # 检查点目录
        self.ckpt_dir = os.path.join("checkpoints", cfg.experiment_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 最佳指标追踪
        self.val_best = {
            "speaker_fgd": {"value": float('inf'), "epoch": 0},
            "listener_fgd": {"value": float('inf'), "epoch": 0},
            "avg_fgd": {"value": float('inf'), "epoch": 0},
            "speaker_bc": {"value": float('-inf'), "epoch": 0},
            "listener_bc": {"value": float('-inf'), "epoch": 0},
        }

        # 创建学习率调度器（必须在 val_best 初始化之后）
        self._init_lr_schedulers()

        logger.info(f"BidirectionalDyadicTrainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if val_dataset:
            logger.info(f"Val batches: {len(self.val_loader)}")

    def _init_lr_schedulers(self):
        """初始化学习率调度器"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

        if self.lr_scheduler_type == 'cosine':
            if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                # 为两个优化器分别创建调度器
                self.speaker_scheduler = CosineAnnealingLR(
                    self.speaker_optimizer,
                    T_max=self.max_epochs,
                    eta_min=1e-6
                )
                self.listener_scheduler = CosineAnnealingLR(
                    self.listener_optimizer,
                    T_max=self.max_epochs,
                    eta_min=1e-6
                )
                logger.info("Using cosine LR scheduler for both optimizers")
            else:
                # 统一优化器
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.max_epochs,
                    eta_min=1e-6
                )
                logger.info("Using cosine LR scheduler")
        elif self.lr_scheduler_type == 'step':
            if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                self.speaker_scheduler = LambdaLR(
                    self.speaker_optimizer,
                    lr_lambda=lambda epoch: 1.0 if epoch < self.max_epochs // 2 else 0.1
                )
                self.listener_scheduler = LambdaLR(
                    self.listener_optimizer,
                    lr_lambda=lambda epoch: 1.0 if epoch < self.max_epochs // 2 else 0.1
                )
                logger.info("Using step LR scheduler for both optimizers")
            else:
                self.scheduler = LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: 1.0 if epoch < self.max_epochs // 2 else 0.1
                )
                logger.info("Using step LR scheduler")
        else:
            logger.warning(f"Unknown LR scheduler type: {self.lr_scheduler_type}, using constant LR")
            self.scheduler = None
            self.speaker_scheduler = None
            self.listener_scheduler = None

    def _step_lr_schedulers(self):
        """更新学习率调度器"""
        if self.lr_scheduler_type == 'cosine':
            if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                self.speaker_scheduler.step()
                self.listener_scheduler.step()
            elif hasattr(self, 'scheduler'):
                self.scheduler.step()
        elif self.lr_scheduler_type == 'step':
            if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                self.speaker_scheduler.step()
                self.listener_scheduler.step()
            elif hasattr(self, 'scheduler'):
                self.scheduler.step()

    def _init_vqvae_models(self):
        """初始化VQ-VAE模型"""
        import argparse
        
        # 创建默认参数
        vq_args = argparse.Namespace(
            num_quantizers=6,
            shared_codebook=False,
            quantize_dropout_prob=0.2,
            mu=0.99,
        )
        
        # 创建三个VQ-VAE模型
        self.vq_model_upper = self._create_rvqvae(vq_args, 78, self.cfg.vqvae_upper_path)
        self.vq_model_hands = self._create_rvqvae(vq_args, 180, self.cfg.vqvae_hands_path)
        
        # Lower body维度取决于是否使用trans
        lower_dim = 54 + (3 if getattr(self.cfg, 'use_trans', False) else 0)
        self.vq_model_lower = self._create_rvqvae(vq_args, lower_dim, self.cfg.vqvae_lower_path)
        
        # 设置为eval模式
        self.vq_model_upper.eval().to(self.device)
        self.vq_model_hands.eval().to(self.device)
        self.vq_model_lower.eval().to(self.device)
        
        logger.info("VQ-VAE models loaded")
    
    def _create_rvqvae(self, vq_args, dim_pose, checkpoint_path):
        """创建单个RVQVAE模型"""
        model = RVQVAE(
            vq_args,
            dim_pose,
            nb_code=1024,
            code_dim=128,
            output_emb_width=128,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None
        )
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')['net']
            model.load_state_dict(state_dict)
            logger.info(f"Loaded VQ-VAE from {checkpoint_path}")
        else:
            logger.warning(f"VQ-VAE checkpoint not found: {checkpoint_path}")
        
        return model
    
    def _init_normalization(self):
        """初始化归一化参数"""
        # 加载均值和标准差
        self.mean = np.load(self.cfg.mean_pose_path)
        self.std = np.load(self.cfg.std_pose_path)
        
        # 转换为tensor并移动到设备
        self.mean_upper = torch.from_numpy(self.mean[upper_body_mask]).float().to(self.device)
        self.std_upper = torch.from_numpy(self.std[upper_body_mask]).float().to(self.device)
        self.mean_hands = torch.from_numpy(self.mean[hands_body_mask]).float().to(self.device)
        self.std_hands = torch.from_numpy(self.std[hands_body_mask]).float().to(self.device)
        self.mean_lower = torch.from_numpy(self.mean[lower_body_mask]).float().to(self.device)
        self.std_lower = torch.from_numpy(self.std[lower_body_mask]).float().to(self.device)
        
        if getattr(self.cfg, 'use_trans', False):
            self.trans_mean = torch.from_numpy(np.load(self.cfg.mean_trans_path)).float().to(self.device)
            self.trans_std = torch.from_numpy(np.load(self.cfg.std_trans_path)).float().to(self.device)
        
        self.pose_norm = getattr(self.cfg.data, 'pose_norm', True)
        
        logger.info(f"Normalization loaded: pose_norm={self.pose_norm}")
    
    def _init_joint_masks(self):
        """初始化关节掩码"""
        ori_joint_list = joints_list.get("beat_smplx_full", {})
        tar_joint_list_upper = list(joints_list.get("beat_smplx_upper", {}).keys())
        tar_joint_list_hands = list(joints_list.get("beat_smplx_hands", {}).keys())
        tar_joint_list_lower = list(joints_list.get("beat_smplx_lower", {}).keys())
        
        self.ori_dim = 165
        
        # 构建索引映射
        joint_indices = {}
        current_idx = 0
        for k, v in ori_joint_list.items():
            start = current_idx
            end = current_idx + v
            joint_indices[k] = (start, end)
            current_idx = end
        
        # 构建掩码
        self.joint_mask_upper = self._build_mask(tar_joint_list_upper, joint_indices)
        self.joint_mask_hands = self._build_mask(tar_joint_list_hands, joint_indices)
        self.joint_mask_lower = self._build_mask(tar_joint_list_lower, joint_indices)
    
    def _build_mask(self, target_joints, joint_indices):
        """构建关节掩码"""
        mask = np.zeros(self.ori_dim)
        for joint_name in target_joints:
            if joint_name in joint_indices:
                start, end = joint_indices[joint_name]
                mask[start:end] = 1
        return mask.astype(bool)
    
    def _init_evaluation_tools(self):
        """初始化评估工具"""
        # 初始化SMPLX模型用于生成关节点
        try:
            self.smplx = smplx.create(
                self.cfg.data.data_path_1 + "smplx_models/",
                model_type='smplx',
                gender='NEUTRAL_2020',
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100,
                ext='npz',
                use_pca=False,
            )
            self.smplx = self.smplx.to(self.device)
            self.smplx.eval()
            logger.info("SMPLX model loaded for evaluation")
        except Exception as e:
            logger.warning(f"Failed to load SMPLX model: {e}")
            self.smplx = None
        
        # 初始化评估模型（用于FGD计算）
        try:
            eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
            eval_args = type('Args', (), {})()
            eval_args.vae_layer = 4
            eval_args.vae_length = 240
            eval_args.vae_test_dim = 330
            eval_args.variational = False
            eval_args.data_path_1 = "./datasets/hub/"
            eval_args.vae_grow = [1, 1, 2, 1]
            
            self.eval_copy = getattr(eval_model_module, 'VAESKConv')(eval_args).to(self.device)
            
            # 加载预训练权重
            eval_ckpt_path = './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/AESKConv_240_100.bin'
            if os.path.exists(eval_ckpt_path):
                from utils import other_tools
                other_tools.load_checkpoints(self.eval_copy, eval_ckpt_path, 'VAESKConv')
                logger.info("Evaluation model (VAESKConv) loaded")
            else:
                logger.warning(f"Evaluation model checkpoint not found: {eval_ckpt_path}")
                self.eval_copy = None
        except Exception as e:
            logger.warning(f"Failed to load evaluation model: {e}")
            self.eval_copy = None
        
        # 初始化对齐计算器（BC指标）
        try:
            avg_vel = getattr(self.train_dataset, 'avg_vel', 0.05)
            self.alignmenter = metric.alignment(
                0.3, 7, avg_vel,
                upper_body=[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            )
            self.align_mask = 60
            logger.info("Alignment calculator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize alignment calculator: {e}")
            self.alignmenter = None
        
        # 初始化L1多样性计算器
        self.l1_calculator_speaker = metric.L1div()
        self.l1_calculator_listener = metric.L1div()
        self.l1_calculator_speaker_gt = metric.L1div()
        self.l1_calculator_listener_gt = metric.L1div()
        logger.info("L1 diversity calculators initialized")
        
        # 用于latent存储的列表
        self.latent_out_speaker = []
        self.latent_ori_speaker = []
        self.latent_out_listener = []
        self.latent_ori_listener = []
    
    def pose_to_latent(self, pose, trans=None, trans_v=None):
        """
        将原始姿态编码为VQ latent
        
        Args:
            pose: [B, T, 165] 原始姿态
            trans: [B, T, 3] 平移(可选)
            trans_v: [B, T, 3] 平移速度(可选)
        
        Returns:
            latent: [B, T//4, 384] VQ latent
        """
        bs, seq = pose.shape[:2]
        
        # 提取身体部位
        pose_upper_3d = pose[..., self.joint_mask_upper]  # [B, T, 39]
        pose_hands_3d = pose[..., self.joint_mask_hands]  # [B, T, 90]
        pose_lower_3d = pose[..., self.joint_mask_lower]  # [B, T, 27]
        
        # 转换为6D旋转
        pose_upper = rc.axis_angle_to_matrix(pose_upper_3d.reshape(bs, seq, 13, 3))
        pose_upper = rc.matrix_to_rotation_6d(pose_upper).reshape(bs, seq, 78)
        
        pose_hands = rc.axis_angle_to_matrix(pose_hands_3d.reshape(bs, seq, 30, 3))
        pose_hands = rc.matrix_to_rotation_6d(pose_hands).reshape(bs, seq, 180)
        
        pose_lower = rc.axis_angle_to_matrix(pose_lower_3d.reshape(bs, seq, 9, 3))
        pose_lower = rc.matrix_to_rotation_6d(pose_lower).reshape(bs, seq, 54)
        
        # 归一化
        if self.pose_norm:
            pose_upper = (pose_upper - self.mean_upper) / self.std_upper
            pose_hands = (pose_hands - self.mean_hands) / self.std_hands
            pose_lower = (pose_lower - self.mean_lower) / self.std_lower
        
        # 处理平移速度
        if trans_v is not None and getattr(self.cfg, 'use_trans', False):
            trans_v_norm = (trans_v - self.trans_mean) / self.trans_std
            pose_lower = torch.cat([pose_lower, trans_v_norm], dim=-1)
        
        # 编码为latent
        with torch.no_grad():
            latent_upper = self.vq_model_upper.encoder(pose_upper.transpose(1, 2))
            latent_hands = self.vq_model_hands.encoder(pose_hands.transpose(1, 2))
            latent_lower = self.vq_model_lower.encoder(pose_lower.transpose(1, 2))
        
        # 转置回 [B, T, D]
        latent_upper = latent_upper.transpose(1, 2)
        latent_hands = latent_hands.transpose(1, 2)
        latent_lower = latent_lower.transpose(1, 2)
        
        # 拼接并缩放
        latent = torch.cat([latent_upper, latent_hands, latent_lower], dim=-1)
        latent = latent / self.cfg.vqvae_latent_scale
        
        return latent
    
    def pose_to_seed_latent(self, pose, n_seed=None):
        """
        将原始姿态编码为seed latent，供denoiser使用
        
        Args:
            pose: [B, T, 165] 原始姿态 (seed)
            n_seed: int, 种子帧数，默认使用 pre_frames_scaled
        
        Returns:
            seed_latent: [B, n_seed, 384] 格式化的seed latent
        """
        if n_seed is None:
            # 使用 pre_frames_scaled = pre_frames * vqvae_squeeze_scale
            n_seed = self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale
        
        bs, seq = pose.shape[:2]
        
        # 提取身体部位
        pose_upper_3d = pose[..., self.joint_mask_upper]  # [B, T, 39]
        pose_hands_3d = pose[..., self.joint_mask_hands]  # [B, T, 90]
        pose_lower_3d = pose[..., self.joint_mask_lower]  # [B, T, 27]
        
        # 转换为6D旋转
        pose_upper = rc.axis_angle_to_matrix(pose_upper_3d.reshape(bs, seq, 13, 3))
        pose_upper = rc.matrix_to_rotation_6d(pose_upper).reshape(bs, seq, 78)
        
        pose_hands = rc.axis_angle_to_matrix(pose_hands_3d.reshape(bs, seq, 30, 3))
        pose_hands = rc.matrix_to_rotation_6d(pose_hands).reshape(bs, seq, 180)
        
        pose_lower = rc.axis_angle_to_matrix(pose_lower_3d.reshape(bs, seq, 9, 3))
        pose_lower = rc.matrix_to_rotation_6d(pose_lower).reshape(bs, seq, 54)
        
        # 归一化
        if self.pose_norm:
            pose_upper = (pose_upper - self.mean_upper) / self.std_upper
            pose_hands = (pose_hands - self.mean_hands) / self.std_hands
            pose_lower = (pose_lower - self.mean_lower) / self.std_lower
        
        # 编码为latent
        with torch.no_grad():
            latent_upper = self.vq_model_upper.encoder(pose_upper.transpose(1, 2))
            latent_hands = self.vq_model_hands.encoder(pose_hands.transpose(1, 2))
            latent_lower = self.vq_model_lower.encoder(pose_lower.transpose(1, 2))
        
        # 转置回 [B, T, D]
        latent_upper = latent_upper.transpose(1, 2)  # [B, T//4, 128]
        latent_hands = latent_hands.transpose(1, 2)  # [B, T//4, 128]
        latent_lower = latent_lower.transpose(1, 2)  # [B, T//4, 128]
        
        # 取最后 n_seed 帧
        latent_upper = latent_upper[:, -n_seed:, :]  # [B, n_seed, 128]
        latent_hands = latent_hands[:, -n_seed:, :]  # [B, n_seed, 128]
        latent_lower = latent_lower[:, -n_seed:, :]  # [B, n_seed, 128]
        
        # 拼接成 [B, n_seed, 384]
        seed_latent = torch.cat([latent_upper, latent_hands, latent_lower], dim=-1)
        seed_latent = seed_latent / self.cfg.vqvae_latent_scale
        
        return seed_latent  # [B, n_seed, 384]
    
    def _sample_training_windows(self, batch):
        """
        从完整序列中采样训练窗口
        数据集现在返回完整序列，需要在这里动态切分
        
        Args:
            batch: 包含完整序列的batch
        
        Returns:
            speaker_input, listener_input, speaker_latent, listener_latent
        """
        batch_size = len(batch['speaker']['pose'])  # 现在pose是列表
        
        # 获取配置参数
        horizon = self.cfg.model.horizon
        pre_frames_scaled = self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale
        
        speaker_inputs = []
        listener_inputs = []
        speaker_targets = []
        listener_targets = []
        
        for b in range(batch_size):
            # 获取当前样本的完整序列
            speaker_pose_full = batch['speaker']['pose'][b]  # [T, 165]
            listener_pose_full = batch['listener']['pose'][b]  # [T, 165]
            speaker_audio_full = batch['speaker']['audio'][b]  # [T, audio_dim]
            speaker_word_full = batch['speaker']['word'][b]  # [T]
            
            seq_len = speaker_pose_full.shape[0]
            
            # 确保序列足够长
            if seq_len < horizon + pre_frames_scaled:
                # 序列太短，跳过或使用整个序列
                continue
            
            # 随机采样起始点
            max_start = seq_len - horizon - pre_frames_scaled
            if max_start > 0:
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start_idx = 0
            
            # 切分数据
            # seed: [start_idx : start_idx + pre_frames_scaled]
            # target: [start_idx + pre_frames_scaled : start_idx + pre_frames_scaled + horizon]
            seed_start = start_idx
            seed_end = start_idx + pre_frames_scaled
            target_start = seed_end
            target_end = target_start + horizon
            
            # 提取seed姿态
            speaker_seed_pose = speaker_pose_full[seed_start:seed_end]  # [pre_frames_scaled, 165]
            listener_seed_pose = listener_pose_full[seed_start:seed_end]  # [pre_frames_scaled, 165]
            
            # 提取target姿态
            speaker_target_pose = speaker_pose_full[target_start:target_end]  # [horizon, 165]
            listener_target_pose = listener_pose_full[target_start:target_end]  # [horizon, 165]
            
            # 提取audio和word（与target对齐）
            speaker_audio = speaker_audio_full[target_start:target_end]  # [horizon, audio_dim]
            speaker_word = speaker_word_full[target_start:target_end]  # [horizon]
            
            # 编码seed为latent
            speaker_seed_latent = self.pose_to_seed_latent(speaker_seed_pose.unsqueeze(0)).squeeze(0)  # [n_seed, 384]
            listener_seed_latent = self.pose_to_seed_latent(listener_seed_pose.unsqueeze(0)).squeeze(0)  # [n_seed, 384]
            
            # 构建输入
            speaker_inputs.append({
                'audio': speaker_audio,
                'word': speaker_word,
                'seed': speaker_seed_latent,
                'id': batch['speaker']['id'][b] if isinstance(batch['speaker']['id'], list) else batch['speaker']['id'][b:b+1].expand(horizon),
            })
            
            listener_inputs.append({
                'seed': listener_seed_latent,
                'id': batch['listener']['id'][b] if isinstance(batch['listener']['id'], list) else batch['listener']['id'][b:b+1].expand(horizon),
            })
            
            speaker_targets.append(speaker_target_pose)
            listener_targets.append(listener_target_pose)
        
        if len(speaker_inputs) == 0:
            return None, None, None, None
        
        # 堆叠成batch
        speaker_input = {
            'audio': torch.stack([s['audio'] for s in speaker_inputs]),
            'word': torch.stack([s['word'] for s in speaker_inputs]),
            'seed': torch.stack([s['seed'] for s in speaker_inputs]),
            'id': torch.stack([s['id'] for s in speaker_inputs]),
        }
        
        listener_input = {
            'seed': torch.stack([s['seed'] for s in listener_inputs]),
            'id': torch.stack([s['id'] for s in listener_inputs]),
        }
        
        speaker_target = torch.stack(speaker_targets)
        listener_target = torch.stack(listener_targets)
        
        return speaker_input, listener_input, speaker_target, listener_target
    
    def latent_to_pose(self, latent):
        """
        将VQ latent解码回原始姿态
        
        Args:
            latent: [B, T, 384] VQ latent
        
        Returns:
            pose: [B, T*4, 165] 原始姿态（轴角格式）
        """
        bs = latent.shape[0]
        
        # 缩放还原
        latent = latent * self.cfg.vqvae_latent_scale
        
        # 分离三个部分
        code_dim = 128
        latent_upper = latent[..., :code_dim]
        latent_hands = latent[..., code_dim:code_dim*2]
        latent_lower = latent[..., code_dim*2:code_dim*3]
        
        # 解码
        with torch.no_grad():
            rec_upper = self.vq_model_upper.latent2origin(latent_upper)[0]
            rec_hands = self.vq_model_hands.latent2origin(latent_hands)[0]
            rec_lower = self.vq_model_lower.latent2origin(latent_lower)[0]
        
        # 反归一化
        if self.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower
        
        # 转换为轴角格式
        bs, n = rec_upper.shape[0], rec_upper.shape[1]
        
        rec_upper_aa = rc.rotation_6d_to_matrix(rec_upper.reshape(bs, n, 13, 6))
        rec_upper_aa = rc.matrix_to_axis_angle(rec_upper_aa).reshape(bs, n, 13 * 3)
        
        rec_hands_aa = rc.rotation_6d_to_matrix(rec_hands.reshape(bs, n, 30, 6))
        rec_hands_aa = rc.matrix_to_axis_angle(rec_hands_aa).reshape(bs, n, 30 * 3)
        
        rec_lower_aa = rc.rotation_6d_to_matrix(rec_lower.reshape(bs, n, 9, 6))
        rec_lower_aa = rc.matrix_to_axis_angle(rec_lower_aa).reshape(bs, n, 9 * 3)
        
        # 合并回完整姿态 [B, n, 165]
        pose = torch.zeros(bs, n, 165, device=self.device)
        
        # 使用掩码填充
        upper_indices = np.where(self.joint_mask_upper)[0]
        hands_indices = np.where(self.joint_mask_hands)[0]
        lower_indices = np.where(self.joint_mask_lower)[0]
        
        for i, idx in enumerate(upper_indices):
            pose[:, :, idx] = rec_upper_aa[:, :, i]
        for i, idx in enumerate(hands_indices):
            pose[:, :, idx] = rec_hands_aa[:, :, i]
        for i, idx in enumerate(lower_indices):
            pose[:, :, idx] = rec_lower_aa[:, :, i]
        
        # latent2origin 已经恢复原始帧率，不需要额外插值
        return pose
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_speaker_loss = 0.0
        total_listener_loss = 0.0
        n_valid_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            if batch is None:
                continue

            # 将数据移动到设备
            batch = self._to_device(batch)

            # 从完整序列中采样训练窗口
            speaker_input, listener_input, speaker_target, listener_target = self._sample_training_windows(batch)
            
            if speaker_input is None:
                # 没有有效的训练样本
                continue

            # 编码target姿态到latent空间
            speaker_latent = self.pose_to_latent(speaker_target)
            listener_latent = self.pose_to_latent(listener_target)

            # 根据梯度更新模式选择训练方式
            if self.separate_gradient_update:
                if self.gradient_update_mode == 'separate_optimizers':
                    # 独立优化器模式：分别更新Speaker和Listener
                    loss, spk_loss, lst_loss = self._train_step_separate_optimizers(
                        speaker_input, listener_input, speaker_latent, listener_latent
                    )
                else:
                    # 交替更新模式
                    loss, spk_loss, lst_loss = self._train_step_alternate(
                        speaker_input, listener_input, speaker_latent, listener_latent, i
                    )
            else:
                # 联合训练模式
                loss, spk_loss, lst_loss = self._train_step_joint(
                    speaker_input, listener_input, speaker_latent, listener_latent
                )

            # 检查NaN
            if torch.isnan(torch.tensor(loss)):
                logger.error(f"NaN loss detected at batch {i}")
                continue

            # 记录损失
            total_loss += loss
            total_speaker_loss += spk_loss
            total_listener_loss += lst_loss
            n_valid_batches += 1

            # 更新进度条
            if i % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'spk': f"{spk_loss:.4f}",
                    'lst': f"{lst_loss:.4f}"
                })

        # 计算平均损失
        if n_valid_batches > 0:
            avg_loss = total_loss / n_valid_batches
            avg_speaker_loss = total_speaker_loss / n_valid_batches
            avg_listener_loss = total_listener_loss / n_valid_batches
        else:
            avg_loss = 0.0
            avg_speaker_loss = 0.0
            avg_listener_loss = 0.0
            logger.warning("No valid batches in this epoch")

        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                   f"Speaker: {avg_speaker_loss:.4f} | Listener: {avg_listener_loss:.4f}")

        # 更新学习率调度器
        self.current_epoch = epoch
        self._step_lr_schedulers()

        # 打印当前学习率
        if self.lr_scheduler_type == 'cosine':
            if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                lr_spk = self.speaker_optimizer.param_groups[0]['lr']
                lr_lst = self.listener_optimizer.param_groups[0]['lr']
                logger.info(f"Learning Rate - Speaker: {lr_spk:.2e}, Listener: {lr_lst:.2e}")
            elif hasattr(self, 'scheduler'):
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning Rate: {lr:.2e}")

        return avg_loss

    def _train_step_joint(self, speaker_input, listener_input, speaker_latent, listener_latent):
        """联合训练模式：同时更新所有参数"""
        self.optimizer.zero_grad()

        losses = self.model.train_forward(
            speaker_data=speaker_input,
            listener_data=listener_input,
            speaker_latent=speaker_latent,
            listener_latent=listener_latent,
            train_consistency=False,
            separate_gradient_update=False
        )

        loss = losses['loss']
        loss.backward()

        if self.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()

        return loss.item(), losses['speaker_flow_loss'].item(), losses['listener_flow_loss'].item()

    def _train_step_alternate(self, speaker_input, listener_input, speaker_latent, listener_latent, batch_idx):
        """交替更新模式：交替更新Speaker和Listener"""
        # 根据batch索引决定更新哪个角色
        update_speaker = (batch_idx % 2 == 0)

        self.optimizer.zero_grad()

        # 前向传播，但只计算指定角色的损失
        losses = self.model.train_forward(
            speaker_data=speaker_input,
            listener_data=listener_input,
            speaker_latent=speaker_latent,
            listener_latent=listener_latent,
            train_consistency=False,
            separate_gradient_update=True,
            update_role='speaker' if update_speaker else 'listener'
        )

        # 只使用对应角色的损失进行反向传播
        loss = losses['loss']
        loss.backward()

        if self.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()

        # 返回两个角色的实际损失值（用于记录），即使只更新了一个角色
        # 注意：虽然只反向传播了一个角色的损失，但两个角色的损失都应该被记录
        spk_loss = losses['speaker_flow_loss'].item()
        lst_loss = losses['listener_flow_loss'].item()

        if update_speaker:
            # 更新了Speaker，返回speaker_loss作为总损失
            return (spk_loss, spk_loss, lst_loss)
        else:
            # 更新了Listener，返回listener_loss作为总损失
            return (lst_loss, spk_loss, lst_loss)

    def _train_step_separate_optimizers(self, speaker_input, listener_input, speaker_latent, listener_latent):
        """独立优化器模式：分别使用不同的优化器更新Speaker和Listener"""
        # 1. 更新Speaker（只使用Speaker损失）
        self.speaker_optimizer.zero_grad()

        losses_spk = self.model.train_forward(
            speaker_data=speaker_input,
            listener_data=listener_input,
            speaker_latent=speaker_latent,
            listener_latent=listener_latent,
            train_consistency=False,
            separate_gradient_update=True,
            update_role='speaker'
        )

        spk_loss = losses_spk['loss']
        spk_loss.backward()

        if self.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.speaker_optimizer.step()

        # 2. 更新Listener（只使用Listener损失）
        self.listener_optimizer.zero_grad()

        losses_lst = self.model.train_forward(
            speaker_data=speaker_input,
            listener_data=listener_input,
            speaker_latent=speaker_latent,
            listener_latent=listener_latent,
            train_consistency=False,
            separate_gradient_update=True,
            update_role='listener'
        )

        lst_loss = losses_lst['loss']
        lst_loss.backward()

        if self.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.listener_optimizer.step()

        # 返回两个角色的损失之和（用于记录）
        total_loss = losses_spk['speaker_flow_loss'].item() + losses_lst['listener_flow_loss'].item()
        return (total_loss,
                losses_spk['speaker_flow_loss'].item(),
                losses_lst['listener_flow_loss'].item())
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        验证评估
        
        Args:
            epoch: 当前epoch数
        
        Returns:
            metrics: 字典包含所有评估指标
        """
        if self.val_dataset is None:
            logger.warning("No validation dataset provided, skipping validation")
            return {}
        
        logger.info(f"Starting validation for epoch {epoch}")
        
        self.model.eval()
        
        # 重置评估计算器
        self.l1_calculator_speaker.reset()
        self.l1_calculator_listener.reset()
        self.l1_calculator_speaker_gt.reset()
        self.l1_calculator_listener_gt.reset()
        self.latent_out_speaker = []
        self.latent_ori_speaker = []
        self.latent_out_listener = []
        self.latent_ori_listener = []
        
        # 用于BC计算的变量
        speaker_align = 0
        listener_align = 0
        speaker_align_gt = 0
        listener_align_gt = 0
        total_length = 0
        smplx_processed_batches = 0
        
        val_pbar = tqdm(self.val_loader, desc=f"Validation epoch {epoch}")
        
        for i, batch in enumerate(val_pbar):
            if batch is None:
                continue
            
            # 限制验证批次数量
            if self.max_val_iterations and i >= self.max_val_iterations:
                break
            
            # 将数据移动到设备
            batch = self._to_device(batch)
            
            # 从完整序列中提取验证样本（取第一个horizon长度）
            # 数据格式：batch['speaker']['pose'] 是列表，每个元素是 [T, 165]
            speaker_pose_full = batch['speaker']['pose'][0]  # [T, 165]
            listener_pose_full = batch['listener']['pose'][0]  # [T, 165]
            speaker_audio_full = batch['speaker']['audio'][0]  # [T, audio_dim]
            speaker_word_full = batch['speaker']['word'][0]  # [T]
            
            # 获取配置参数
            horizon = self.cfg.model.horizon
            pre_frames_scaled = self.cfg.pre_frames * self.cfg.vqvae_squeeze_scale
            
            # 确保序列足够长
            seq_len = speaker_pose_full.shape[0]
            if seq_len < horizon + pre_frames_scaled:
                logger.warning(f"Sequence too short for validation: {seq_len} < {horizon + pre_frames_scaled}")
                continue
            
            # 提取seed和target（从序列开头取）
            speaker_seed_pose = speaker_pose_full[:pre_frames_scaled]  # [pre_frames_scaled, 165]
            listener_seed_pose = listener_pose_full[:pre_frames_scaled]  # [pre_frames_scaled, 165]
            speaker_target_pose = speaker_pose_full[pre_frames_scaled:pre_frames_scaled + horizon]  # [horizon, 165]
            listener_target_pose = listener_pose_full[pre_frames_scaled:pre_frames_scaled + horizon]  # [horizon, 165]
            speaker_audio = speaker_audio_full[:horizon]  # [horizon, audio_dim]
            speaker_word = speaker_word_full[:horizon]  # [horizon]
            
            # 编码seed为latent
            speaker_seed_latent = self.pose_to_seed_latent(speaker_seed_pose.unsqueeze(0))  # [1, n_seed, 384]
            listener_seed_latent = self.pose_to_seed_latent(listener_seed_pose.unsqueeze(0))  # [1, n_seed, 384]
            
            # 获取真值latent
            speaker_latent_gt = self.pose_to_latent(speaker_target_pose.unsqueeze(0))  # [1, T_latent, 384]
            listener_latent_gt = self.pose_to_latent(listener_target_pose.unsqueeze(0))  # [1, T_latent, 384]
            
            # ========== 滑动窗口推理 ==========
            # 计算窗口参数（latent空间）
            total_frames_latent = self.cfg.model.horizon // self.cfg.vqvae_squeeze_scale  # 144 // 4 = 36帧latent
            pre_frames_latent = self.cfg.pre_frames  # 4帧latent
            window_size = self.cfg.inference_window_size  # 36帧latent
            effective_generation = window_size - pre_frames_latent  # 36 - 4 = 32帧latent
            
            # 计算需要多少个窗口
            num_windows = (total_frames_latent - pre_frames_latent) // effective_generation
            if num_windows > 1:
                num_windows = num_windows - 1  # 跳过最后一个不完整的窗口
            
            # 初始化结果存储
            all_speaker_pred = []
            all_listener_pred = []
            
            # 滑动窗口循环
            for i in range(num_windows):
                # 第一个窗口使用初始seed
                if i == 0:
                    speaker_seed_tmp = speaker_seed_latent[:, :pre_frames_latent, :]
                    listener_seed_tmp = listener_seed_latent[:, :pre_frames_latent, :]
                else:
                    # 后续窗口使用上一个窗口的最后pre_frames帧
                    speaker_seed_tmp = last_speaker_pred[:, -pre_frames_latent:, :]
                    listener_seed_tmp = last_listener_pred[:, -pre_frames_latent:, :]
                
                # 准备输入数据
                speaker_id = batch['speaker']['id'][0] if isinstance(batch['speaker']['id'], list) else batch['speaker']['id'][0:1]
                listener_id = batch['listener']['id'][0] if isinstance(batch['listener']['id'], list) else batch['listener']['id'][0:1]
                
                speaker_input = {
                    'audio': speaker_audio.unsqueeze(0),
                    'word': speaker_word.unsqueeze(0),
                    'seed': speaker_seed_tmp,
                    'id': speaker_id.unsqueeze(0).expand(1, horizon),
                }
                listener_input = {
                    'seed': listener_seed_tmp,
                    'id': listener_id.unsqueeze(0).expand(1, horizon),
                }
                
                # 生成Speaker和Listener手势（联合生成）
                speaker_pred, listener_pred = self.model.generate(
                    speaker_data=speaker_input,
                    listener_data=listener_input,
                    speaker_gt=speaker_latent_gt,
                    listener_gt=listener_latent_gt,
                    mode='conditional',
                    num_steps=self.cfg.model.get('num_inference_steps', 2),
                    guidance_scale=self.cfg.model.get('guidance_scale', 2.0)
                )
                
                # 保存最后一个窗口的完整结果
                if i == num_windows - 1:
                    all_speaker_pred.append(speaker_pred)
                    all_listener_pred.append(listener_pred)
                else:
                    # 拼接结果（跳过pre_frames重叠部分）
                    all_speaker_pred.append(speaker_pred[:, pre_frames_scaled:, :])
                    all_listener_pred.append(listener_pred[:, pre_frames_scaled:, :])
                
                # 保存当前窗口的预测用于下一个窗口
                last_speaker_pred = speaker_pred
                last_listener_pred = listener_pred
            
            # 拼接所有窗口的结果
            speaker_pred = torch.cat(all_speaker_pred, dim=1)
            listener_pred = torch.cat(all_listener_pred, dim=1)
            
            # 转换为姿态空间用于评估
            # speaker_pred shape: [B, 384, 1, T] -> [B, T, 384]
            speaker_pred_3d = speaker_pred.squeeze(2).permute(0, 2, 1)
            listener_pred_3d = listener_pred.squeeze(2).permute(0, 2, 1)
            speaker_pose_pred = self.latent_to_pose(speaker_pred_3d)
            listener_pose_pred = self.latent_to_pose(listener_pred_3d)
            speaker_pose_gt = speaker_target_pose.unsqueeze(0)
            listener_pose_gt = listener_target_pose.unsqueeze(0)
            
            # 计算FGD的latent表示
            if self.eval_copy is not None:
                # 将姿态转换为6D格式
                bs, n = speaker_pose_pred.shape[0], speaker_pose_pred.shape[1]
                
                speaker_pose_6d = rc.axis_angle_to_matrix(speaker_pose_pred.reshape(bs * n, 55, 3))
                speaker_pose_6d = rc.matrix_to_rotation_6d(speaker_pose_6d).reshape(bs, n, 330)
                
                listener_pose_6d = rc.axis_angle_to_matrix(listener_pose_pred.reshape(bs * n, 55, 3))
                listener_pose_6d = rc.matrix_to_rotation_6d(listener_pose_6d).reshape(bs, n, 330)
                
                speaker_pose_gt_6d = rc.axis_angle_to_matrix(speaker_pose_gt.reshape(bs * n, 55, 3))
                speaker_pose_gt_6d = rc.matrix_to_rotation_6d(speaker_pose_gt_6d).reshape(bs, n, 330)
                
                listener_pose_gt_6d = rc.axis_angle_to_matrix(listener_pose_gt.reshape(bs * n, 55, 3))
                listener_pose_gt_6d = rc.matrix_to_rotation_6d(listener_pose_gt_6d).reshape(bs, n, 330)
                
                # 计算latent
                vae_test_len = getattr(self.cfg, 'vae_test_len', 32)
                
                # 计算FGD：只使用vae_test_len的整数倍帧数
                remain = n % vae_test_len
                if n - remain > 0:
                    valid_len = n - remain
                    latent_sp = self.eval_copy.map2latent(speaker_pose_6d[:, :valid_len])
                    latent_ls = self.eval_copy.map2latent(listener_pose_6d[:, :valid_len])
                    latent_sp_gt = self.eval_copy.map2latent(speaker_pose_gt_6d[:, :valid_len])
                    latent_ls_gt = self.eval_copy.map2latent(listener_pose_gt_6d[:, :valid_len])
                    
                    # reshape为[-1, vae_test_len]，与原框架保持一致
                    self.latent_out_speaker.append(latent_sp.reshape(-1, vae_test_len).detach().cpu().numpy())
                    self.latent_out_listener.append(latent_ls.reshape(-1, vae_test_len).detach().cpu().numpy())
                    self.latent_ori_speaker.append(latent_sp_gt.reshape(-1, vae_test_len).detach().cpu().numpy())
                    self.latent_ori_listener.append(latent_ls_gt.reshape(-1, vae_test_len).detach().cpu().numpy())
            
            # 使用SMPLX计算关节点用于BC和L1Div
            if self.smplx is not None:
                smplx_processed_batches += 1
                bs, n = speaker_pose_pred.shape[0], speaker_pose_pred.shape[1]
                bs_gt, n_gt = speaker_pose_gt.shape[0], speaker_pose_gt.shape[1]
                
                # 检查形状是否匹配
                if bs != bs_gt or n != n_gt:
                    logger.warning(f"Shape mismatch: pred=[{bs}, {n}, 165], gt=[{bs_gt}, {n_gt}, 165]")
                    # 使用较小的维度
                    bs = min(bs, bs_gt)
                    n = min(n, n_gt)
                    speaker_pose_pred = speaker_pose_pred[:bs, :n, :]
                    speaker_pose_gt = speaker_pose_gt[:bs, :n, :]
                    listener_pose_pred = listener_pose_pred[:bs, :n, :]
                    listener_pose_gt = listener_pose_gt[:bs, :n, :]
                
                # 分批处理以减少内存使用
                chunk_size = 512
                joints_speaker_pred = []
                joints_speaker_gt = []
                joints_listener_pred = []
                joints_listener_gt = []
                
                # 先将姿态reshape为 [bs*n, 165]，并确保连续
                speaker_pose_pred_flat = speaker_pose_pred.reshape(bs * n, 165).contiguous()
                speaker_pose_gt_flat = speaker_pose_gt.reshape(bs * n, 165).contiguous()
                listener_pose_pred_flat = listener_pose_pred.reshape(bs * n, 165).contiguous()
                listener_pose_gt_flat = listener_pose_gt.reshape(bs * n, 165).contiguous()
                
                for j in range(0, bs * n, chunk_size):
                    end_idx = min(j + chunk_size, bs * n)
                    chunk_batch_size = end_idx - j
                    
                    # 创建SMPLX参数张量（零张量）
                    betas = torch.zeros(chunk_batch_size, 300, device=self.device)
                    expression = torch.zeros(chunk_batch_size, 100, device=self.device)
                    jaw_pose = torch.zeros(chunk_batch_size, 3, device=self.device)
                    leye_pose = torch.zeros(chunk_batch_size, 3, device=self.device)
                    reye_pose = torch.zeros(chunk_batch_size, 3, device=self.device)
                    
                    # Speaker预测 - 按照原生代码方式：先切片列，再切片区
                    sp_pred = self.smplx(
                        betas=betas,
                        expression=expression,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        global_orient=speaker_pose_pred_flat[:, :3][j:end_idx],
                        body_pose=speaker_pose_pred_flat[:, 3:66][j:end_idx],
                        left_hand_pose=speaker_pose_pred_flat[:, 75:120][j:end_idx],
                        right_hand_pose=speaker_pose_pred_flat[:, 120:165][j:end_idx],
                        return_joints=True
                    )['joints'].detach().cpu().numpy()
                    joints_speaker_pred.append(sp_pred)
                    
                    # Speaker真值
                    sp_gt = self.smplx(
                        betas=betas,
                        expression=expression,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        global_orient=speaker_pose_gt_flat[:, :3][j:end_idx],
                        body_pose=speaker_pose_gt_flat[:, 3:66][j:end_idx],
                        left_hand_pose=speaker_pose_gt_flat[:, 75:120][j:end_idx],
                        right_hand_pose=speaker_pose_gt_flat[:, 120:165][j:end_idx],
                        return_joints=True
                    )['joints'].detach().cpu().numpy()
                    joints_speaker_gt.append(sp_gt)
                    
                    # Listener预测
                    ls_pred = self.smplx(
                        betas=betas,
                        expression=expression,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        global_orient=listener_pose_pred_flat[:, :3][j:end_idx],
                        body_pose=listener_pose_pred_flat[:, 3:66][j:end_idx],
                        left_hand_pose=listener_pose_pred_flat[:, 75:120][j:end_idx],
                        right_hand_pose=listener_pose_pred_flat[:, 120:165][j:end_idx],
                        return_joints=True
                    )['joints'].detach().cpu().numpy()
                    joints_listener_pred.append(ls_pred)
                    
                    # Listener真值
                    ls_gt = self.smplx(
                        betas=betas,
                        expression=expression,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        global_orient=listener_pose_gt_flat[:, :3][j:end_idx],
                        body_pose=listener_pose_gt_flat[:, 3:66][j:end_idx],
                        left_hand_pose=listener_pose_gt_flat[:, 75:120][j:end_idx],
                        right_hand_pose=listener_pose_gt_flat[:, 120:165][j:end_idx],
                        return_joints=True
                    )['joints'].detach().cpu().numpy()
                    joints_listener_gt.append(ls_gt)
                
                # 合并结果
                joints_speaker_pred = np.concatenate(joints_speaker_pred, axis=0).reshape(bs, n, 127, 3)[:, :, :55, :]
                joints_speaker_gt = np.concatenate(joints_speaker_gt, axis=0).reshape(bs, n, 127, 3)[:, :, :55, :]
                joints_listener_pred = np.concatenate(joints_listener_pred, axis=0).reshape(bs, n, 127, 3)[:, :, :55, :]
                joints_listener_gt = np.concatenate(joints_listener_gt, axis=0).reshape(bs, n, 127, 3)[:, :, :55, :]
                
                # 计算L1Div
                # 与单人模型保持一致：reshape为 [bs, n, features]，然后取第一个batch [0]
                # 最终形状为 [n, features]，表示 n 个时间帧的关节点特征
                speaker_pred_flat = joints_speaker_pred.reshape(bs, n, -1)[0]  # [n, features]
                speaker_gt_flat = joints_speaker_gt.reshape(bs, n, -1)[0]
                listener_pred_flat = joints_listener_pred.reshape(bs, n, -1)[0]
                listener_gt_flat = joints_listener_gt.reshape(bs, n, -1)[0]
                
                self.l1_calculator_speaker.run(speaker_pred_flat)
                self.l1_calculator_speaker_gt.run(speaker_gt_flat)
                self.l1_calculator_listener.run(listener_pred_flat)
                self.l1_calculator_listener_gt.run(listener_gt_flat)
                
                # 累加总长度（用于BC计算）
                total_length += n * bs
                
                # 计算BC（Beat Consistency）
                if self.alignmenter is not None:
                    # 获取音频路径
                    audio_path = batch['speaker'].get('audio_name', [None])
                    
                    if audio_path and len(audio_path) > 0 and audio_path[0] is not None and os.path.exists(audio_path[0]):
                        try:
                            in_audio_eval, sr = librosa.load(audio_path[0])
                            in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr)
                            
                            a_offset = int(self.align_mask * (self.cfg.data.audio_sr / self.cfg.data.pose_fps))
                            onset_bt = self.alignmenter.load_audio(
                                in_audio_eval[:int(self.cfg.data.audio_sr / self.cfg.data.pose_fps * n)],
                                a_offset, len(in_audio_eval) - a_offset, True
                            )
                            
                            # 对每个batch计算BC
                            for b in range(bs):
                                # 将关节点从 [n, 55, 3] reshape 为 [n, 165]
                                joints_sp_flat = joints_speaker_pred[b, :n, :, :].reshape(n, -1)
                                joints_sp_gt_flat = joints_speaker_gt[b, :n, :, :].reshape(n, -1)
                                joints_ls_flat = joints_listener_pred[b, :n, :, :].reshape(n, -1)
                                joints_ls_gt_flat = joints_listener_gt[b, :n, :, :].reshape(n, -1)
                                
                                # Speaker BC
                                beat_vel_sp = self.alignmenter.load_pose(
                                    joints_sp_flat, self.align_mask, n - self.align_mask, 30, True
                                )
                                speaker_align += self.alignmenter.calculate_align(onset_bt, beat_vel_sp, 30) * (n - 2 * self.align_mask)
                                
                                beat_vel_sp_gt = self.alignmenter.load_pose(
                                    joints_sp_gt_flat, self.align_mask, n - self.align_mask, 30, True
                                )
                                speaker_align_gt += self.alignmenter.calculate_align(onset_bt, beat_vel_sp_gt, 30) * (n - 2 * self.align_mask)
                                
                                # Listener BC
                                beat_vel_ls = self.alignmenter.load_pose(
                                    joints_ls_flat, self.align_mask, n - self.align_mask, 30, True
                                )
                                listener_align += self.alignmenter.calculate_align(onset_bt, beat_vel_ls, 30) * (n - 2 * self.align_mask)
                                
                                beat_vel_ls_gt = self.alignmenter.load_pose(
                                    joints_ls_gt_flat, self.align_mask, n - self.align_mask, 30, True
                                )
                                listener_align_gt += self.alignmenter.calculate_align(onset_bt, beat_vel_ls_gt, 30) * (n - 2 * self.align_mask)
                        except Exception as e:
                            logger.warning(f"Failed to calculate BC for batch: {e}")
        
        # 计算最终指标
        metrics = {}
        
        # FGD (Frechet Gesture Distance)
        if len(self.latent_out_speaker) > 0 and len(self.latent_ori_speaker) > 0:
            latent_out_sp = np.concatenate(self.latent_out_speaker, axis=0)
            latent_ori_sp = np.concatenate(self.latent_ori_speaker, axis=0)
            latent_out_ls = np.concatenate(self.latent_out_listener, axis=0)
            latent_ori_ls = np.concatenate(self.latent_ori_listener, axis=0)
            
            speaker_fgd = data_tools.FIDCalculator.frechet_distance(latent_out_sp, latent_ori_sp)
            listener_fgd = data_tools.FIDCalculator.frechet_distance(latent_out_ls, latent_ori_ls)
            avg_fgd = (speaker_fgd + listener_fgd) / 2
            
            metrics['speaker_fgd'] = speaker_fgd
            metrics['listener_fgd'] = listener_fgd
            metrics['avg_fgd'] = avg_fgd
            
            logger.info(f"Speaker FGD: {speaker_fgd:.4f}")
            logger.info(f"Listener FGD: {listener_fgd:.4f}")
            logger.info(f"Average FGD: {avg_fgd:.4f}")
        
        # 输出处理统计
        logger.info(f"[Validation Check] SMPLX processed batches: {smplx_processed_batches}")
        
        # L1Div (L1 Diversity)
        speaker_l1div = self.l1_calculator_speaker.avg()
        listener_l1div = self.l1_calculator_listener.avg()
        speaker_l1div_gt = self.l1_calculator_speaker_gt.avg()
        listener_l1div_gt = self.l1_calculator_listener_gt.avg()
        
        metrics['speaker_l1div'] = speaker_l1div
        metrics['listener_l1div'] = listener_l1div
        metrics['speaker_l1div_gt'] = speaker_l1div_gt
        metrics['listener_l1div_gt'] = listener_l1div_gt
        
        logger.info(f"Speaker L1Div: {speaker_l1div:.4f} (GT: {speaker_l1div_gt:.4f})")
        logger.info(f"Listener L1Div: {listener_l1div:.4f} (GT: {listener_l1div_gt:.4f})")
        
        # BC (Beat Consistency)
        if total_length > 0:
            actual_samples = min(self.max_val_iterations, len(self.val_loader)) if self.max_val_iterations else len(self.val_loader)
            denominator = total_length - 2 * actual_samples * self.align_mask
            
            if denominator > 0:
                speaker_bc = speaker_align / denominator
                listener_bc = listener_align / denominator
                speaker_bc_gt = speaker_align_gt / denominator
                listener_bc_gt = listener_align_gt / denominator
                
                metrics['speaker_bc'] = speaker_bc
                metrics['listener_bc'] = listener_bc
                metrics['speaker_bc_gt'] = speaker_bc_gt
                metrics['listener_bc_gt'] = listener_bc_gt
                
                logger.info(f"Speaker BC: {speaker_bc:.4f} (GT: {speaker_bc_gt:.4f})")
                logger.info(f"Listener BC: {listener_bc:.4f} (GT: {listener_bc_gt:.4f})")
        
        # 更新最佳指标
        self._update_best_metrics(metrics, epoch)
        
        return metrics
    
    def _update_best_metrics(self, metrics, epoch):
        """更新最佳指标并保存最佳模型"""
        for metric_name in ['speaker_fgd', 'listener_fgd', 'avg_fgd', 'speaker_bc', 'listener_bc']:
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            current_best = self.val_best[metric_name]['value']
            
            # 判断是否为更好的值（FGD越低越好，BC越高越好）
            if 'fgd' in metric_name:
                is_better = value < current_best
            else:
                is_better = value > current_best
            
            if is_better:
                self.val_best[metric_name] = {'value': float(value), 'epoch': epoch}
                logger.info(f"New best {metric_name}: {value:.4f} at epoch {epoch}")

                # 保存最佳模型
                best_path = os.path.join(self.ckpt_dir, f"best_{metric_name}.pt")

                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'metric_name': metric_name,
                    'metric_value': value,
                }

                # 保存优化器状态
                if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
                    best_checkpoint['speaker_optimizer_state_dict'] = self.speaker_optimizer.state_dict()
                    best_checkpoint['listener_optimizer_state_dict'] = self.listener_optimizer.state_dict()
                elif self.optimizer is not None:
                    best_checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

                torch.save(best_checkpoint, best_path)
    
    def _to_device(self, batch):
        """递归地将数据移动到设备"""
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            # 处理列表（变长序列）
            return [self._to_device(item) for item in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            # 早停状态
            'best_fgd': self.best_fgd,
            'patience_counter': self.patience_counter,
            'early_stopped': self.early_stopped,
        }

        # 保存优化器状态
        if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
            checkpoint['speaker_optimizer_state_dict'] = self.speaker_optimizer.state_dict()
            checkpoint['listener_optimizer_state_dict'] = self.listener_optimizer.state_dict()
            # 保存学习率调度器状态
            if hasattr(self, 'speaker_scheduler') and self.speaker_scheduler is not None:
                checkpoint['speaker_scheduler_state_dict'] = self.speaker_scheduler.state_dict()
            if hasattr(self, 'listener_scheduler') and self.listener_scheduler is not None:
                checkpoint['listener_scheduler_state_dict'] = self.listener_scheduler.state_dict()
        else:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            # 保存学习率调度器状态
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self, resume_from=None):
        """
        训练循环
        
        Args:
            resume_from: 检查点路径，用于恢复训练
        """
        start_epoch = 0
        
        # 从检查点恢复
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        logger.info(f"Starting bidirectional dyadic training from epoch {start_epoch}...")
        
        # 训练前进行初始验证（获得随机初始化模型的性能基准）
        if self.val_dataset is not None and start_epoch == 0:
            logger.info("Running initial validation before training...")
            self.validate(0)
        
        for epoch in range(start_epoch, self.max_epochs):
            # 训练一个epoch
            self.train_epoch(epoch)
            
            # 定期保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)
            
            # 定期验证（从epoch 1开始，每隔val_interval验证一次）
            if self.val_dataset is not None and (epoch + 1) % self.val_interval == 0:
                metrics = self.validate(epoch + 1)
                
                # 早停检查（以 avg_fgd 为准）
                if 'avg_fgd' in metrics:
                    current_fgd = metrics['avg_fgd']
                    if current_fgd < self.best_fgd - self.early_stopping_min_delta:
                        self.best_fgd = current_fgd
                        self.patience_counter = 0
                        logger.info(f"New best FGD: {current_fgd:.4f}, reset patience counter")
                    else:
                        self.patience_counter += 1
                        logger.info(f"FGD not improved for {self.patience_counter} epochs")
                        
                        if self.patience_counter >= self.early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                            self.early_stopped = True
                            break
        
        # 输出最佳指标
        if self.early_stopped:
            logger.info(f"Training stopped early after {epoch + 1} epochs!")
        else:
            logger.info("Training complete!")
        
        logger.info("Best metrics:")
        for metric_name, best_info in self.val_best.items():
            logger.info(f"  {metric_name}: {best_info['value']:.4f} (epoch {best_info['epoch']})")
    
    def load_checkpoint(self, checkpoint_path):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            epoch: 恢复后的起始epoch
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from epoch 0")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态（使用strict=False以允许模型结构变化）
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Loaded model state dict with strict=False (ignoring missing keys)")

        # 加载优化器状态
        if self.separate_gradient_update and self.gradient_update_mode == 'separate_optimizers':
            if 'speaker_optimizer_state_dict' in checkpoint:
                self.speaker_optimizer.load_state_dict(checkpoint['speaker_optimizer_state_dict'])
            if 'listener_optimizer_state_dict' in checkpoint:
                self.listener_optimizer.load_state_dict(checkpoint['listener_optimizer_state_dict'])
            # 加载学习率调度器状态
            if 'speaker_scheduler_state_dict' in checkpoint and hasattr(self, 'speaker_scheduler'):
                self.speaker_scheduler.load_state_dict(checkpoint['speaker_scheduler_state_dict'])
            if 'listener_scheduler_state_dict' in checkpoint and hasattr(self, 'listener_scheduler'):
                self.listener_scheduler.load_state_dict(checkpoint['listener_scheduler_state_dict'])
        else:
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 加载学习率调度器状态
            if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 获取epoch
        epoch = checkpoint.get('epoch', 0)

        # 加载最佳指标（如果存在）
        if 'val_best' in checkpoint:
            self.val_best = checkpoint['val_best']
            logger.info(f"Loaded best metrics from checkpoint")

        # 加载早停状态（如果存在）
        if 'best_fgd' in checkpoint:
            self.best_fgd = checkpoint['best_fgd']
            logger.info(f"Loaded best FGD: {self.best_fgd:.4f}")
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
            logger.info(f"Loaded patience counter: {self.patience_counter}")
        if 'early_stopped' in checkpoint:
            self.early_stopped = checkpoint['early_stopped']

        logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {epoch + 1}")

        # 返回下一个epoch
        return epoch + 1


# 入口点
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from models.bidirectional_dyadic_lsm import BidirectionalDyadicLSM
    from dataloaders.bidirectional_dyadic_dataset import BidirectionalDyadicDataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config)
    cfg.experiment_name = os.path.splitext(os.path.basename(args.config))[0]
    
    # 创建模型
    model = BidirectionalDyadicLSM(cfg)
    
    # 创建数据集
    train_ds = BidirectionalDyadicDataset(cfg.dataset.train.params, split='train')
    val_ds = BidirectionalDyadicDataset(cfg.dataset.val.params, split='val') if 'val' in cfg.dataset else None
    
    # 创建训练器
    trainer = BidirectionalDyadicTrainer(cfg, model, train_ds, val_ds)
    trainer.train(resume_from=args.resume)