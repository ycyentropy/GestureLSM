import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.LSM import GestureLSM, sample_beta_distribution
from loguru import logger


def reshape_coefs(t):
    return t.reshape((t.shape[0], 1, 1))


class OnlineListenerLoader:
    """
    在线 Listener 动作加载器
    模拟真实在线交互：按推理时间步逐步提供 Listener 历史动作
    """
    def __init__(self, listener_latents, context_size=8):
        """
        Args:
            listener_latents: [batch, total_frames, latent_dim] 完整 Listener 数据
            context_size: 历史帧数
        """
        self.listener_latents = listener_latents
        self.context_size = context_size
        self.batch_size = listener_latents.shape[0]
        self.latent_dim = listener_latents.shape[-1]
        self.device = listener_latents.device
    
    def get_history(self, frame_idx):
        """
        获取指定帧时刻的 Listener 历史动作
        
        Args:
            frame_idx: 当前帧索引（latent 空间）
        
        Returns:
            listener_window: [batch, context_size, latent_dim]
        """
        listener_start = max(0, frame_idx - self.context_size)
        listener_end = frame_idx
        
        if listener_end - listener_start < self.context_size:
            # 历史不足，用零填充（右对齐）
            listener_window = torch.zeros(
                self.batch_size, self.context_size, self.latent_dim,
                device=self.device
            )
            actual_len = listener_end - listener_start
            if actual_len > 0:
                listener_window[:, -actual_len:, :] = self.listener_latents[:, listener_start:listener_end, :]
            return listener_window
        else:
            return self.listener_latents[:, listener_start:listener_end, :]


class InteractiveGestureLSM(GestureLSM):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        logger.info("Initializing InteractiveGestureLSM with Listener Feedback")
        
        self.window_size = cfg.model.get("window_size", 18)
        self.context_size = cfg.model.get("context_size", 4)
        self.noise_range = cfg.model.get("noise_range", [0, 1])
        
        self.ladder_step = cfg.model.get("ladder_step", 1)
        self.inertia_weight = cfg.model.get("inertia_weight", 0.1)
        self.ofs_threshold = cfg.model.get("ofs_threshold", 0.8)
        
        self.consistency_ratio = cfg.model.get("consistency_ratio", 0.25)
        
        # 是否使用 listener 反馈（消融实验用）
        self.use_listener_feedback = cfg.model.get("use_listener_feedback", True)
        if not self.use_listener_feedback:
            logger.info("Listener feedback is DISABLED for ablation study")
        
        self.cond_drop_prob = cfg.model.get("cond_drop_prob", 0.1)
        
        self.context_noise_prob = cfg.model.get("context_noise_prob", 0.3)
        self.max_context_noise = cfg.model.get("max_context_noise", 0.1)
        
        self.listener_drop_prob = cfg.model.get("listener_drop_prob", 0.1)
        
        self.scheduled_sampling_prob = cfg.model.get("scheduled_sampling_prob", 0.0)
        if self.scheduled_sampling_prob > 0:
            logger.info(f"Scheduled Sampling enabled with probability {self.scheduled_sampling_prob}")
        
        self.loss_type = cfg.model.get("loss_type", "mse")
        self.huber_delta = cfg.model.get("huber_delta", 1.0)
        if self.loss_type not in ["mse", "huber"]:
            logger.warning(f"Unknown loss_type: {self.loss_type}, falling back to mse")
            self.loss_type = "mse"
        
        # 为 listener dropout 创建专门的 null embedding
        # listener_latent 的维度是 [batch, seq_len, input_dim * num_joints]
        listener_embed_dim = self.input_dim * self.num_joints
        self.null_listener_embed = nn.Parameter(
            torch.zeros(1, listener_embed_dim), requires_grad=True
        )
        
        self.window_buffer = None
        self.context_buffer = None
        
        if self.window_size % self.ladder_step != 0:
            logger.warning(f"window_size ({self.window_size}) should be divisible by ladder_step ({self.ladder_step})")
        
        self.use_dual_phase = cfg.model.get("use_dual_phase", True)
        self.pre_roll_prob = cfg.model.get("pre_roll_prob", 0.2)
        self.pre_roll_steps = cfg.model.get("pre_roll_steps", self.window_size)
        if self.use_dual_phase:
            logger.info(f"Dual-phase timestep scheduling enabled: pre_roll_prob={self.pre_roll_prob}, pre_roll_steps={self.pre_roll_steps}")
        
        self.timestep_weighting = cfg.model.get("timestep_weighting", "none")
        self.timestep_weight_scale = cfg.model.get("timestep_weight_scale", 1.0)
        if self.timestep_weighting != "none":
            logger.info(f"Timestep weighting enabled: mode={self.timestep_weighting}, scale={self.timestep_weight_scale}")
    
    def init_rolling_window(self, batch_size, device, idle_pose=None):
        self.window_buffer = torch.randn(
            batch_size, self.window_size, self.input_dim * self.num_joints,
            device=device
        )
        
        if idle_pose is not None:
            # idle_pose 已经是 [batch_size, seq_len, latent_dim] 形状
            self.context_buffer = idle_pose
        else:
            self.context_buffer = torch.randn(
                batch_size, self.context_size, self.input_dim * self.num_joints,
                device=device
            ) * 0.001
    
    def apply_rolling_noise(self, x0, batch_size, device, use_ladder=True):
        """
        应用滚动噪声调度
        
        Returns:
            x_t: 加噪后的数据
            t_values: 时间步序列
            noise: 噪声
            phase_mask: 阶段掩码 (True=预滚动阶段, False=滚动阶段)
        """
        if self.use_dual_phase:
            return self._apply_dual_phase_noise(x0, batch_size, device)
        elif use_ladder and self.ladder_step > 1:
            x_t, t_values, noise = self._apply_ladder_noise(x0, batch_size, device)
            phase_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return x_t, t_values, noise, phase_mask
        else:
            x_t, t_values, noise = self._apply_linear_noise(x0, batch_size, device)
            phase_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return x_t, t_values, noise, phase_mask
    
    def _apply_linear_noise(self, x0, batch_size, device):
        t_0 = torch.rand(batch_size, device=device) * 0.09 + 0.01
        
        t_values = []
        for n in range(self.window_size):
            t_n = 1.0 - t_0 - n * (1.0 - t_0) / (self.window_size - 1)
            t_values.append(t_n)
        t_values = torch.stack(t_values, dim=1).to(device)
        
        noise = torch.randn_like(x0)
        t_coef = t_values.unsqueeze(-1)
        x_t = t_coef * x0 + (1 - t_coef) * noise
        return x_t, t_values, noise
    
    def _get_pre_rolling_timesteps(self, batch_size, device):
        """
        RFLAV 预滚动阶段时间步调度器
        
        用于处理窗口初始化场景（从纯噪声开始生成）
        公式: t_k = clamp(1 - (k/T + w), 0, 1)
        
        当 w=1 时，所有帧都是纯噪声
        当 w=0 时，首帧无噪，末帧为全噪声（正好是滚动阶段的初始状态）
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            t_values: [batch_size, window_size] 时间步序列
        """
        T = self.window_size
        w = torch.rand(batch_size, device=device)
        
        k = torch.arange(T, device=device).float().unsqueeze(0)
        t_values = 1.0 - (k / T + w.unsqueeze(1))
        t_values = torch.clamp(t_values, min=0.0, max=1.0)
        
        return t_values
    
    def _get_rolling_timesteps(self, batch_size, device):
        """
        RFLAV 滚动阶段时间步调度器
        
        用于连续无限生成场景
        公式: t_k = 1 - (k + w) / T
        
        特性:
        - 相邻位置时间步间隔固定为 1/T
        - 噪声水平从前往后递增
        - 每个滚动步骤生成一帧完全去噪的帧
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            t_values: [batch_size, window_size] 时间步序列
        """
        T = self.window_size
        w = torch.rand(batch_size, device=device)
        
        k = torch.arange(T, device=device).float().unsqueeze(0)
        t_values = 1.0 - (k + w.unsqueeze(1)) / T
        
        return t_values
    
    def _apply_dual_phase_noise(self, x0, batch_size, device):
        """
        RFLAV 双阶段噪声调度
        
        训练时以 pre_roll_prob 概率采样预滚动阶段，
        以 (1 - pre_roll_prob) 概率采样滚动阶段
        
        这确保训练覆盖推理的所有场景，避免训练-推理不一致
        
        Returns:
            x_t: 加噪后的数据
            t_values: 时间步序列
            noise: 噪声
            phase_mask: 阶段掩码 (True=预滚动阶段, False=滚动阶段)
        """
        phase_mask = torch.rand(batch_size, device=device) < self.pre_roll_prob
        
        t_pre_roll = self._get_pre_rolling_timesteps(batch_size, device)
        t_roll = self._get_rolling_timesteps(batch_size, device)
        
        t_values = torch.where(phase_mask.unsqueeze(1), t_pre_roll, t_roll)
        
        noise = torch.randn_like(x0)
        t_coef = t_values.unsqueeze(-1)
        x_t = t_coef * x0 + (1 - t_coef) * noise
        
        return x_t, t_values, noise, phase_mask
    
    def _apply_ladder_noise(self, x0, batch_size, device):
        l = self.ladder_step
        N = self.window_size
        num_ladders = N // l
        
        t_0_l = torch.randint(1, num_ladders + 1, (batch_size,), device=device).float()
        
        t_values = []
        for n in range(N):
            k = n // l
            t_n = num_ladders + (N // l) - 1 - (t_0_l + k)
            t_values.append(t_n)
        t_values = torch.stack(t_values, dim=1).to(device)
        
        max_t = num_ladders + (N // l) - 1
        t_values = t_values / max_t
        
        noise = torch.randn_like(x0)
        t_coef = t_values.unsqueeze(-1)
        x_t = t_coef * x0 + (1 - t_coef) * noise
        return x_t, t_values, noise
    
    def apply_conditional_dropout(self, at_feat, cond_drop_prob=None):
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        
        batch_size = at_feat.shape[0]
        keep_mask = torch.rand(batch_size, device=at_feat.device) > cond_drop_prob
        
        # 获取与 at_feat 形状匹配的 null embedding
        target_seq_len = at_feat.shape[1]
        target_feat_dim = at_feat.shape[2]
        null_cond_embed = self.denoiser.get_null_cond_embed(target_seq_len, target_feat_dim, at_feat.device)
        
        at_feat_dropped = at_feat.clone()
        at_feat_dropped[~keep_mask] = null_cond_embed.unsqueeze(0).expand((~keep_mask).sum(), -1, -1)
        
        return at_feat_dropped
    
    def apply_context_noise(self, context_latents, phase_mask=None):
        """
        对 context 添加高斯噪声
        
        Args:
            context_latents: context 数据
            phase_mask: 阶段掩码 (True=预滚动阶段, False=滚动阶段)
                        预滚动阶段不添加噪声，因为 context 是 seed
        
        Returns:
            context_latents: 可能添加了噪声的 context
        """
        if not self.training:
            return context_latents
        
        batch_size = context_latents.shape[0]
        noise_mask = torch.rand(batch_size, device=context_latents.device) < self.context_noise_prob
        
        if phase_mask is not None:
            noise_mask = noise_mask & (~phase_mask)
        
        if noise_mask.any():
            noise = torch.randn_like(context_latents) * self.max_context_noise
            context_latents = context_latents + noise * noise_mask.unsqueeze(1).unsqueeze(2).float()
        
        return context_latents
    
    def apply_listener_dropout(self, listener_latent):
        if not self.training or listener_latent is None:
            return listener_latent
        
        batch_size = listener_latent.shape[0]
        drop_mask = torch.rand(batch_size, device=listener_latent.device) < self.listener_drop_prob
        
        if drop_mask.any():
            listener_latent = listener_latent.clone()
            target_seq_len = listener_latent.shape[1]
            # 使用专门的 null_listener_embed，并扩展到目标序列长度
            null_embed = self.null_listener_embed.expand(target_seq_len, -1)
            listener_latent[drop_mask] = null_embed.unsqueeze(0).expand(drop_mask.sum(), -1, -1)
        
        return listener_latent
    
    def apply_scheduled_sampling(self, context_latents, generated_buffer, window_start, device, phase_mask=None):
        """
        Scheduled Sampling: 以一定概率使用预测值替代真值
        
        Args:
            context_latents: 真值 context [batch, seq_len, dim] (flow_batch_size)
            generated_buffer: 已生成的预测值缓存 [batch, generated_len, dim]
            window_start: 当前窗口起始位置
            device: 设备
            phase_mask: 阶段掩码 (True=预滚动阶段, False=滚动阶段)
                        预滚动阶段不应用 scheduled sampling，因为 context 是 seed
            
        Returns:
            context_latents: 可能被替换为预测值的 context
        """
        if not self.training or self.scheduled_sampling_prob <= 0:
            return context_latents
        
        if generated_buffer is None or generated_buffer.shape[1] < self.context_size:
            return context_latents
        
        actual_batch_size = context_latents.shape[0]
        use_predicted_mask = torch.rand(actual_batch_size, device=device) < self.scheduled_sampling_prob
        
        if phase_mask is not None:
            use_predicted_mask = use_predicted_mask & (~phase_mask)
        
        if use_predicted_mask.any():
            context_latents = context_latents.clone()
            
            context_start_in_generated = max(0, window_start - self.context_size)
            context_end_in_generated = window_start
            
            if context_end_in_generated <= generated_buffer.shape[1]:
                predicted_context = generated_buffer[:, context_start_in_generated:context_end_in_generated, :]
                
                if predicted_context.shape[1] < self.context_size:
                    padding = torch.zeros(actual_batch_size, self.context_size - predicted_context.shape[1], 
                                         context_latents.shape[-1], device=device)
                    predicted_context = torch.cat([padding, predicted_context], dim=1)
                
                context_latents[use_predicted_mask] = predicted_context[use_predicted_mask]
        
        return context_latents
    
    def _compute_logit_normal_weight(self, t_values):
        """
        Logit-Normal 加权函数
        
        公式: λ_t = exp(-0.5 * (logit(t))^2) / (t * (1-t) * sqrt(2π))
        
        特性: 中间时间步权重高，两端权重低
        """
        t = t_values.clamp(min=1e-6, max=1-1e-6)
        logit_t = torch.log(t / (1 - t))
        weight = torch.exp(-0.5 * logit_t ** 2)
        weight = weight / (t * (1 - t) * torch.sqrt(torch.tensor(2 * torch.pi, device=t.device)))
        return weight
    
    def _compute_window_aware_weight(self, t_values, window_size):
        """
        窗口感知加权函数
        
        组合 Logit-Normal 加权和位置加权：
        - Logit-Normal: 中间时间步权重高
        - 位置加权: 首帧权重高（输出帧更重要）
        
        对于滚动生成，首帧是输出帧，应该有更高权重
        """
        T = window_size
        batch_size = t_values.shape[0]
        
        k = torch.arange(T, device=t_values.device).float()
        position_weight = 1.0 - k / T
        position_weight = position_weight.unsqueeze(0).expand(batch_size, -1)
        
        logit_weight = self._compute_logit_normal_weight(t_values)
        
        combined_weight = position_weight * logit_weight
        
        combined_weight = combined_weight / combined_weight.sum(dim=1, keepdim=True) * T
        
        return combined_weight
    
    def _compute_timestep_weights(self, t_values):
        """
        根据配置计算时间步权重
        
        Args:
            t_values: [batch_size, window_size] 时间步序列
            
        Returns:
            weights: [batch_size, window_size] 权重序列，或 None
        """
        if self.timestep_weighting == "none":
            return None
        elif self.timestep_weighting == "logit_normal":
            weights = self._compute_logit_normal_weight(t_values)
        elif self.timestep_weighting == "window_aware":
            weights = self._compute_window_aware_weight(t_values, self.window_size)
        else:
            logger.warning(f"Unknown timestep_weighting: {self.timestep_weighting}, using none")
            return None
        
        weights = weights * self.timestep_weight_scale
        
        return weights
    
    def compute_rdla_loss(self, pred, target, t_values=None):
        """
        RDLA损失函数 = Flow Loss + Inertia约束项
        
        公式: L_RDLA = Σ||x_{j+n}^0 - x̂_{j+n}||² - 2λ * Σ⟨error_n, error_{n+1}⟩
        
        其中:
        - 第一项: 基础损失 (帧预测误差, 支持MSE或Huber)
        - 第二项: 惯性约束项 (相邻帧误差内积)
        - error_n = x_{j+n}^0 - x̂_{j+n} (第n帧的预测误差)
        - λ = inertia_weight (惯性权重)
        
        作用:
        - 基础损失: 确保每个帧的生成精度
        - 惯性约束: 内积为正奖励平滑，内积为负惩罚突变
        
        设计背景:
        RDLA通过阶梯式噪声调度同时去噪l帧，虽提升了速度，但可能导致帧块间过渡突兀。
        inertia约束项正是为缓解该问题而提出的针对性约束。
        
        时间步加权:
        - 如果提供 t_values，根据配置计算权重并应用
        - 支持 logit_normal 和 window_aware 两种加权模式
        """
        error = pred - target
        
        if self.loss_type == "huber":
            abs_error = torch.abs(error)
            quadratic = torch.min(abs_error, torch.tensor(self.huber_delta, device=error.device))
            linear = abs_error - quadratic
            frame_loss = (0.5 * quadratic ** 2 + self.huber_delta * linear).mean(dim=-1)
        else:
            frame_loss = (error ** 2).mean(dim=-1)
        
        timestep_weights = self._compute_timestep_weights(t_values) if t_values is not None else None
        
        if timestep_weights is not None:
            weighted_frame_loss = frame_loss * timestep_weights
            base_loss = weighted_frame_loss.mean()
        else:
            base_loss = frame_loss.mean()
        
        if self.inertia_weight > 0:
            error_n = error[:, :-1, :]
            error_n_plus_1 = error[:, 1:, :]
            
            inner_product = (error_n * error_n_plus_1).mean(dim=-1)
            
            inertia_term = -1.0 * self.inertia_weight * inner_product.mean()
        else:
            inertia_term = 0.0
        
        rdla_loss = base_loss + inertia_term
        
        return rdla_loss
    
    def get_audio_window(self, audio_features, frame_idx, batch_size, device):
        audio_start = frame_idx
        audio_end = min(audio_features.shape[1], frame_idx + self.window_size)
        audio_window = audio_features[:, audio_start:audio_end, :]
        
        audio_window_len = audio_end - audio_start
        target_audio_len = self.window_size
        
        if audio_window_len < target_audio_len:
            padding = torch.zeros(batch_size, target_audio_len - audio_window_len, audio_features.shape[-1], device=device)
            audio_window = torch.cat([audio_window, padding], dim=1)
        
        return audio_window
    
    def roll_window(self, new_frame, step=1):
        if new_frame.dim() == 2:
            new_frame = new_frame.unsqueeze(1)
        
        self.window_buffer = torch.cat([
            self.window_buffer[:, step:, :],
            new_frame
        ], dim=1)
    
    def roll_window_with_context(self, new_frame, step=1):
        if new_frame.dim() == 2:
            new_frame = new_frame.unsqueeze(1)
        
        self.window_buffer = torch.cat([
            self.window_buffer[:, step:, :],
            new_frame
        ], dim=1)
        
        self.context_buffer = torch.cat([
            self.context_buffer[:, step:, :],
            self.window_buffer[:, :step, :]
        ], dim=1)
        
        if self.context_buffer.shape[1] > self.context_size:
            self.context_buffer = self.context_buffer[:, -self.context_size:, :]
    
    def apply_ofs_smoothing(self, frames, threshold=None):
        if threshold is None:
            threshold = self.ofs_threshold
        
        if frames.shape[1] < 3:
            return frames
        
        smoothed = frames.clone()
        
        for i in range(1, frames.shape[1] - 1):
            prev_frame = frames[:, i-1, :]
            next_frame = frames[:, i+1, :]
            
            cos_sim = F.cosine_similarity(prev_frame, next_frame, dim=-1)
            
            mask = cos_sim < threshold
            if mask.any():
                smoothed[:, i, :][mask] = (prev_frame[mask] + next_frame[mask]) / 2
        
        return smoothed
    
    def train_forward(self, condition_dict, latents, listener_latents=None):
        batch_size, seq_len, latent_dim = latents.shape
        device = latents.device
        
        losses = {}
        
        flow_batch_size = int(batch_size * (1 - self.consistency_ratio))
        
        seed = condition_dict['y']['seed']
        pre_frames = seed.shape[1]
        
        audio_features = self.modality_encoder(
            condition_dict['y']['audio_onset'],
            condition_dict['y']['word']
        )
        
        if self.use_id:
            id_tensor = condition_dict['y']['id'][:, 0]
            id_emb = self.id_embeddings(id_tensor)
            id_emb_expanded = id_emb.unsqueeze(1).repeat(1, audio_features.shape[1], 1)
            audio_features = torch.cat([audio_features, id_emb_expanded], dim=-1)
        
        window_start = pre_frames
        window_end = window_start + self.window_size
        
        window_latents = latents[:, window_start:window_end, :]
        
        x_t, t_values, noise, phase_mask = self.apply_rolling_noise(
            window_latents, batch_size, device, use_ladder=True
        )
        
        context_latents = latents[:, max(0, window_start - self.context_size):window_start, :]
        if context_latents.shape[1] < self.context_size:
            padding = torch.zeros(batch_size, self.context_size - context_latents.shape[1], latent_dim, device=device)
            context_latents = torch.cat([padding, context_latents], dim=1)
        
        context_latents = self.apply_context_noise(context_latents, phase_mask)
        
        audio_window = audio_features[:, window_start:window_end, :]
        audio_window_len = audio_window.shape[1]
        target_audio_len = self.window_size
        if audio_window_len < target_audio_len:
            padding = torch.zeros(batch_size, target_audio_len - audio_window_len, audio_features.shape[-1], device=device)
            audio_window = torch.cat([audio_window, padding], dim=1)
        audio_features_flow = self.apply_conditional_dropout(audio_window[:flow_batch_size])
        
        listener_window = None
        if listener_latents is not None and self.use_listener_feedback:
            listener_start = max(0, window_start - self.context_size)
            listener_end = window_start
            if listener_end - listener_start < self.context_size:
                listener_window = torch.zeros(batch_size, self.context_size, listener_latents.shape[-1], device=device)
                actual_len = listener_end - listener_start
                if actual_len > 0:
                    listener_window[:, -actual_len:, :] = listener_latents[:, listener_start:listener_end, :]
            else:
                listener_window = listener_latents[:, listener_start:listener_end, :]
            listener_window = self.apply_listener_dropout(listener_window)
        
        full_input_4d = x_t.reshape(batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        model_output = self.denoiser(
            x=full_input_4d[:flow_batch_size],
            timesteps=t_values[:flow_batch_size, 0],
            seed=condition_dict['y']['seed'][:flow_batch_size],
            at_feat=audio_features_flow,
            listener_latent=listener_window[:flow_batch_size] if listener_window is not None else None
        )
        model_output = model_output.permute(0, 3, 1, 2).reshape(flow_batch_size, -1, self.num_joints * self.input_dim)
        
        window_output = model_output
        
        if self.flow_mode == "v":
            target = window_latents[:flow_batch_size] - noise[:flow_batch_size]
        else:
            target = window_latents[:flow_batch_size]
        
        window_t_values = t_values[:flow_batch_size]
        flow_loss = self.compute_rdla_loss(window_output, target, window_t_values)
        
        losses["flow_loss"] = flow_loss
        
        consistency_loss = torch.tensor(0.0, device=device)
        if flow_batch_size < batch_size:
            consistency_loss = self._compute_consistency_loss(
                condition_dict, latents, audio_features, batch_size, flow_batch_size,
                device, window_start, window_end, listener_latents
            )
        losses["consistency_loss"] = consistency_loss
        
        total_loss = losses["flow_loss"] + losses["consistency_loss"]
        losses["loss"] = total_loss
        
        return losses
    
    def _compute_consistency_loss(self, condition_dict, latents, audio_features, 
                                   batch_size, flow_batch_size, device, 
                                   window_start, window_end, listener_latents=None):
        cons_start = flow_batch_size
        cons_batch_size = batch_size - flow_batch_size
        
        if cons_batch_size <= 0:
            return torch.tensor(0.0, device=device)
        
        window_latents = latents[cons_start:, window_start:window_end, :]
        latent_dim = window_latents.shape[-1]
        
        context_latents = latents[cons_start:, max(0, window_start - self.context_size):window_start, :]
        if context_latents.shape[1] < self.context_size:
            padding = torch.zeros(cons_batch_size, self.context_size - context_latents.shape[1], latent_dim, device=device)
            context_latents = torch.cat([padding, context_latents], dim=1)
        context_latents = self.apply_context_noise(context_latents)
        
        seed = condition_dict['y']['seed'][cons_start:]
        
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(device)
        delta_probs = torch.ones((deltas.shape[0],), device=device) / deltas.shape[0]
        
        t = sample_beta_distribution(cons_batch_size, alpha=2, beta=1.2).to(device)
        d = deltas[delta_probs.multinomial(cons_batch_size, replacement=True)]
        
        x0_noise = torch.randn_like(window_latents)
        t_coef = reshape_coefs(t)
        x_t = t_coef * window_latents + (1 - t_coef) * x0_noise
        
        force_cfg = np.random.choice([True, False], size=cons_batch_size, p=[0.8, 0.2])
        
        audio_window = audio_features[cons_start:, window_start:window_end, :]
        audio_window_len = audio_window.shape[1]
        target_audio_len = self.window_size
        if audio_window_len < target_audio_len:
            padding = torch.zeros(cons_batch_size, target_audio_len - audio_window_len, audio_features.shape[-1], device=device)
            audio_window = torch.cat([audio_window, padding], dim=1)
        audio_features_cons = self._apply_force_cfg(audio_window, force_cfg)
        
        listener_window = None
        if listener_latents is not None and self.use_listener_feedback:
            listener_start = max(0, window_start - self.context_size)
            listener_end = window_start
            if listener_end - listener_start < self.context_size:
                listener_window = torch.zeros(cons_batch_size, self.context_size, listener_latents.shape[-1], device=device)
                actual_len = listener_end - listener_start
                if actual_len > 0:
                    listener_window[:, -actual_len:, :] = listener_latents[cons_start:, listener_start:listener_end, :]
            else:
                listener_window = listener_latents[cons_start:, listener_start:listener_end, :]
        
        full_input_4d = x_t.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        
        with torch.no_grad():
            pred_t = self.denoiser(
                x=full_input_4d,
                timesteps=t,
                seed=seed,
                at_feat=audio_features_cons,
                cond_time=d,
                listener_latent=listener_window
            )
            pred_t = pred_t.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, self.num_joints * self.input_dim)
            pred_t_window = pred_t
            
            d_coef = reshape_coefs(d)
            if self.flow_mode == "v":
                speed_t = pred_t_window
            else:
                speed_t = pred_t_window - x0_noise
            
            x_td = x_t + d_coef * speed_t
            full_input_td_4d = x_td.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
            
            pred_td = self.denoiser(
                x=full_input_td_4d,
                timesteps=t + d,
                seed=seed,
                at_feat=audio_features_cons,
                cond_time=d,
                listener_latent=listener_window
            )
            pred_td = pred_td.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, self.num_joints * self.input_dim)
            pred_td_window = pred_td
            
            if self.flow_mode == "v":
                speed_td = pred_td_window
            else:
                speed_td = pred_td_window - x0_noise
            
            speed_target = (speed_t + speed_td) / 2
        
        model_pred = self.denoiser(
            x=full_input_4d,
            timesteps=t,
            seed=seed,
            at_feat=audio_features_cons,
            cond_time=2 * d,
            listener_latent=listener_window
        )
        model_pred = model_pred.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, self.num_joints * self.input_dim)
        model_pred_window = model_pred
        
        if self.flow_mode == "v":
            speed_pred = model_pred_window
        else:
            speed_pred = model_pred_window - x0_noise
        
        if self.loss_type == "huber":
            consistency_loss = F.huber_loss(speed_pred, speed_target, reduction="mean", delta=self.huber_delta)
        else:
            consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")
        
        return consistency_loss
    
    def _apply_force_cfg(self, at_feat, force_cfg):
        batch_size = at_feat.shape[0]
        target_seq_len = at_feat.shape[1]
        target_feat_dim = at_feat.shape[2]
        null_cond_embed = self.denoiser.get_null_cond_embed(target_seq_len, target_feat_dim, at_feat.device)
        
        at_feat_forced = at_feat.clone()
        force_cfg_tensor = torch.tensor(force_cfg, device=at_feat.device)
        if force_cfg_tensor.sum() > 0:
            at_feat_forced[force_cfg_tensor] = null_cond_embed.unsqueeze(0).expand(force_cfg_tensor.sum(), -1, -1)
        
        return at_feat_forced
    
    def _get_inference_t_values(self, batch_size, window_size, t_start, device):
        t_values = []
        for n in range(window_size):
            t_n = t_start + n * (1.0 - t_start) / (window_size - 1)
            t_values.append(t_n)
        t_values = torch.stack(t_values, dim=0).unsqueeze(0).expand(batch_size, -1).to(device)
        return t_values
    
    def _get_ladder_t_values(self, batch_size, window_size, ladder_step, t_0_l, device):
        """
        计算RDLA阶梯噪声调度的时间值
        
        Args:
            batch_size: 批次大小
            window_size: 窗口大小 N
            ladder_step: 阶梯步长 l
            t_0_l: 当前阶梯噪声水平 (1, 2, ..., l)
            device: 设备
        
        Returns:
            t_values: [batch_size, window_size] 每帧的时间值
        """
        l = ladder_step
        N = window_size
        num_ladders = N // l
        
        t_values = []
        for n in range(N):
            k = n // l
            t_n = t_0_l + k * l
            t_values.append(t_n)
        t_values = torch.tensor(t_values, device=device).unsqueeze(0).expand(batch_size, -1)
        
        max_t = t_values[:, -1:].clamp(min=1.0)
        t_values = t_values / max_t
        
        return t_values
    
    @torch.no_grad()
    def generate_rdla(self, condition_dict, audio_features, listener_latents=None,
                      guidance_scale=2.0):
        """
        RDLA推理流程：每次迭代只进行1步去噪，然后平移l帧
        
        核心流程：
        1. 每次迭代只进行1步去噪
        2. 窗口平移l帧
        3. 经过(阶梯数)次迭代后，完整窗口处理完毕
        
        去噪步数 = 阶梯数 = window_size / ladder_step
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        self.init_rolling_window(batch_size, device, idle_pose=seed)
        
        generated_sequence = []
        total_frames = audio_features.shape[1]
        
        l = self.ladder_step
        N = self.window_size
        num_ladders = N // l  # 阶梯数 = 去噪步数
        
        pre_frames = condition_dict['y']['seed'].shape[1]
        
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        
        t_0_l = num_ladders  # 初始噪声水平 = 阶梯数
        
        while frame_idx < total_frames:
            frames_to_generate = min(l, total_frames - frame_idx)
            
            dt = 1.0 / num_ladders
            
            t_values = self._get_ladder_t_values_v2(batch_size, N, l, num_ladders, t_0_l, device)
            
            full_input = self.window_buffer
            
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            listener_window = None
            if listener_latents is not None and self.use_listener_feedback:
                listener_start = max(0, frame_idx - self.context_size)
                listener_end = frame_idx
                if listener_end - listener_start < self.context_size:
                    listener_window = torch.zeros(batch_size, self.context_size, listener_latents.shape[-1], device=device)
                    actual_len = listener_end - listener_start
                    if actual_len > 0:
                        listener_window[:, -actual_len:, :] = listener_latents[:, listener_start:listener_end, :]
                else:
                    listener_window = listener_latents[:, listener_start:listener_end, :]
            
            if guidance_scale > 1.0:
                v_pred = self._guided_denoise_rdla(
                    full_input_4d, t_values, condition_dict, 
                    audio_features, frame_idx, frames_to_generate, guidance_scale,
                    listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                
                timesteps = t_values[:, 0] if t_values.dim() == 2 else t_values
                v_pred = self.denoiser(
                    x=full_input_4d,
                    timesteps=timesteps,
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred = v_pred.squeeze(2)
            v_pred_3d = v_pred.permute(0, 2, 1)
            
            window_frames = self.window_buffer.shape[1]
            self.window_buffer = self.window_buffer + dt * v_pred_3d[:, -window_frames:, :]
            
            t_0_l -= 1
            
            output_frames = self.window_buffer[:, :frames_to_generate, :]
            
            if frames_to_generate > 1 and self.ofs_threshold < 1.0:
                output_frames = self.apply_ofs_smoothing(output_frames)
            
            for i in range(frames_to_generate):
                generated_sequence.append(output_frames[:, i, :])
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, frames_to_generate:, :],
                output_frames
            ], dim=1)
            
            if self.context_buffer.shape[1] > self.context_size:
                self.context_buffer = self.context_buffer[:, -self.context_size:, :]
            
            if t_0_l == 0:
                self.window_buffer = torch.randn(
                    batch_size, self.window_size, self.input_dim * self.num_joints, device=device
                )
                t_0_l = num_ladders
            else:
                new_noise_frames = torch.randn(
                    batch_size, frames_to_generate, self.input_dim * self.num_joints, device=device
                )
                self.window_buffer = torch.cat([
                    self.window_buffer[:, frames_to_generate:, :],
                    new_noise_frames
                ], dim=1)
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    def _get_ladder_t_values_v2(self, batch_size, window_size, ladder_step, num_ladders, t_0_l, device):
        """
        计算RDLA阶梯噪声调度的时间值 (v2: 基于阶梯数)
        
        Args:
            batch_size: 批次大小
            window_size: 窗口大小 N
            ladder_step: 阶梯步长 l
            num_ladders: 阶梯数 = N / l
            t_0_l: 当前噪声水平 (num_ladders, num_ladders-1, ..., 1)
            device: 设备
        
        Returns:
            t_values: [batch_size, window_size] 每帧的时间值
        """
        l = ladder_step
        N = window_size
        
        t_values = []
        for n in range(N):
            k = n // l  # 阶梯索引
            t_n = t_0_l + k
            t_values.append(t_n)
        t_values = torch.tensor(t_values, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        
        max_t = num_ladders + (N // l) - 1
        t_values = t_values / max_t
        
        return t_values
    
    def _guided_denoise_rdla(self, x, t_values, condition_dict, audio_features, frame_idx, frames_to_generate, guidance_scale, listener_window=None):
        batch_size = x.shape[0]
        device = x.device
        
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([condition_dict['y']['seed']] * 2, dim=0)
        
        at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
        
        target_seq_len = at_feat.shape[1]
        target_feat_dim = at_feat.shape[2]
        null_cond_embed = self.denoiser.get_null_cond_embed(target_seq_len, target_feat_dim, at_feat.device)
        
        at_feat_uncond = null_cond_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        listener_doubled = None
        if listener_window is not None:
            listener_doubled = torch.cat([listener_window, listener_window], dim=0)
        
        if t_values.dim() == 2:
            timesteps_doubled = torch.cat([t_values[:, 0], t_values[:, 0]], dim=0)
        else:
            timesteps_doubled = torch.cat([t_values, t_values], dim=0)
        
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
            listener_latent=listener_doubled
        )
        
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output
    
    @torch.no_grad()
    def generate_rflav(self, condition_dict, audio_features, listener_latents=None,
                       guidance_scale=2.0, num_iterations=None):
        """
        RFLAV 风格推理流程：使用双阶段时间步调度
        
        核心流程：
        1. 预滚动阶段：从纯噪声开始，逐步初始化窗口
        2. 滚动阶段：连续生成，每次生成一帧
        
        与 RDLA 的区别：
        - 使用连续线性时间步调度，而非阶梯式
        - 预滚动阶段确保训练-推理一致性
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        pre_frames = seed.shape[1]
        T = self.window_size
        total_frames = audio_features.shape[1]
        
        self.window_buffer = torch.randn(
            batch_size, T, self.input_dim * self.num_joints, device=device
        )
        self.context_buffer = seed.clone()
        
        generated_sequence = []
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        
        is_pre_roll_phase = True
        pre_roll_step = 0
        num_pre_roll_steps = self.pre_roll_steps
        
        for _ in range(num_pre_roll_steps):
            w = 1.0 - pre_roll_step / num_pre_roll_steps
            k = torch.arange(T, device=device).float()
            t_values = torch.clamp(1.0 - (k / T + w), min=0.0, max=1.0)
            t_values = t_values.unsqueeze(0).expand(batch_size, -1)
            
            dt = 1.0 / num_pre_roll_steps
            
            full_input = self.window_buffer
            
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            listener_window = None
            if listener_latents is not None and self.use_listener_feedback:
                listener_start = max(0, frame_idx - self.context_size)
                listener_end = frame_idx
                if listener_end - listener_start < self.context_size:
                    listener_window = torch.zeros(batch_size, self.context_size, listener_latents.shape[-1], device=device)
                    actual_len = listener_end - listener_start
                    if actual_len > 0:
                        listener_window[:, -actual_len:, :] = listener_latents[:, listener_start:listener_end, :]
                else:
                    listener_window = listener_latents[:, listener_start:listener_end, :]
            
            if guidance_scale > 1.0:
                v_pred = self._guided_denoise_rflav(
                    full_input_4d, t_values, condition_dict, 
                    audio_features, frame_idx, guidance_scale, listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                v_pred = self.denoiser(
                    x=full_input_4d,
                    timesteps=t_values[:, 0],
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred = v_pred.squeeze(2)
            v_pred_3d = v_pred.permute(0, 2, 1)
            
            self.window_buffer = self.window_buffer + dt * v_pred_3d
            
            pre_roll_step += 1
        
        w_curr = 0.0
        
        while frame_idx < total_frames:
            k = torch.arange(T, device=device).float()
            t_curr = 1.0 - (k + w_curr) / T
            t_curr = t_curr.unsqueeze(0).expand(batch_size, -1)
            
            full_input = self.window_buffer
            
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            listener_window = None
            if listener_latents is not None and self.use_listener_feedback:
                listener_start = max(0, frame_idx - self.context_size)
                listener_end = frame_idx
                if listener_end - listener_start < self.context_size:
                    listener_window = torch.zeros(batch_size, self.context_size, listener_latents.shape[-1], device=device)
                    actual_len = listener_end - listener_start
                    if actual_len > 0:
                        listener_window[:, -actual_len:, :] = listener_latents[:, listener_start:listener_end, :]
                else:
                    listener_window = listener_latents[:, listener_start:listener_end, :]
            
            if guidance_scale > 1.0:
                v_pred = self._guided_denoise_rflav(
                    full_input_4d, t_curr, condition_dict, 
                    audio_features, frame_idx, guidance_scale, listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                v_pred = self.denoiser(
                    x=full_input_4d,
                    timesteps=t_curr[:, 0],
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred = v_pred.squeeze(2)
            v_pred_3d = v_pred.permute(0, 2, 1)
            
            delta_t_full = (1.0 - t_curr).unsqueeze(-1)
            
            window_full = self.window_buffer + delta_t_full * v_pred_3d
            
            clean_frame = window_full[:, 0:1, :]
            generated_sequence.append(clean_frame.squeeze(1))
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, 1:, :],
                clean_frame
            ], dim=1)
            if self.context_buffer.shape[1] > self.context_size:
                self.context_buffer = self.context_buffer[:, -self.context_size:, :]
            
            window_remain = window_full[:, 1:, :]
            
            new_noise = torch.randn(batch_size, 1, self.input_dim * self.num_joints, device=device)
            self.window_buffer = torch.cat([window_remain, new_noise], dim=1)
            
            dt = 1.0 / T
            w_curr = (w_curr + dt) % 1.0
            
            k = torch.arange(T, device=device).float()
            t_new = 1.0 - (k + w_curr) / T
            t_new = t_new.unsqueeze(0).expand(batch_size, -1)
            
            full_input = self.window_buffer
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            if guidance_scale > 1.0:
                v_pred_incr = self._guided_denoise_rflav(
                    full_input_4d, t_new, condition_dict, 
                    audio_features, frame_idx, guidance_scale, listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                v_pred_incr = self.denoiser(
                    x=full_input_4d,
                    timesteps=t_new[:, 0],
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred_incr = v_pred_incr.squeeze(2)
            v_pred_incr_3d = v_pred_incr.permute(0, 2, 1)
            
            self.window_buffer = self.window_buffer + dt * v_pred_incr_3d
            
            frame_idx += 1
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    def _guided_denoise_rflav(self, x, t_values, condition_dict, audio_features, frame_idx, guidance_scale, listener_window=None):
        """
        RFLAV 风格的引导去噪
        
        与 RDLA 版本的区别：使用完整的时间步序列而非单一时间步
        """
        batch_size = x.shape[0]
        device = x.device
        
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([condition_dict['y']['seed']] * 2, dim=0)
        
        at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
        
        target_seq_len = at_feat.shape[1]
        target_feat_dim = at_feat.shape[2]
        null_cond_embed = self.denoiser.get_null_cond_embed(target_seq_len, target_feat_dim, at_feat.device)
        
        at_feat_uncond = null_cond_embed.unsqueeze(0).expand(batch_size, -1, -1)
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        listener_doubled = None
        if listener_window is not None:
            listener_doubled = torch.cat([listener_window, listener_window], dim=0)
        
        timesteps_doubled = torch.cat([t_values[:, 0], t_values[:, 0]], dim=0)
        
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
            listener_latent=listener_doubled
        )
        
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output
    
    @torch.no_grad()
    def generate_online_rflav(self, condition_dict, audio_features, listener_loader,
                              guidance_scale=2.0, total_frames=None):
        """
        RFLAV 风格在线推理流程：使用双阶段时间步调度
        
        适用于在线交互场景，Listener 数据通过 listener_loader 逐步获取
        
        Args:
            total_frames: 生成总帧数（latent长度）。如果为None，则使用audio_features.shape[1]
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        pre_frames = seed.shape[1]
        self.pre_frames = pre_frames
        T = self.window_size
        if total_frames is None:
            total_frames = audio_features.shape[1]
        
        self.window_buffer = torch.randn(
            batch_size, T, self.input_dim * self.num_joints, device=device
        )
        self.context_buffer = seed.clone()
        
        generated_sequence = []
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        
        is_pre_roll_phase = True
        pre_roll_step = 0
        num_pre_roll_steps = self.pre_roll_steps
        
        while frame_idx < total_frames:
            if is_pre_roll_phase:
                w = 1.0 - pre_roll_step / num_pre_roll_steps
                k = torch.arange(T, device=device).float()
                t_values = torch.clamp(1.0 - (k / T + w), min=0.0, max=1.0)
                t_values = t_values.unsqueeze(0).expand(batch_size, -1)
                
                dt = 1.0 / num_pre_roll_steps
                
                pre_roll_step += 1
                if pre_roll_step >= num_pre_roll_steps:
                    is_pre_roll_phase = False
            else:
                w = 0.0
                k = torch.arange(T, device=device).float()
                t_values = 1.0 - (k + w) / T
                t_values = t_values.unsqueeze(0).expand(batch_size, -1)
                
                dt = 1.0 / T
            
            full_input = self.window_buffer
            
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            listener_window = None
            if self.use_listener_feedback:
                listener_window = listener_loader.get_history(frame_idx)
            
            if guidance_scale > 1.0:
                v_pred = self._guided_denoise_rflav(
                    full_input_4d, t_values, condition_dict, 
                    audio_features, frame_idx, guidance_scale, listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                v_pred = self.denoiser(
                    x=full_input_4d,
                    timesteps=t_values[:, 0],
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred = v_pred.squeeze(2)
            v_pred_3d = v_pred.permute(0, 2, 1)
            
            window_frames = self.window_buffer.shape[1]
            self.window_buffer = self.window_buffer + dt * v_pred_3d[:, -window_frames:, :]
            
            output_frame = self.window_buffer[:, 0:1, :]
            generated_sequence.append(output_frame.squeeze(1))
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, 1:, :],
                output_frame
            ], dim=1)
            if self.context_buffer.shape[1] > pre_frames:
                self.context_buffer = self.context_buffer[:, -pre_frames:, :]
            
            self.window_buffer = torch.cat([
                self.window_buffer[:, 1:, :],
                torch.randn(batch_size, 1, self.input_dim * self.num_joints, device=device)
            ], dim=1)
            
            frame_idx += 1
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence

    @torch.no_grad()
    def generate(self, condition_dict, audio_features, listener_latents=None,
                 num_steps=2, guidance_scale=2.0):
        """
        标准推理流程：每次迭代进行完整的num_steps步去噪
        
        注意：这是原始实现，不符合RDLA标准流程
        推荐使用 generate_rdla() 方法
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        self.init_rolling_window(batch_size, device, idle_pose=seed)
        
        epsilon = 1e-8
        timesteps = torch.linspace(epsilon, 1 - epsilon, num_steps + 1, device=device)
        
        generated_sequence = []
        total_frames = audio_features.shape[1]
        
        l = self.ladder_step
        
        pre_frames = condition_dict['y']['seed'].shape[1]
        
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        while frame_idx < total_frames:
            frames_to_generate = min(l, total_frames - frame_idx)
            
            for step in range(num_steps):
                t_curr = timesteps[step]
                t_next = timesteps[step + 1]
                dt = t_next - t_curr
                
                full_input = self.window_buffer
                
                nframes = full_input.shape[1]
                full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
                full_input_4d = full_input_4d.permute(0, 2, 3, 1)
                
                listener_window = None
                if listener_latents is not None and self.use_listener_feedback:
                    listener_start = max(0, frame_idx - self.context_size)
                    listener_end = frame_idx
                    if listener_end - listener_start < self.context_size:
                        listener_window = torch.zeros(batch_size, self.context_size, listener_latents.shape[-1], device=device)
                        actual_len = listener_end - listener_start
                        if actual_len > 0:
                            listener_window[:, -actual_len:, :] = listener_latents[:, listener_start:listener_end, :]
                    else:
                        listener_window = listener_latents[:, listener_start:listener_end, :]
                
                if guidance_scale > 1.0:
                    v_pred = self._guided_denoise(
                        full_input_4d, t_curr, condition_dict, 
                        audio_features, frame_idx, frames_to_generate, guidance_scale,
                        listener_window
                    )
                else:
                    at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                    
                    v_pred = self.denoiser(
                        x=full_input_4d,
                        timesteps=torch.full((batch_size,), t_curr, device=device),
                        seed=condition_dict['y']['seed'],
                        at_feat=at_feat,
                        listener_latent=listener_window
                    )
                
                v_pred = v_pred.squeeze(2)
                v_pred_3d = v_pred.permute(0, 2, 1)
                
                window_frames = self.window_buffer.shape[1]
                self.window_buffer = self.window_buffer + dt * v_pred_3d[:, -window_frames:, :]
            
            output_frames = self.window_buffer[:, :frames_to_generate, :]
            
            if frames_to_generate > 1 and self.ofs_threshold < 1.0:
                output_frames = self.apply_ofs_smoothing(output_frames)
            
            for i in range(frames_to_generate):
                generated_sequence.append(output_frames[:, i, :])
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, frames_to_generate:, :],
                output_frames
            ], dim=1)
            
            if self.context_buffer.shape[1] > self.context_size:
                self.context_buffer = self.context_buffer[:, -self.context_size:, :]
            
            self.window_buffer = torch.randn(
                batch_size, self.window_size, self.input_dim * self.num_joints, device=device
            )
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    @torch.no_grad()
    def generate_online(self, condition_dict, audio_features, listener_loader,
                        num_steps=2, guidance_scale=2.0):
        """
        标准在线推理流程：每次迭代进行完整的num_steps步去噪
        
        注意：这是原始实现，不符合RDLA标准流程
        推荐使用 generate_online_rdla() 方法
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        self.init_rolling_window(batch_size, device, idle_pose=seed)
        
        epsilon = 1e-8
        timesteps = torch.linspace(epsilon, 1 - epsilon, num_steps + 1, device=device)
        
        generated_sequence = []
        total_frames = audio_features.shape[1]
        
        l = self.ladder_step
        
        pre_frames = condition_dict['y']['seed'].shape[1]
        
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        while frame_idx < total_frames:
            frames_to_generate = min(l, total_frames - frame_idx)
            
            for step in range(num_steps):
                t_curr = timesteps[step]
                t_next = timesteps[step + 1]
                dt = t_next - t_curr
                
                full_input = self.window_buffer
                
                nframes = full_input.shape[1]
                full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
                full_input_4d = full_input_4d.permute(0, 2, 3, 1)
                
                listener_window = None
                if self.use_listener_feedback:
                    listener_window = listener_loader.get_history(frame_idx)
                
                if guidance_scale > 1.0:
                    v_pred = self._guided_denoise(
                        full_input_4d, t_curr, condition_dict, 
                        audio_features, frame_idx, frames_to_generate, guidance_scale,
                        listener_window
                    )
                else:
                    at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                    
                    v_pred = self.denoiser(
                        x=full_input_4d,
                        timesteps=torch.full((batch_size,), t_curr, device=device),
                        seed=condition_dict['y']['seed'],
                        at_feat=at_feat,
                        listener_latent=listener_window
                    )
                
                v_pred = v_pred.squeeze(2)
                v_pred_3d = v_pred.permute(0, 2, 1)
                
                window_frames = self.window_buffer.shape[1]
                self.window_buffer = self.window_buffer + dt * v_pred_3d[:, -window_frames:, :]
            
            output_frames = self.window_buffer[:, :frames_to_generate, :]
            
            if frames_to_generate > 1 and self.ofs_threshold < 1.0:
                output_frames = self.apply_ofs_smoothing(output_frames)
            
            for i in range(frames_to_generate):
                generated_sequence.append(output_frames[:, i, :])
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, frames_to_generate:, :],
                output_frames
            ], dim=1)
            
            if self.context_buffer.shape[1] > self.context_size:
                self.context_buffer = self.context_buffer[:, -self.context_size:, :]
            
            self.window_buffer = torch.randn(
                batch_size, self.window_size, self.input_dim * self.num_joints, device=device
            )
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    @torch.no_grad()
    def generate_online_rdla(self, condition_dict, audio_features, listener_loader,
                             guidance_scale=2.0, total_frames=None):
        """
        RDLA在线推理流程：每次迭代只进行1步去噪，然后平移l帧
        
        核心流程：
        1. 每次迭代只进行1步去噪
        2. 窗口平移l帧
        3. 经过(阶梯数)次迭代后，完整窗口处理完毕
        
        去噪步数 = 阶梯数 = window_size / ladder_step
        
        Args:
            total_frames: 生成总帧数（latent长度）。如果为None，则使用audio_features.shape[1]
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        seed = condition_dict['y']['seed']
        self.init_rolling_window(batch_size, device, idle_pose=seed)
        
        generated_sequence = []
        if total_frames is None:
            total_frames = audio_features.shape[1]
        
        l = self.ladder_step
        N = self.window_size
        num_ladders = N // l
        
        pre_frames = condition_dict['y']['seed'].shape[1]
        self.pre_frames = pre_frames
        
        for i in range(pre_frames):
            generated_sequence.append(seed[:, i, :])
        
        frame_idx = pre_frames
        
        t_0_l = num_ladders
        
        while frame_idx < total_frames:
            frames_to_generate = min(l, total_frames - frame_idx)
            
            dt = 1.0 / num_ladders
            
            t_values = self._get_ladder_t_values_v2(batch_size, N, l, num_ladders, t_0_l, device)
            
            full_input = self.window_buffer
            
            nframes = full_input.shape[1]
            full_input_4d = full_input.view(batch_size, nframes, self.num_joints, self.input_dim)
            full_input_4d = full_input_4d.permute(0, 2, 3, 1)
            
            listener_window = None
            if self.use_listener_feedback:
                listener_window = listener_loader.get_history(frame_idx)
            
            if guidance_scale > 1.0:
                v_pred = self._guided_denoise_rdla(
                    full_input_4d, t_values, condition_dict, 
                    audio_features, frame_idx, frames_to_generate, guidance_scale,
                    listener_window
                )
            else:
                at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
                
                timesteps = t_values[:, 0] if t_values.dim() == 2 else t_values
                v_pred = self.denoiser(
                    x=full_input_4d,
                    timesteps=timesteps,
                    seed=condition_dict['y']['seed'],
                    at_feat=at_feat,
                    listener_latent=listener_window
                )
            
            v_pred = v_pred.squeeze(2)
            v_pred_3d = v_pred.permute(0, 2, 1)
            
            window_frames = self.window_buffer.shape[1]
            self.window_buffer = self.window_buffer + dt * v_pred_3d[:, -window_frames:, :]
            
            t_0_l -= 1
            
            output_frames = self.window_buffer[:, :frames_to_generate, :]
            
            if frames_to_generate > 1 and self.ofs_threshold < 1.0:
                output_frames = self.apply_ofs_smoothing(output_frames)
            
            for i in range(frames_to_generate):
                generated_sequence.append(output_frames[:, i, :])
            
            self.context_buffer = torch.cat([
                self.context_buffer[:, frames_to_generate:, :],
                output_frames
            ], dim=1)
            
            if self.context_buffer.shape[1] > pre_frames:
                self.context_buffer = self.context_buffer[:, -pre_frames:, :]
            
            if t_0_l == 0:
                self.window_buffer = torch.randn(
                    batch_size, self.window_size, self.input_dim * self.num_joints, device=device
                )
                t_0_l = num_ladders
            else:
                new_noise_frames = torch.randn(
                    batch_size, frames_to_generate, self.input_dim * self.num_joints, device=device
                )
                self.window_buffer = torch.cat([
                    self.window_buffer[:, frames_to_generate:, :],
                    new_noise_frames
                ], dim=1)
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    def _guided_denoise(self, x, t, condition_dict, audio_features, frame_idx, frames_to_generate, guidance_scale, listener_window=None):
        batch_size = x.shape[0]
        device = x.device
        
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([condition_dict['y']['seed']] * 2, dim=0)
        
        at_feat = self.get_audio_window(audio_features, frame_idx, batch_size, device)
        
        target_seq_len = at_feat.shape[1]
        target_feat_dim = at_feat.shape[2]
        null_cond_embed = self.denoiser.get_null_cond_embed(target_seq_len, target_feat_dim, at_feat.device)
        
        at_feat_uncond = null_cond_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        listener_doubled = None
        if listener_window is not None:
            listener_doubled = torch.cat([listener_window, listener_window], dim=0)
        
        timesteps_doubled = torch.full((batch_size * 2,), t, device=x.device)
        
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
            listener_latent=listener_doubled
        )
        
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output
