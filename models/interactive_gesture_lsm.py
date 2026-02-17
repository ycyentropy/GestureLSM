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
        if use_ladder and self.ladder_step > 1:
            return self._apply_ladder_noise(x0, batch_size, device)
        else:
            return self._apply_linear_noise(x0, batch_size, device)
    
    def _apply_linear_noise(self, x0, batch_size, device):
        t_0 = torch.rand(batch_size, device=device) * 0.09 + 0.01
        
        t_values = []
        for n in range(self.window_size):
            t_n = t_0 + n * (1.0 - t_0) / (self.window_size - 1)
            t_values.append(t_n)
        t_values = torch.stack(t_values, dim=1).to(device)
        
        noise = torch.randn_like(x0)
        t_coef = t_values.unsqueeze(-1)
        x_t = t_coef * x0 + (1 - t_coef) * noise
        return x_t, t_values, noise
    
    def _apply_ladder_noise(self, x0, batch_size, device):
        l = self.ladder_step
        N = self.window_size
        num_ladders = N // l
        
        t_0_l = torch.randint(1, l + 1, (batch_size,), device=device).float()
        
        t_values = []
        for n in range(N):
            k = n // l
            t_n = t_0_l + k * l
            t_values.append(t_n)
        t_values = torch.stack(t_values, dim=1).to(device)
        
        max_t = t_values[:, -1:]
        t_values = t_values / max_t.clamp(min=1.0)
        
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
    
    def apply_context_noise(self, context_latents):
        if not self.training:
            return context_latents
        
        batch_size = context_latents.shape[0]
        noise_mask = torch.rand(batch_size, device=context_latents.device) < self.context_noise_prob
        
        if noise_mask.any():
            noise_scale = torch.rand(batch_size, device=context_latents.device) * self.max_context_noise
            noise_scale = noise_scale.unsqueeze(1).unsqueeze(2)
            noise = torch.randn_like(context_latents) * noise_scale
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
    
    def compute_inertia_loss(self, pred, target):
        if self.inertia_weight <= 0:
            return torch.tensor(0.0, device=pred.device)
        
        error = pred - target
        
        error_n = error[:, :-1, :]
        error_n_plus_1 = error[:, 1:, :]
        
        inner_product = (error_n * error_n_plus_1).sum(dim=-1)
        
        inertia_loss = self.inertia_weight * inner_product.abs().mean()
        
        return inertia_loss
    
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
        
        window_starts = list(range(pre_frames, seq_len - self.window_size + 1, self.ladder_step))
        if len(window_starts) == 0:
            window_starts = [pre_frames]
        
        total_flow_loss = 0.0
        total_inertia_loss = 0.0
        total_consistency_loss = 0.0
        num_windows = len(window_starts)
        
        for window_start in window_starts:
            window_end = window_start + self.window_size
            
            if window_end > seq_len:
                window_end = seq_len
                window_start = max(pre_frames, window_end - self.window_size)
            
            window_latents = latents[:, window_start:window_end, :]
            
            context_latents = latents[:, max(0, window_start - self.context_size):window_start, :]
            if context_latents.shape[1] < self.context_size:
                padding = torch.zeros(batch_size, self.context_size - context_latents.shape[1], latent_dim, device=device)
                context_latents = torch.cat([padding, context_latents], dim=1)
            context_latents = self.apply_context_noise(context_latents)
            
            audio_features_flow = self.apply_conditional_dropout(audio_features[:flow_batch_size])
            
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
            
            x_t, t_values, noise = self.apply_rolling_noise(
                window_latents, batch_size, device, use_ladder=True
            )
            
            full_input = torch.cat([context_latents, x_t], dim=1)
            
            full_input_4d = full_input.reshape(batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
            model_output = self.denoiser(
                x=full_input_4d[:flow_batch_size],
                timesteps=t_values[:flow_batch_size, 0],
                seed=condition_dict['y']['seed'][:flow_batch_size],
                at_feat=audio_features_flow,
                listener_latent=listener_window[:flow_batch_size] if listener_window is not None else None
            )
            model_output = model_output.permute(0, 3, 1, 2).reshape(flow_batch_size, -1, self.num_joints * self.input_dim)
            
            if self.flow_mode == "v":
                target = window_latents[:flow_batch_size] - noise[:flow_batch_size]
            else:
                target = window_latents[:flow_batch_size]
            
            window_output = model_output[:, self.context_size:, :]
            
            flow_loss = F.mse_loss(window_output, target)
            inertia_loss = self.compute_inertia_loss(window_output, target)
            
            total_flow_loss += flow_loss
            total_inertia_loss += inertia_loss
            
            if flow_batch_size < batch_size:
                consistency_loss = self._compute_consistency_loss(
                    condition_dict, latents, audio_features, batch_size, flow_batch_size,
                    device, window_start, window_end, listener_latents
                )
                total_consistency_loss += consistency_loss
        
        losses["flow_loss"] = total_flow_loss / num_windows
        losses["inertia_loss"] = total_inertia_loss / num_windows
        losses["consistency_loss"] = total_consistency_loss / num_windows
        
        total_loss = losses["flow_loss"] + losses["inertia_loss"] + losses["consistency_loss"]
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
        audio_features_cons = self._apply_force_cfg(audio_features[cons_start:], force_cfg)
        
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
        
        full_input = torch.cat([context_latents, x_t], dim=1)
        full_input_4d = full_input.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        
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
            pred_t_window = pred_t[:, self.context_size:, :]
            
            d_coef = reshape_coefs(d)
            if self.flow_mode == "v":
                speed_t = pred_t_window
            else:
                speed_t = pred_t_window - x0_noise
            
            x_td = x_t + d_coef * speed_t
            
            full_input_td = torch.cat([context_latents, x_td], dim=1)
            full_input_td_4d = full_input_td.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
            
            pred_td = self.denoiser(
                x=full_input_td_4d,
                timesteps=t + d,
                seed=seed,
                at_feat=audio_features_cons,
                cond_time=d,
                listener_latent=listener_window
            )
            pred_td = pred_td.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, self.num_joints * self.input_dim)
            pred_td_window = pred_td[:, self.context_size:, :]
            
            if self.flow_mode == "v":
                speed_td = pred_td_window
            else:
                speed_td = pred_td_window - x0_noise
            
            speed_target = (speed_t + speed_td) / 2
        
        full_input_4d = full_input.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        model_pred = self.denoiser(
            x=full_input_4d,
            timesteps=t,
            seed=seed,
            at_feat=audio_features_cons,
            cond_time=2 * d,
            listener_latent=listener_window
        )
        model_pred = model_pred.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, self.num_joints * self.input_dim)
        model_pred_window = model_pred[:, self.context_size:, :]
        
        if self.flow_mode == "v":
            speed_pred = model_pred_window
        else:
            speed_pred = model_pred_window - x0_noise
        
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
    
    @torch.no_grad()
    def generate(self, condition_dict, audio_features, listener_latents=None,
                 num_steps=2, guidance_scale=2.0):
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
                
                full_input = torch.cat([self.context_buffer, self.window_buffer], dim=1)
                
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
                    at_feat = audio_features[:, frame_idx:frame_idx+frames_to_generate, :]
                    if at_feat.shape[1] == 0:
                        at_feat = audio_features[:, frame_idx:frame_idx+1, :]
                    
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
            
            self.window_buffer = torch.randn(
                batch_size, self.window_size, self.input_dim * self.num_joints, device=device
            )
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    @torch.no_grad()
    def generate_online(self, condition_dict, audio_features, listener_loader,
                        num_steps=2, guidance_scale=2.0):
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
                
                full_input = torch.cat([self.context_buffer, self.window_buffer], dim=1)
                
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
                    at_feat = audio_features[:, frame_idx:frame_idx+frames_to_generate, :]
                    if at_feat.shape[1] == 0:
                        at_feat = audio_features[:, frame_idx:frame_idx+1, :]
                    
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
            
            self.window_buffer = torch.randn(
                batch_size, self.window_size, self.input_dim * self.num_joints, device=device
            )
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    def _guided_denoise(self, x, t, condition_dict, audio_features, frame_idx, frames_to_generate, guidance_scale, listener_window=None):
        batch_size = x.shape[0]
        
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([condition_dict['y']['seed']] * 2, dim=0)
        
        at_feat = audio_features[:, frame_idx:frame_idx+frames_to_generate, :]
        if at_feat.shape[1] == 0:
            at_feat = audio_features[:, frame_idx:frame_idx+1, :]
        
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
