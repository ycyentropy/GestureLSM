import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.LSM import GestureLSM, sample_beta_distribution
from loguru import logger

def reshape_coefs(t):
    """Reshape coefficients for broadcasting with 3D latent [B, T, D]."""
    return t.reshape((t.shape[0], 1, 1))

class RollingGestureLSM(GestureLSM):
    """流式GestureLSM模型，结合流匹配、滚动窗口机制和RDLA加速"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        logger.info("Initializing RollingGestureLSM with RDLA")
        
        # 滚动窗口参数
        self.window_size = cfg.model.get("window_size", 18)
        self.context_size = cfg.model.get("context_size", 4)
        self.noise_range = cfg.model.get("noise_range", [0, 1])
        
        # RDLA 参数
        self.ladder_step = cfg.model.get("ladder_step", 1)
        self.inertia_weight = cfg.model.get("inertia_weight", 0.1)
        self.ofs_threshold = cfg.model.get("ofs_threshold", 0.8)
        
        # Consistency Loss 参数
        self.consistency_ratio = cfg.model.get("consistency_ratio", 0.25)
        
        # CFG 参数
        self.cond_drop_prob = cfg.model.get("cond_drop_prob", 0.1)
        
        # 窗口和上下文缓冲区
        self.window_buffer = None
        self.context_buffer = None
        
        # 验证阶梯步长与窗口大小的兼容性
        if self.window_size % self.ladder_step != 0:
            logger.warning(f"window_size ({self.window_size}) should be divisible by ladder_step ({self.ladder_step})")
    
    def init_rolling_window(self, batch_size, device, idle_pose=None):
        """初始化滚动窗口"""
        self.window_buffer = torch.randn(
            batch_size, self.window_size, self.input_dim * self.num_joints,
            device=device
        )
        
        if idle_pose is not None:
            self.context_buffer = idle_pose.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            self.context_buffer = torch.randn(
                batch_size, self.context_size, self.input_dim * self.num_joints,
                device=device
            ) * 0.001
    
    def apply_rolling_noise(self, x0, batch_size, device, use_ladder=True):
        """应用滚动噪声调度到窗口，支持阶梯式噪声（RDLA）"""
        if use_ladder and self.ladder_step > 1:
            return self._apply_ladder_noise(x0, batch_size, device)
        else:
            return self._apply_linear_noise(x0, batch_size, device)
    
    def _apply_linear_noise(self, x0, batch_size, device):
        """线性噪声调度（标准滚动扩散）"""
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
        """阶梯式噪声调度（RDLA）"""
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
        """应用条件dropout用于CFG训练"""
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        
        batch_size = at_feat.shape[0]
        keep_mask = torch.rand(batch_size, device=at_feat.device) > cond_drop_prob
        
        null_cond_embed = self.denoiser.get_null_cond_embed(at_feat.shape[-1], at_feat.device)
        
        at_feat_dropped = at_feat.clone()
        at_feat_dropped[~keep_mask] = null_cond_embed.unsqueeze(0).expand((~keep_mask).sum(), -1, -1)
        
        return at_feat_dropped
    
    def compute_inertia_loss(self, pred, target):
        """计算惯性损失，惩罚相邻帧的突变"""
        if self.inertia_weight <= 0:
            return torch.tensor(0.0, device=pred.device)
        
        error = pred - target
        
        error_n = error[:, :-1, :]
        error_n_plus_1 = error[:, 1:, :]
        
        inner_product = (error_n * error_n_plus_1).sum(dim=-1)
        
        inertia_loss = self.inertia_weight * inner_product.abs().mean()
        
        return inertia_loss
    
    def roll_window(self, new_frame, step=1):
        """窗口平移：丢弃首帧，添加新帧
        
        Args:
            new_frame: 如果 step=1, shape=[B, D]; 如果 step>1, shape=[B, step, D]
            step: 平移步数
        """
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
        """OFS即时平滑策略"""
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
    
    def train_forward(self, condition_dict, latents):
        """滚动窗口训练前向传播，融合Consistency Loss和惯性损失"""
        batch_size, seq_len, latent_dim = latents.shape
        device = latents.device
        
        losses = {}
        
        flow_batch_size = int(batch_size * (1 - self.consistency_ratio))
        
        max_start = seq_len - self.window_size - self.context_size
        j = torch.randint(0, max_start, (1,)).item()
        
        window_start = j + self.context_size
        window_end = window_start + self.window_size
        window_latents = latents[:, window_start:window_end, :]
        
        context_latents = latents[:, j:j+self.context_size, :]
        
        audio_features = self.modality_encoder(
            condition_dict['y']['audio_onset'],
            condition_dict['y']['word']
        )
        
        if self.use_id:
            id_tensor = condition_dict['y']['id'][:, 0]
            id_emb = self.id_embeddings(id_tensor)
            id_emb_expanded = id_emb.unsqueeze(1).repeat(1, audio_features.shape[1], 1)
            audio_features = torch.cat([audio_features, id_emb_expanded], dim=-1)
        
        audio_features_flow = self.apply_conditional_dropout(audio_features[:flow_batch_size])
        
        x_t, t_values, noise = self.apply_rolling_noise(
            window_latents, batch_size, device, use_ladder=True
        )
        
        full_input = torch.cat([context_latents, x_t], dim=1)
        
        full_input_4d = full_input.reshape(batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        model_output = self.denoiser(
            x=full_input_4d[:flow_batch_size],
            timesteps=t_values[:flow_batch_size, 0],
            seed=condition_dict['y']['seed'][:flow_batch_size],
            at_feat=audio_features_flow
        )
        model_output = model_output.permute(0, 3, 1, 2).reshape(flow_batch_size, -1, self.num_joints * self.input_dim)
        
        if self.flow_mode == "v":
            target = window_latents[:flow_batch_size] - noise[:flow_batch_size]
        else:
            target = window_latents[:flow_batch_size]
        
        window_output = model_output[:, self.context_size:, :]
        
        flow_loss = F.mse_loss(window_output, target)
        
        inertia_loss = self.compute_inertia_loss(window_output, target)
        
        losses["flow_loss"] = flow_loss
        losses["inertia_loss"] = inertia_loss
        
        if flow_batch_size < batch_size:
            consistency_loss = self._compute_consistency_loss(
                condition_dict, latents, audio_features, batch_size, flow_batch_size,
                device, j, window_start, window_end
            )
            losses["consistency_loss"] = consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=device)
        
        total_loss = flow_loss + inertia_loss + consistency_loss
        losses["loss"] = total_loss
        
        return losses
    
    def _compute_consistency_loss(self, condition_dict, latents, audio_features, 
                                   batch_size, flow_batch_size, device, j, 
                                   window_start, window_end):
        """计算Consistency Loss"""
        cons_start = flow_batch_size
        cons_batch_size = batch_size - flow_batch_size
        
        if cons_batch_size <= 0:
            return torch.tensor(0.0, device=device)
        
        window_latents = latents[cons_start:, window_start:window_end, :]
        context_latents = latents[cons_start:, j:j+self.context_size, :]
        
        seed_vectors = condition_dict['y']['seed'][cons_start:]
        
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(device)
        delta_probs = torch.ones((deltas.shape[0],), device=device) / deltas.shape[0]
        
        t = sample_beta_distribution(cons_batch_size, alpha=2, beta=1.2).to(device)
        d = deltas[delta_probs.multinomial(cons_batch_size, replacement=True)]
        
        x0_noise = torch.randn_like(window_latents)
        t_coef = reshape_coefs(t)
        x_t = t_coef * window_latents + (1 - t_coef) * x0_noise
        
        force_cfg = np.random.choice([True, False], size=cons_batch_size, p=[0.8, 0.2])
        audio_features_cons = self._apply_force_cfg(audio_features[cons_start:], force_cfg)
        
        full_input = torch.cat([context_latents, x_t], dim=1)
        full_input_4d = full_input.reshape(cons_batch_size, -1, self.num_joints, self.input_dim).permute(0, 2, 3, 1)
        
        with torch.no_grad():
            pred_t = self.denoiser(
                x=full_input_4d,
                timesteps=t,
                seed=seed_vectors,
                at_feat=audio_features_cons,
                cond_time=d
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
                seed=seed_vectors,
                at_feat=audio_features_cons,
                cond_time=d
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
            seed=seed_vectors,
            at_feat=audio_features_cons,
            cond_time=2 * d
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
        """应用强制CFG"""
        batch_size = at_feat.shape[0]
        null_cond_embed = self.denoiser.get_null_cond_embed(at_feat.shape[-1], at_feat.device)
        
        at_feat_forced = at_feat.clone()
        force_cfg_tensor = torch.tensor(force_cfg, device=at_feat.device)
        if force_cfg_tensor.sum() > 0:
            at_feat_forced[force_cfg_tensor] = null_cond_embed.unsqueeze(0).expand(force_cfg_tensor.sum(), -1, -1)
        
        return at_feat_forced
    
    @torch.no_grad()
    def generate(self, condition_dict, audio_features, 
                 num_steps=2, guidance_scale=2.0):
        """流式生成，支持阶梯式加速和OFS平滑"""
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        self.init_rolling_window(batch_size, device)
        
        epsilon = 1e-8
        delta_t = 1.0 / num_steps
        timesteps = torch.linspace(epsilon, 1 - epsilon, num_steps + 1, device=device)
        
        generated_sequence = []
        total_frames = audio_features.shape[1]
        
        l = self.ladder_step
        
        frame_idx = 0
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
                
                if guidance_scale > 1.0:
                    v_pred = self._guided_denoise(
                        full_input_4d, t_curr, condition_dict, 
                        audio_features, frame_idx, frames_to_generate, guidance_scale
                    )
                else:
                    at_feat = audio_features[:, frame_idx:frame_idx+frames_to_generate, :]
                    if at_feat.shape[1] == 0:
                        at_feat = audio_features[:, frame_idx:frame_idx+1, :]
                    
                    v_pred = self.denoiser(
                        x=full_input_4d,
                        timesteps=torch.full((batch_size,), t_curr, device=device),
                        seed=condition_dict['y']['seed'],
                        at_feat=at_feat
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
            
            new_frames = torch.randn(batch_size, frames_to_generate, self.input_dim * self.num_joints, device=device)
            self.roll_window(new_frames, step=frames_to_generate)
            
            frame_idx += frames_to_generate
        
        generated_sequence = torch.stack(generated_sequence, dim=1)
        return generated_sequence
    
    def _guided_denoise(self, x, t, condition_dict, audio_features, frame_idx, frames_to_generate, guidance_scale):
        """带CFG的去噪"""
        batch_size = x.shape[0]
        
        x_doubled = torch.cat([x] * 2, dim=0)
        seed_doubled = torch.cat([condition_dict['y']['seed']] * 2, dim=0)
        
        at_feat = audio_features[:, frame_idx:frame_idx+frames_to_generate, :]
        if at_feat.shape[1] == 0:
            at_feat = audio_features[:, frame_idx:frame_idx+1, :]
        
        null_cond_embed = self.denoiser.get_null_cond_embed(at_feat.shape[-1], at_feat.device)
        
        at_feat_uncond = null_cond_embed[:at_feat.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1)
        
        at_feat_combined = torch.cat([at_feat, at_feat_uncond], dim=0)
        
        timesteps_doubled = torch.full((batch_size * 2,), t, device=x.device)
        
        output = self.denoiser(
            x=x_doubled,
            timesteps=timesteps_doubled,
            seed=seed_doubled,
            at_feat=at_feat_combined,
        )
        
        pred_cond, pred_uncond = output.chunk(2, dim=0)
        guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        
        return guided_output
