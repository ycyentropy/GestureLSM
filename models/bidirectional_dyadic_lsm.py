import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.LSM import GestureLSM, sample_beta_distribution
from models.config import instantiate_from_config
from loguru import logger

def reshape_coefs(t):
    """Reshape coefficients for broadcasting with 3D latent [B, T, D]."""
    return t.reshape((t.shape[0], 1, 1))

class BidirectionalDyadicLSM(GestureLSM):
    """
    经过优化的双向生成的Dyadic GestureLSM模型
    支持高效并行的训练和真正的双人联合生成
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        logger.info("Initializing Optimized BidirectionalDyadicLSM")
        
        # 注意：seed 直接传入 denoiser，不需要投影层
        # denoiser 内部通过 embed_text 层处理 seed
        
        # [修改] 去掉输出头，与单人模型保持一致
        # Denoiser 直接输出 latent，不经过额外的输出头
        
        self.window_size = cfg.model.get("window_size", 18)
        self.horizon = cfg.model.get("horizon", 144)
        self.use_id = cfg.model.get("use_id", True)
        self.main_id_weight = cfg.model.get("main_id_weight", 0.6)
        self.int_id_weight = cfg.model.get("int_id_weight", 0.4)

    def encode_conditions(self, main_data, interlocutor_data):
        # ... (保持原有的 encode_conditions 逻辑不变) ...
        # 为了节省篇幅，这里假设直接复用原代码的逻辑
        # 请务必保留原文件中的 encode_conditions 函数实现
        return super().encode_conditions(main_data, interlocutor_data) if hasattr(super(), 'encode_conditions') else self._original_encode_conditions(main_data, interlocutor_data)

    # 复制你原代码中的 encode_conditions 以确保完整性
    def _original_encode_conditions(self, main_data, interlocutor_data):
        """
        编码条件特征
        注意：Speaker 和 Listener 都使用 Speaker 的音频和文本作为条件
        """
        # 编码音频特征（main_data 包含 audio 和 word）
        audio = main_data['audio']
        word = main_data.get('word')
        at_feat = self.modality_encoder(audio, word)  # [B, h, audio_dim]
        
        # 处理ID嵌入（如果启用）
        if self.use_id:
            main_id = main_data['id']
            if main_id.dim() > 1: 
                main_id = main_id[:, 0]
            main_id_emb = self.id_embeddings(main_id.long().clamp(0, self.max_id - 1))
            
            int_id = interlocutor_data['id']
            if int_id.dim() > 1: 
                int_id = int_id[:, 0]
            int_id_emb = self.id_embeddings(int_id.long().clamp(0, self.max_id - 1))
            
            style_emb = (self.main_id_weight * main_id_emb + self.int_id_weight * int_id_emb)
        else:
            style_emb = None
        
        # 返回音频特征和style嵌入
        # 注意：seed 不在这里处理，而是直接传入 denoiser
        return at_feat, style_emb

    def compute_flow_loss(self, model_output, target, t, loss_type='mse'):
        # ... (保持原代码不变) ...
        # 建议保留原有的 NaN 检查和 Loss 计算逻辑
        t_clamped = torch.clamp(t, min=1e-6)
        if loss_type == 'huber':
            loss = F.huber_loss(model_output, target, delta=1.0, reduction='none')
        else:
            loss = F.mse_loss(model_output, target, reduction='none')
        
        # 确保维度匹配后再 mean
        # loss shape: [B, C, 1, T] -> mean over C and 1 -> [B, T]
        loss = loss.mean(dim=[1, 2]) # Mean over C and spatial dim
        # t_clamped shape: [B] -> reshape to [B, 1] for broadcasting
        t_clamped = t_clamped.unsqueeze(1)  # [B, 1]
        flow_loss = (loss / t_clamped).mean()
        return torch.clamp(flow_loss, max=1e6)

    def train_forward(self, speaker_data, listener_data,
                      speaker_latent, listener_latent, train_consistency=True):
        """
        [修改] 优化的训练前向传播，合并Batch
        """
        losses = {}
        batch_size = speaker_latent.shape[0]
        device = speaker_latent.device

        # 准备条件（Speaker 和 Listener 使用相同的 Speaker 音频）
        condition_data = {'audio': speaker_data['audio'], 'word': speaker_data['word']}
        interlocutor_for_speaker = {'seed': listener_data['seed'], 'id': listener_data['id']}
        interlocutor_for_listener = {'seed': speaker_data['seed'], 'id': speaker_data['id']}

        # 编码条件（只编码一次，两者共享）
        at_feat, _ = self.encode_conditions(condition_data, interlocutor_for_speaker)
        # Speaker 和 Listener 使用相同的 at_feat
        at_feat_sp = at_feat
        at_feat_ls = at_feat

        # 3. 构造合并的 Batch (Stacking)
        # 目标: [Speaker, Listener]
        # 注意：latent 是 [B, T, 384]，需要 reshape 为 [B, 3, 128, T] 传入 Denoiser
        targets = torch.cat([speaker_latent, listener_latent], dim=0) # [2B, T, 384]

        # 构造 Noisy Input
        x0_noise = torch.randn_like(targets)
        t = sample_beta_distribution(2 * batch_size, alpha=2, beta=1.2).to(device)
        t_coef = reshape_coefs(t)
        x_t = t_coef * targets + (1 - t_coef) * x0_noise
        # [2B, T, 384] -> [2B, 3, 128, T]
        x_t_denoiser = x_t.reshape(2 * batch_size, -1, 3, 128).permute(0, 2, 3, 1)

        # 构造 Other Condition (Ground Truth)
        # 对于 Speaker，Other 是 Listener GT；对于 Listener，Other 是 Speaker GT
        # 所以 x_other 的顺序应该是 [Listener_GT, Speaker_GT]
        x_other_targets = torch.cat([listener_latent, speaker_latent], dim=0)
        x_other_denoiser = None
        if self.denoiser.use_bidirectional_attn:
            # [2B, T, 384] -> [2B, 3, 128, T]
            x_other_denoiser = x_other_targets.reshape(2 * batch_size, -1, 3, 128).permute(0, 2, 3, 1)

        # 构造其他合并条件
        at_feats = torch.cat([at_feat_sp, at_feat_ls], dim=0)
        seeds = torch.cat([speaker_data['seed'], listener_data['seed']], dim=0)
        
        # 4. 单次调用 Denoiser (GPU 并行效率最高)
        # Denoiser 内部如果应用了 Batch Trick，这里会非常快
        output_combined = self.denoiser(
            x=x_t_denoiser,
            timesteps=t,
            seed=seeds,
            at_feat=at_feats,
            x_other=x_other_denoiser,
        )

        # 5. 直接使用 Denoiser 输出（去掉输出头，与单人模型保持一致）
        # Denoiser 输出 [2B, 384, 1, T]，直接拆分为 Speaker 和 Listener
        speaker_pred, listener_pred = torch.chunk(output_combined, 2, dim=0)  # [B, 384, 1, T]

        # 6. 计算 Flow Loss
        t_sp, t_ls = torch.chunk(t, 2, dim=0)
        x0_noise_sp, x0_noise_ls = torch.chunk(x0_noise, 2, dim=0)
        
        # 构造 target：与 model_output 格式一致 [B, 384, 1, T]
        # speaker_latent: [B, T, 384] -> [B, 384, 1, T]
        target_sp = speaker_latent.permute(0, 2, 1).unsqueeze(2)  # [B, 384, 1, T]
        target_ls = listener_latent.permute(0, 2, 1).unsqueeze(2)  # [B, 384, 1, T]
        
        if self.flow_mode == "v":
            target_v_sp = target_sp - x0_noise_sp.permute(0, 2, 1).unsqueeze(2)  # [B, 384, 1, T]
            target_v_ls = target_ls - x0_noise_ls.permute(0, 2, 1).unsqueeze(2)  # [B, 384, 1, T]
        else:
            target_v_sp = target_sp  # [B, 384, 1, T]
            target_v_ls = target_ls  # [B, 384, 1, T]

        # 获取损失权重（alpha和1-alpha）
        alpha = self.cfg.model.get('loss_alpha', 0.5)
        speaker_weight = alpha
        listener_weight = 1 - alpha

        loss_type = self.cfg.model.get('loss_type', 'mse')
        losses['speaker_flow_loss'] = self.compute_flow_loss(speaker_pred, target_v_sp, t_sp, loss_type) * speaker_weight
        losses['listener_flow_loss'] = self.compute_flow_loss(listener_pred, target_v_ls, t_ls, loss_type) * listener_weight

        # 7. Consistency Loss (保留部分采样逻辑，同样可以考虑合并)
        if train_consistency:
            # 为了代码简洁，这里暂时分别计算，如果需要极致优化也可以合并
            losses['speaker_cons_loss'] = self._compute_consistency_loss(
                speaker_latent, speaker_data['seed'], at_feat_sp, x0_noise_sp
            ) * speaker_weight
            losses['listener_cons_loss'] = self._compute_consistency_loss(
                listener_latent, listener_data['seed'], at_feat_ls, x0_noise_ls
            ) * listener_weight
        else:
            losses['speaker_cons_loss'] = torch.tensor(0.0, device=device)
            losses['listener_cons_loss'] = torch.tensor(0.0, device=device)

        # 8. 汇总 Loss
        total_loss = sum(losses.values())
        losses['loss'] = total_loss
        return losses

    def _compute_consistency_loss(self, latent, seed, at_feat, x0_noise):
        # ... (保持你原代码中的修复版逻辑) ...
        # 注意：这里不做修改，直接使用你提供的修正后的逻辑
        return super()._compute_consistency_loss(latent, seed, at_feat, x0_noise) if hasattr(super(), '_compute_consistency_loss') else self._original_consistency_loss(latent, seed, at_feat, x0_noise)

    # 这里的 _original_consistency_loss 对应你 prompt 中提供的 _compute_consistency_loss 实现
    def _original_consistency_loss(self, latent, seed, at_feat, x0_noise):
        """
        完整的consistency损失实现，与原框架保持一致
        """
        batch_size = latent.shape[0]
        device = latent.device

        # 1. 采样delta和t
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(device)
        delta_probs = torch.ones((deltas.shape[0],)).to(device) / deltas.shape[0]

        # 2. 分割batch：75% flow, 25% consistency
        flow_batch_size = int(batch_size * 3 / 4)
        cons_batch_size = batch_size - flow_batch_size

        # 3. 采样t和d
        t = sample_beta_distribution(batch_size, alpha=2, beta=1.2).to(device)
        d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
        d[:flow_batch_size] = 0  # flow部分d=0

        # 4. 计算x_t
        t_coef = reshape_coefs(t)
        x_t = t_coef * latent + (1 - t_coef) * x0_noise
        t = t_coef.flatten()

        # 将x_t reshape为denoiser期望的格式 [B, T, 384] -> [B, 3, 128, T]
        x_t_denoiser = x_t.reshape(batch_size, -1, 3, 128).permute(0, 2, 3, 1)

        # 5. CFG处理（80% true, 20% false）
        import numpy as np
        force_cfg = np.random.choice(
            [True, False], size=cons_batch_size, p=[0.8, 0.2]
        )

        # Apply force_cfg
        at_feat_cfg = at_feat.clone()
        # 对于consistency部分应用CFG
        for i in range(cons_batch_size):
            if force_cfg[i]:
                at_feat_cfg[flow_batch_size + i] = 0

        # 6. 第一步：计算v_t（无梯度）
        with torch.no_grad():
            pred_t = self.denoiser(
                x=x_t_denoiser[flow_batch_size:],
                timesteps=t[flow_batch_size:],
                seed=seed[flow_batch_size:],
                at_feat=at_feat_cfg[flow_batch_size:],
                cond_time=d[flow_batch_size:],
            )

            d_coef = reshape_coefs(d)
            if self.flow_mode == "v":
                speed_t = pred_t
            else:
                # 需要将x0_noise也reshape为[B, 3, 128, T]
                x0_noise_denoiser = x0_noise.reshape(batch_size, -1, 3, 128).permute(0, 2, 3, 1)
                speed_t = pred_t - x0_noise_denoiser[flow_batch_size:]

            # 将speed_t reshape回[B, T, 384]用于计算x_td
            speed_t_reshaped = speed_t.permute(0, 3, 1, 2).reshape(cons_batch_size, -1, 384)
            x_td = x_t[flow_batch_size:] + d_coef[flow_batch_size:] * speed_t_reshaped
            # 将x_td reshape回[B, 3, 128, T]用于denoiser
            x_td_denoiser = x_td.reshape(cons_batch_size, -1, 3, 128).permute(0, 2, 3, 1)
            d = d_coef.flatten()

            # 7. 第二步：计算v_{t+d}（无梯度）
            pred_td = self.denoiser(
                x=x_td_denoiser,
                timesteps=t[flow_batch_size:] + d[flow_batch_size:],
                seed=seed[flow_batch_size:],
                at_feat=at_feat_cfg[flow_batch_size:],
                cond_time=d[flow_batch_size:],
            )

            if self.flow_mode == "v":
                speed_td = pred_td
            else:
                speed_td = pred_td - x0_noise_denoiser[flow_batch_size:]

            # 8. 计算目标速度
            speed_target = (speed_t + speed_td) / 2

        # 9. 第三步：计算v_t（有梯度，使用2d作为cond_time）
        model_pred = self.denoiser(
            x=x_t_denoiser[flow_batch_size:],
            timesteps=t[flow_batch_size:],
            seed=seed[flow_batch_size:],
            at_feat=at_feat_cfg[flow_batch_size:],
            cond_time=2 * d[flow_batch_size:],
        )

        if self.flow_mode == "v":
            speed_pred = model_pred
        else:
            speed_pred = model_pred - x0_noise_denoiser[flow_batch_size:]

        # 10. 计算MSE损失
        consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")

        return consistency_loss

    @torch.no_grad()
    def generate(self, speaker_data, listener_data, 
                 speaker_gt=None, listener_gt=None,
                 mode='joint', # [修改] 新增 mode: 'joint' or 'conditional'
                 num_steps=2, guidance_scale=2.0):
        """
        [修改] 支持联合生成的推理函数
        mode='joint': 从零开始同时生成两个角色 (不需要 GT)
        mode='conditional': 已知一方 GT，生成另一方 (原逻辑)
        """
        self.eval()
        device = speaker_data['audio'].device
        batch_size = speaker_data['audio'].shape[0]

        # 准备条件特征（Speaker 和 Listener 使用相同的 Speaker 音频）
        condition_data = {'audio': speaker_data['audio'], 'word': speaker_data['word']}
        int_for_sp = {'seed': listener_data['seed'], 'id': listener_data['id']}
        int_for_ls = {'seed': speaker_data['seed'], 'id': speaker_data['id']}

        # 编码条件（只编码一次）
        at_feat, _ = self.encode_conditions(condition_data, int_for_sp)
        at_feat_sp = at_feat
        at_feat_ls = at_feat

        # 2. 初始化噪声
        shape = (batch_size, self.input_dim * self.num_joints, 1, self.seq_len)
        x_sp = torch.randn(shape, device=device)
        x_ls = torch.randn(shape, device=device)
        
        timesteps = torch.linspace(1e-8, 1.0 - 1e-8, num_steps + 1).to(device)

        # 3. 推理循环
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            t_tensor = t_curr.unsqueeze(0).expand(batch_size)
            dt = t_next - t_curr

            # 准备 x_other
            if mode == 'conditional':
                # 原有逻辑：必须提供 GT
                x_other_sp_in = listener_gt.reshape(batch_size, -1, 3, 128).permute(0, 2, 3, 1) if listener_gt is not None else None
                x_other_ls_in = speaker_gt.reshape(batch_size, -1, 3, 128).permute(0, 2, 3, 1) if speaker_gt is not None else None
            else: 
                # Joint 模式：使用当前估计值作为对方的输入
                # 注意：这里使用的是 Noisy 状态 x_t 作为近似。
                # 更高级的方法是使用上一轮预测的 x_pred (Clean Estimate)
                x_other_sp_in = x_ls # Speaker 看到的 Other 是 Listener
                x_other_ls_in = x_sp # Listener 看到的 Other 是 Speaker

            # 同样合并 Batch 进行推理
            x_in = torch.cat([x_sp, x_ls], dim=0)
            t_in = torch.cat([t_tensor, t_tensor], dim=0)
            seeds_in = torch.cat([speaker_data['seed'], listener_data['seed']], dim=0)
            at_feats_in = torch.cat([at_feat_sp, at_feat_ls], dim=0)
            
            if self.denoiser.use_bidirectional_attn:
                x_other_in = torch.cat([x_other_sp_in, x_other_ls_in], dim=0)
            else:
                x_other_in = None

            # 预测
            output = self.denoiser(x_in, t_in, seed=seeds_in, at_feat=at_feats_in, x_other=x_other_in)
            
            # 直接使用 Denoiser 输出（去掉输出头，与单人模型保持一致）
            # Denoiser 输出 [2B, C, 1, T]，直接拆分为 Speaker 和 Listener
            pred_sp, pred_ls = torch.chunk(output, 2, dim=0)

            # Euler Step (Flow Matching)
            # x_{t+1} = x_t + dt * v_pred
            x_sp = x_sp + dt * pred_sp
            x_ls = x_ls + dt * pred_ls
            
        return x_sp, x_ls