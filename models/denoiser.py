import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# 请确保 .layers 路径下有这些模块
from .layers.utils import *
from .layers.transformer import SpatialTemporalBlock, CrossAttentionBlock, BidirectionalCrossAttention

class GestureDenoiser(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_cross_attn_layers=3, # [修改] 新增参数，不再硬编码
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        n_seed=8,
        flip_sin_to_cos=True,
        freq_shift=0,
        cond_proj_dim=None,
        use_exp=False,
        seq_len=32,
        embed_context_multiplier=4,
        use_id=False,
        id_embedding_dim=256,
        use_bidirectional_attn=False,
        bidirectional_layers=2,
        bidirectional_fusion="add",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_exp = use_exp
        self.joint_num = 3 if not self.use_exp else 4
        self.use_id = use_id
        self.id_embedding_dim = id_embedding_dim
        self.n_seed = n_seed
        self.seq_len = seq_len
        self.embed_context_multiplier = embed_context_multiplier
        self.use_bidirectional_attn = use_bidirectional_attn
        self.bidirectional_layers = bidirectional_layers
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Cross attention context dimension
        self.cross_attn_context_dim = self.input_dim * self.joint_num * self.embed_context_multiplier

        # [修改] 使用参数控制层数，而非硬编码 range(3)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=self.latent_dim * self.joint_num,
                num_heads=self.num_heads,
                mlp_ratio=self.ff_size // self.latent_dim,
                drop_path=self.dropout,
                context_dim=self.cross_attn_context_dim
            ) 
            for _ in range(num_cross_attn_layers)
        ])
        
        # [修改] 重命名为 temporal_blocks，语义更清晰
        self.temporal_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.ff_size // self.latent_dim,
                drop_path=self.dropout
            ) 
            for _ in range(self.num_layers)
        ])
        
        # 双向交叉注意力层（用于双人交互）
        if self.use_bidirectional_attn:
            self.bidirectional_attn_blocks = nn.ModuleList([
                BidirectionalCrossAttention(
                    dim=self.latent_dim * self.joint_num,
                    num_heads=self.num_heads,
                    mlp_ratio=self.ff_size // self.latent_dim,
                    drop_path=self.dropout,
                    fusion=bidirectional_fusion,
                )
                for _ in range(self.bidirectional_layers)
            ])
        else:
            self.bidirectional_attn_blocks = None
            
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # 调整 embed_text 输入维度以适应 seed 形状
        # seed_in shape: [bs_curr, n_seed, 384]，展平后为 [bs_curr, n_seed*384]
        # 但实际输入可能已经是展平的，所以先检查形状
        # 这里使用固定的384*4=1536作为输入维度
        self.embed_text = nn.Linear(384 * 4, self.latent_dim)

        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

        self.rel_pos = SinusoidalEmbeddings(self.latent_dim)
        self.input_process = InputProcess(self.input_dim , self.latent_dim)
        self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim, self.activation, cond_proj_dim=cond_proj_dim, zero_init_cond=True)
        time_dim = self.latent_dim
        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        if cond_proj_dim is not None:
            self.cond_proj = Timesteps(cond_proj_dim, flip_sin_to_cos, freq_shift)
        
        # Null condition embedding for classifier-free guidance
        self.null_cond_embed = nn.Parameter(torch.zeros(self.seq_len, self.latent_dim * self.joint_num), requires_grad=True)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def prob_mask_like(self, shape, prob, device):
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
    
    def get_null_cond_embed(self, target_dim, device):
        """
        获取 Null condition embedding。
        注意：在训练过程中修改 Parameter 形状是非常危险的，会导致优化器失效。
        此修改仅保留安全性检查。
        """
        current_dim = self.null_cond_embed.shape[1]
        
        if current_dim == target_dim:
            return self.null_cond_embed.to(device)
        
        if self.training:
            # 训练模式下警告，避免静默错误
            print(f"Warning: Requesting null_cond_embed resize from {current_dim} to {target_dim} during training. This may break the optimizer.")

        # 如果维度不匹配，创建一个临时的 Tensor (如果是推理阶段) 或者尝试 Parameter (如果在初始化阶段)
        # 这里为了保持原逻辑的兼容性，进行动态调整，但需谨慎使用
        if target_dim > current_dim:
            new_data = torch.cat([
                self.null_cond_embed.data.to(device),
                torch.randn(self.seq_len, target_dim - current_dim, device=device) * 0.01
            ], dim=1)
        else:
            new_data = self.null_cond_embed.data.to(device)[:, :target_dim]
            
        if not self.training:
             # 推理时不更新梯度，直接返回 Tensor 即可，不修改 self.null_cond_embed
            return new_data
        else:
            # 训练时必须重置 Parameter，但如前所述，这通常不可行。
            # 建议确保初始化时的 dim 就是最大所需 dim
            self.null_cond_embed = nn.Parameter(new_data, requires_grad=True)
            return self.null_cond_embed

    def forward(self, x, timesteps, cond_time=None, seed=None, at_feat=None, x_other=None):
        """
        x: [batch_size, njoints, nfeats, max_frames]
        x_other: [batch_size, njoints, nfeats, max_frames]
        """
        # [修改] 统一处理输入维度，处理最后的单维度
        if x.dim() == 4 and x.shape[-1] == 1: # just in case
            x = x.squeeze(-1)
        
        # 确保格式为 [bs, njoints, nfeats, nframes]
        if x.shape[2] == 1 and x.dim() > 3: 
             # 原代码逻辑，虽然奇怪但保留兼容性，建议检查数据源
            x = x.squeeze(2)
            x = x.reshape(x.shape[0], self.joint_num, -1, x.shape[2])

        # [修改] 决定是否进行 Batch 并行处理
        # 只有当启用了双向且提供了 x_other 时才进行并行
        do_interaction = (self.use_bidirectional_attn and 
                          x_other is not None and 
                          self.bidirectional_attn_blocks is not None)

        if do_interaction:
            # 处理 x_other 维度
            if x_other.shape[2] == 1 and x_other.dim() > 3:
                 x_other = x_other.squeeze(2)
                 x_other = x_other.reshape(x_other.shape[0], self.joint_num, -1, x_other.shape[2])
            
            # [关键优化] 在 Batch 维度拼接，一次性计算特征
            x_in = torch.cat([x, x_other], dim=0) # [2*B, J, C, T]
            
            # 扩展条件数据
            timesteps_in = torch.cat([timesteps, timesteps], dim=0)
            
            if seed is not None:
                seed_in = torch.cat([seed, seed], dim=0)
            else:
                seed_in = None
                
            if cond_time is not None:
                cond_time_in = torch.cat([cond_time, cond_time], dim=0)
            else:
                cond_time_in = None
            
            # 假设 at_feat (Condition) 是共享的或已对齐，这里简单复制
            if at_feat is not None:
                at_feat_in = at_feat.repeat(2, 1, 1)
            else:
                at_feat_in = None
        else:
            x_in = x
            timesteps_in = timesteps
            seed_in = seed
            cond_time_in = cond_time
            at_feat_in = at_feat

        # --- 开始共享主干网络计算 ---
        bs_curr, njoints, nfeats, nframes = x_in.shape
        
        # 1. Time Embedding
        time_emb = self.time_proj(timesteps_in)
        time_emb = time_emb.to(dtype=x_in.dtype)

        if cond_time_in is not None and self.cond_proj is not None:
            # cond_time 已在上面处理过维度
            cond_emb = self.cond_proj(cond_time_in)
            cond_emb = cond_emb.to(dtype=x_in.dtype)
            emb_t = self.time_embedding(time_emb, cond_emb)
        else:
            emb_t = self.time_embedding(time_emb)
        
        # 2. Seed Embedding
        if self.n_seed != 0 and seed_in is not None and self.embed_text is not None:
            # seed_in shape: [bs_curr, n_seed, 384]
            # 展平为 [bs_curr, n_seed*384]
            # 但实际输入可能已经是展平的，所以先检查形状
            if seed_in.ndim == 3:
                # 三维输入：[bs_curr, n_seed, 384]
                # 取前4帧以匹配输入维度
                if seed_in.shape[1] > 4:
                    seed_in = seed_in[:, :4, :]
                emb_seed = self.embed_text(seed_in.reshape(bs_curr, -1))
            else:
                # 二维输入：[bs_curr, n_seed*384]
                # 取前1536个特征以匹配输入维度
                if seed_in.shape[1] > 1536:
                    seed_in = seed_in[:, :1536]
                emb_seed = self.embed_text(seed_in)
        else:
            emb_seed = 0 # 避免未定义

        # 3. Input Process
        xseq = self.input_process(x_in)

        # 4. Fuse Style/Time
        # [bs_curr, joint_num, nframes, latent_dim]
        embed_style_2 = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.joint_num, nframes, -1)
        xseq = torch.cat([embed_style_2, xseq], axis=-1)
        xseq = self.input_process2(xseq)
        
        # 5. Rotary Positional Embedding
        xseq = xseq.reshape(bs_curr * self.joint_num, nframes, -1)
        pos_emb = self.rel_pos(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq = xseq.reshape(bs_curr, self.joint_num, nframes, -1)
        # -> [bs_curr, nframes, joint_num * latent_dim]
        xseq = xseq.permute(0, 2, 1, 3).reshape(bs_curr, nframes, self.joint_num * self.latent_dim)

        # 6. Cross Attention Blocks (Condition injection)
        for block in self.cross_attn_blocks:
            xseq = block(xseq, at_feat_in)

        # 7. Temporal Blocks (Self Attention)
        # -> [bs_curr, nframes, joint_num, latent_dim] -> [bs_curr, joint_num, nframes, latent_dim]
        xseq = xseq.reshape(bs_curr, nframes, self.joint_num, self.latent_dim).permute(0, 2, 1, 3)
        for block in self.temporal_blocks:
            xseq = block(xseq)
        
        # --- 双向交互与输出处理 ---
        if do_interaction:
            # [关键] 拆分特征
            # xseq 目前是 [2*B, J, T, D]，需要拆分为 Self 和 Other
            xseq_self, xseq_other = torch.chunk(xseq, 2, dim=0)
            
            # 准备交互输入: Flatten J and D -> [B, T, J*D]
            bs_orig = x.shape[0]
            xseq_self_flat = xseq_self.permute(0, 2, 1, 3).reshape(bs_orig, nframes, self.joint_num * self.latent_dim)
            xseq_other_flat = xseq_other.permute(0, 2, 1, 3).reshape(bs_orig, nframes, self.joint_num * self.latent_dim)
            
            # 执行交互
            for bi_block in self.bidirectional_attn_blocks:
                xseq_self_flat, xseq_other_flat = bi_block(xseq_self_flat, xseq_other_flat)
            
            # 恢复形状用于 OutputProcess: [B, T, J*D] -> [B, T, J, D] -> [B, J, T, D]
            output_feat = xseq_self_flat.reshape(bs_orig, nframes, self.joint_num, self.latent_dim).permute(0, 2, 1, 3)
        else:
            output_feat = xseq

        # Final Projection
        output = self.output_process(output_feat)
        
        return output