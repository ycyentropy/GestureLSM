import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers.utils import *
from .layers.transformer import SpatialTemporalBlock, CrossAttentionBlock
from .layers.interaction_layers import CausalFeedbackCrossAttention, get_causal_mask


class InteractiveGestureDenoiser(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_cross_attn_layers=3,
        num_feedback_attn_layers=1,
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
        use_trans=False,
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
        self.use_trans = use_trans

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.cross_attn_context_dim = self.latent_dim * self.joint_num

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
        
        self.feedback_cross_attn_blocks = nn.ModuleList([
            CausalFeedbackCrossAttention(
                dim=self.latent_dim * self.joint_num,
                num_heads=self.num_heads,
                mlp_ratio=self.ff_size // self.latent_dim,
                drop_path=self.dropout,
                context_dim=self.input_dim * self.joint_num,
            )
            for _ in range(num_feedback_attn_layers)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.ff_size // self.latent_dim,
                drop_path=self.dropout
            ) 
            for _ in range(self.num_layers)
        ])
        
        self.embed_text = nn.Linear(self.input_dim * self.joint_num * self.n_seed, self.latent_dim)

        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

        self.rel_pos = SinusoidalEmbeddings(self.latent_dim)
        self.input_process = InputProcess(self.input_dim, self.latent_dim)
        self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim, self.activation, cond_proj_dim=cond_proj_dim, zero_init_cond=True)
        time_dim = self.latent_dim
        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        if cond_proj_dim is not None:
            self.cond_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        
        self.null_cond_embed = nn.Parameter(torch.zeros(self.seq_len, self.latent_dim * self.joint_num), requires_grad=True)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_null_cond_embed(self, target_seq_len, target_feat_dim, device):
        """
        获取 null condition embedding
        
        Args:
            target_seq_len: 目标序列长度（时间维度）
            target_feat_dim: 目标特征维度
            device: 设备
            
        Raises:
            ValueError: 如果维度不匹配
        """
        current_seq_len, current_feat_dim = self.null_cond_embed.shape
        
        if current_seq_len != target_seq_len or current_feat_dim != target_feat_dim:
            raise ValueError(
                f"null_cond_embed dimension mismatch! "
                f"Expected ({target_seq_len}, {target_feat_dim}), "
                f"but got ({current_seq_len}, {current_feat_dim}). "
                f"Please check config: seq_len should equal audio feature sequence length, "
                f"and latent_dim*joint_num should match audio feature dimension."
            )
        
        return self.null_cond_embed.to(device)

    def forward(self, x, timesteps, cond_time=None, seed=None, at_feat=None, listener_latent=None):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        if x.shape[2] == 1 and x.dim() > 3: 
            x = x.squeeze(2)
            x = x.reshape(x.shape[0], self.joint_num, -1, x.shape[2])

        bs_curr, njoints, nfeats, nframes = x.shape
        
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=x.dtype)

        if cond_time is not None and hasattr(self, 'cond_proj') and self.cond_proj is not None:
            cond_time = cond_time.expand(bs_curr).clone()
            cond_emb = self.cond_proj(cond_time)
            cond_emb = cond_emb.to(dtype=x.dtype)
            emb_t = self.time_embedding(time_emb, cond_emb)
        else:
            emb_t = self.time_embedding(time_emb)
        
        if self.n_seed != 0 and seed is not None and self.embed_text is not None:
            if seed.ndim == 3:
                emb_seed = self.embed_text(seed.reshape(bs_curr, -1))
            else:
                emb_seed = self.embed_text(seed)
        else:
            emb_seed = 0

        xseq = self.input_process(x)

        embed_style_2 = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.joint_num, nframes, -1)
        xseq = torch.cat([embed_style_2, xseq], axis=-1)
        xseq = self.input_process2(xseq)
        
        xseq = xseq.reshape(bs_curr * self.joint_num, nframes, -1)
        pos_emb = self.rel_pos(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq = xseq.reshape(bs_curr, self.joint_num, nframes, -1)

        for block in self.temporal_blocks:
            xseq = block(xseq)

        xseq = xseq.permute(0, 2, 1, 3).reshape(bs_curr, nframes, self.joint_num * self.latent_dim)

        for block in self.cross_attn_blocks:
            xseq = block(xseq, at_feat)

        if listener_latent is not None:
            for block in self.feedback_cross_attn_blocks:
                xseq = block(xseq, listener_latent)
        
        output = self.output_process(xseq.reshape(bs_curr, nframes, self.joint_num, self.latent_dim).permute(0, 2, 1, 3))
        
        return output
