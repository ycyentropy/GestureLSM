import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Mlp, DropPath, LayerScale


class CausalCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            context_dim=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.context_dim = context_dim if context_dim is not None else dim
        self.kv = nn.Linear(self.context_dim, dim * 2, bias=False)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, attn_mask=None):
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalFeedbackCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            context_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.cross_attn = CausalCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            context_dim=context_dim,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context, causal_mask=None):
        x = x + self.drop_path1(self.ls1(
            self.cross_attn(self.norm1(x), context, attn_mask=causal_mask)
        ))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask
