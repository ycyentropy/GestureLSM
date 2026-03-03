import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.layer import BasicBlock
from einops import rearrange
import pickle
import math
from models.wavlm.WavLM import WavLM, WavLMConfig


class WavEncoder(nn.Module):
    """
    音频特征编码器（简化版）
    
    输入：[batch, seq_len, audio_dim]
    输出：[batch, seq_len, out_dim]
    
    不再使用过度下采样的卷积网络，而是使用简单的线性投影
    """
    def __init__(self, audio_dim, out_dim):
        super().__init__() 
        self.out_dim = out_dim
        self.audio_dim = audio_dim
        
        self.projection = nn.Sequential(
            nn.Linear(audio_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, wav_data):
        """
        Args:
            wav_data: [batch, seq_len, audio_dim]
        Returns:
            [batch, seq_len, out_dim]
        """
        return self.projection(wav_data)


class ModalityEncoder(nn.Module):
    def __init__(self, 
                 data_path, 
                 t_fix_pre, 
                 audio_dim, 
                 audio_in=128,
                 raw_audio=False,
                 latent_dim=256,
                 audio_fps=30,
                 use_exp=False,
                 target_length=None,
                 val_target_length=None,
                 spatial_temporal=False,
                 vocab_path=None,
                 vqvae_squeeze_scale=4,
                 ):
        super().__init__()
        
        self.raw_audio = raw_audio
        self.latent_dim = latent_dim
        self.audio_fps = audio_fps
        self.audio_dim = audio_dim
        self.audio_in = audio_in
        self.vqvae_squeeze_scale = vqvae_squeeze_scale
        
        if vocab_path is None:
            vocab_path = f"{data_path}weights/vocab.pkl"
        with open(vocab_path, 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding), freeze=t_fix_pre
        )
        word_dim = pre_trained_embedding.shape[1]
        
        self.wav_encoder = WavEncoder(audio_in, audio_dim)
        self.text_encoder_body = nn.Linear(word_dim, audio_dim)

        if self.raw_audio:
            self.audio_projection = nn.Linear(1024, audio_dim)

        if self.raw_audio:
            if use_exp:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim * (4 if spatial_temporal else 1))
            else:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim * (3 if spatial_temporal else 1))
        else:
            if use_exp:
                self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim * (4 if spatial_temporal else 1))
            else:
                self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim * (3 if spatial_temporal else 1))
    
    def forward(self, audio, word, raw_audio=None, squeeze_scale=4):
        """
        参考 modality_encoder_raw.py 的处理流程：
        1. 编码音频和文本
        2. 插值到中间长度（text_len），在更高分辨率上融合
        3. 拼接
        4. 投影
        5. 池化到目标长度
        
        Args:
            audio: [batch, audio_seq_len, audio_in] 音频特征（Mel 频谱）
            word: [batch, text_seq_len] 文本 token ID（与 pose 对齐）
            raw_audio: 可选的原始音频特征（WavLM）
            squeeze_scale: VQ-VAE 下采样倍数（默认 4）
        
        Returns:
            [batch, target_len, latent_dim] 多模态特征
            输出长度与 VQ-VAE 的 latent 长度对齐
        """
        batch_size = audio.shape[0]
        audio_len = audio.shape[1]
        
        word_clamped = torch.clamp(word, 0, self.text_pre_encoder_body.num_embeddings - 1)
        text_feat = self.text_encoder_body(self.text_pre_encoder_body(word_clamped))
        text_len = text_feat.shape[1]
        
        audio_feat = self.wav_encoder(audio)
        
        if audio_feat.shape[1] != text_len:
            audio_feat = F.interpolate(
                audio_feat.transpose(1, 2),
                size=text_len,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
        
        if raw_audio is not None and self.raw_audio:
            raw_feat = self.audio_projection(raw_audio)
            
            if raw_feat.shape[1] != text_len:
                raw_feat = F.interpolate(
                    raw_feat.transpose(1, 2),
                    size=text_len,
                    mode='linear',
                    align_corners=True
                ).transpose(1, 2)
            
            at_feat = torch.cat([audio_feat, raw_feat, text_feat], dim=2)
        else:
            at_feat = torch.cat([audio_feat, text_feat], dim=2)
        
        at_feat = self.mix_audio_text(at_feat)
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)
        at_feat = at_feat.transpose(1, 2)
        
        return at_feat

    @torch.no_grad()
    def load_and_freeze_wavlm(self, wavlm_path='./dataloaders/wavlm/WavLM-Base+.pt'):
        checkpoint = torch.load(wavlm_path)
        self.wavlm_cfg = WavLMConfig(checkpoint['cfg'])
        self.audio_encoder = WavLM(self.wavlm_cfg)
        self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    

    def extract_wavlm_feats(self, wav_input_16khz):
        assert self.audio_encoder is not None, "Please load the wavlm model first"
        if isinstance(wav_input_16khz, np.ndarray):
            wav_input_16khz = torch.from_numpy(wav_input_16khz)
        if wav_input_16khz.dim() == 1:
            wav_input_16khz = wav_input_16khz.unsqueeze(0)
        wav_input_16khz = wav_input_16khz.cuda()

        if self.wavlm_cfg.normalize:
            wav_input_16khz = F.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        
        wavlm_feats = self.audio_encoder.extract_features(wav_input_16khz)[0]
        wavlm_feats = wavlm_feats.detach()
        
        target_size = math.ceil(wavlm_feats.shape[1] / 50 * self.audio_fps)
        wavlm_feats = F.interpolate(
            wavlm_feats.transpose(1, 2),
            size=target_size,
            align_corners=True,
            mode='linear'
        ).transpose(1, 2)
        return wavlm_feats
