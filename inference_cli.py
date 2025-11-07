#!/usr/bin/env python3
"""
命令行版本的GestureLSM推理脚本
基于demo.py，但去掉了Gradio Web界面，改为通过命令行指定音频输入路径
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import warnings
import random
import subprocess
from omegaconf import OmegaConf
from loguru import logger
from dataloaders.build_vocab import Vocab  # 添加Vocab类的导入

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import config, logger_tools, other_tools_hf, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from utils import rotation_conversions as rc
from models.vq.model import RVQVAE
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 初始化ASR模型
pipe = pipeline(
  "automatic-speech-recognition",
  model="local_models/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)       

debug = False


class BaseTrainer(object):
    def __init__(self, args, cfg, audio_path):
        """初始化训练器，用于推理"""
        # 保存配置
        self.args = args
        self.cfg = cfg
        
        # 创建临时目录
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend
        tmp_dir = args.out_path + "custom/"+ time_name_expend + hf_dir
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/")
        
        # 处理音频文件
        self.audio_path = tmp_dir + "/tmp.wav"
        if isinstance(audio_path, str):
            # 如果是文件路径，加载音频
            audio, sr = librosa.load(audio_path, sr=args.audio_sr)
            sf.write(self.audio_path, audio, sr)
        else:
            # 如果是音频数据，直接写入
            sf.write(self.audio_path, audio_path[1], audio_path[0])
        
        # 使用ASR模型获取文本转录
        file_path = tmp_dir+"/tmp.lab"
        self.textgrid_path = tmp_dir + "/tmp.TextGrid"
        if not debug:
            audio, ssr = librosa.load(self.audio_path, sr=args.audio_sr)
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # 使用MFA进行强制对齐
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            result = subprocess.run(command, capture_output=True, text=True)
            print(result)
        
        # 设置参数
        self.args = args
        self.rank = 0
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        
        # 加载数据集
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
        )
        logger.info(f"Init test dataloader success")
        
        # 加载模型
        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        
        # 加载SMPLX模型
        import smplx
        self.smplx = smplx.create(
            args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()
        
        # 设置关节掩码
        from dataloaders.data_tools import joints_list
        self.ori_joint_list = joints_list[args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        
        # 初始化VQ-VAE模型
        self._init_vqvae_models()
        
        # 设置归一化参数
        self.use_trans = args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        # 提取身体部位特定的归一化
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).cuda())
        
        # 平移归一化（如果需要）
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()
    
    def _init_vqvae_models(self):
        """初始化VQ-VAE模型"""
        # 面部VQ模型
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_face, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", 
                                   self.args.e_name)
        
        # 身体部位VQ模型
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }
        
        self.vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config['dim_pose'], part)
            self.vq_models[part] = model
        
        # 设置所有VQ模型为评估模式
        self.vq_model_face.eval().to(self.rank)
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vqvae_latent_scale = self.args.vqvae_latent_scale
        self.args.vae_length = 240
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """创建单个RVQVAE模型"""
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        
        # 加载预训练权重
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        return model
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        """反选择操作，将过滤后的张量恢复到原始形状"""
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        """加载数据"""
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg

        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
      
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        style_feature = None
        
        return {
            "in_audio": in_audio,
            "wavlm": wavlm,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "tar_contact": tar_contact,
            "style_feature":style_feature,
        }
    
    def inference(self, audio_path=None):
        """执行推理过程"""
        # 如果提供了新的音频路径，更新音频文件
        if audio_path and audio_path != self.audio_path:
            audio, sr = librosa.load(audio_path, sr=self.args.audio_sr)
            sf.write(self.audio_path, audio, sr)
            
            # 重新进行ASR和MFA处理
            file_path = os.path.join(os.path.dirname(self.audio_path), "tmp.lab")
            if not debug:
                text = pipe(audio, batch_size=8)["text"]
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(text)
                
                # 使用MFA进行强制对齐
                command = ["mfa", "align", os.path.dirname(self.audio_path), "english_us_arpa", "english_us_arpa", os.path.dirname(self.audio_path)]
                result = subprocess.run(command, capture_output=True, text=True)
                print(result)
        
        # 执行推理
        results = self.test_demo(999)
        return results
    
    def _g_test(self, loaded_data):
        """生成测试数据"""
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]

            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

            shape_ = (bs, 3*128, 1, 32)
            sample = self.model(cond_)['latents']
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            last_sample = sample.clone()
            
            rec_latent_upper = sample[...,:128]
            rec_latent_hands = sample[...,128:2*128]
            rec_latent_lower = sample[...,2*128:]
            
            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1]=rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower

        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_exps = tar_exps
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }
    
    def test_demo(self, epoch):
        """测试演示方法，执行推理并保存结果"""
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        self.model.eval()
        self.smplx.eval()
        
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                
                # 使用示例NPZ文件获取betas
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                results_npz_file_save_path = results_save_path+f"result_{self.time_name_expend}"+'.npz'
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                
                # 渲染视频
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_npz_file_save_path, 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib=False,
                    args=self.args
                    )
                
                # 返回结果
                results = {
                    'video_path': render_vid_path,
                    'npz_path': results_npz_file_save_path,
                    'motion_data': {
                        'betas': gt_npz["betas"],
                        'poses': rec_pose_np,
                        'expressions': rec_exp_np,
                        'trans': rec_trans_np,
                        'model': 'smplx2020',
                        'gender': 'neutral',
                        'mocap_frame_rate': 30,
                    }
                }
        
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GestureLSM命令行推理工具')
    
    # 必需参数
    parser.add_argument('--audio_path', type=str, required=True,
                        help='输入音频文件的路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径，例如: configs/shortcut_rvqvae_128_hf.yaml')
    
    # 可选参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录，默认为 ./outputs')
    parser.add_argument('--output_name', type=str, default=None,
                        help='输出文件名前缀，如果不指定则使用音频文件名')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--render_video', action='store_true',
                        help='是否渲染视频输出')
    parser.add_argument('--save_npz', action='store_true', default=True,
                        help='是否保存NPZ格式的结果')
    
    # 渲染参数
    parser.add_argument('--render_video_fps', type=int, default=30,
                        help='渲染视频的帧率')
    parser.add_argument('--render_video_width', type=int, default=1000,
                        help='渲染视频的宽度')
    parser.add_argument('--render_video_height', type=int, default=1000,
                        help='渲染视频的高度')
    parser.add_argument('--render_concurrent_num', type=int, default=1,
                        help='渲染时的并发数')
    parser.add_argument('--render_tmp_img_filetype', type=str, default='jpg',
                        help='渲染时临时图像文件类型')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只渲染一秒视频')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    args, cfg = config.parse_args(config_path)
    return args, cfg


def main():
    """主函数"""
    # 解析命令行参数
    cmd_args = parse_args()
    
    # 加载配置文件
    args, cfg = load_config(cmd_args.config)
    
    # 更新配置
    args.audio_path = cmd_args.audio_path
    args.out_path = cmd_args.output_dir
    args.output_name = cmd_args.output_name
    args.seed = cmd_args.seed
    args.render_video = cmd_args.render_video
    args.save_npz = cmd_args.save_npz
    
    # 渲染参数
    args.render_video_fps = cmd_args.render_video_fps
    args.render_video_width = cmd_args.render_video_width
    args.render_video_height = cmd_args.render_video_height
    args.render_concurrent_num = cmd_args.render_concurrent_num
    args.render_tmp_img_filetype = cmd_args.render_tmp_img_filetype
    args.debug = cmd_args.debug
    
    print("=" * 50)
    print("GestureLSM 命令行推理工具")
    print("=" * 50)
    print(f"音频文件: {args.audio_path}")
    print(f"配置文件: {cmd_args.config}")
    print(f"输出目录: {args.out_path}")
    print(f"随机种子: {args.seed}")
    print(f"渲染视频: {'是' if args.render_video else '否'}")
    print("=" * 50)
    
    try:
        # 设置随机种子
        other_tools_hf.set_random_seed(args)
        other_tools_hf.print_exp_info(args)
        
        # 初始化训练器
        print("初始化模型...")
        trainer = BaseTrainer(args, cfg, args.audio_path)
        
        # 加载模型检查点
        print(f"加载模型检查点: {args.test_ckpt}")
        other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
        
        # 运行推理
        print("开始推理...")
        results = trainer.inference()
        
        # 创建输出目录
        os.makedirs(args.out_path, exist_ok=True)
        
        # 确定输出文件名
        if args.output_name is None:
            args.output_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        
        # 复制结果文件到输出目录
        import shutil
        if args.save_npz and 'npz_path' in results:
            final_npz_path = os.path.join(args.out_path, f"{args.output_name}.npz")
            shutil.copy(results['npz_path'], final_npz_path)
            print(f"NPZ文件已保存到: {final_npz_path}")
        
        if args.render_video and 'video_path' in results:
            final_video_path = os.path.join(args.out_path, f"{args.output_name}.mp4")
            shutil.copy(results['video_path'], final_video_path)
            print(f"视频已保存到: {final_video_path}")
        
        print("=" * 50)
        print("推理完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()