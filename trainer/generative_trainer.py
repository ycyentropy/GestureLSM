import os
import pprint
import random
import sys
import time
import warnings
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataloaders import data_tools
from dataloaders.data_tools import joints_list
from loguru import logger
from models.vq.model import RVQVAE
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from utils import (
    data_transfer,
    logger_tools,
    metric,
    other_tools,
    other_tools_hf,
    rotation_conversions as rc,
)
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask


def convert_15d_to_6d(motion):
    """
    Convert 15D motion to 6D motion, the current motion is 15D, but the eval model is 6D
    """
    bs = motion.shape[0]
    motion_6d = motion.reshape(bs, -1, 55, 15)[:, :, :, 6:12]
    motion_6d = motion_6d.reshape(bs, -1, 55 * 6)
    return motion_6d


class CustomTrainer(BaseTrainer):
    """
    Generative Trainer to support various generative models
    """

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        self.joints = 55
        self.max_iterations = 50  # 默认值

        self.ori_joint_list = joints_list["beat_smplx_joints"]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[
                self.ori_joint_list[joint_name][1]
                - self.ori_joint_list[joint_name][0] : self.ori_joint_list[joint_name][
                    1
                ]
            ] = 1

        self.tracker = other_tools.EpochTracker(
            ["fgd", "bc", "l1div", "l1div_gt", "align_gt", "predict_x0_loss", "test_clip_fgd"],
            [True, True, True, True, True, True, True],
        )

        ##### Model #####

        model_module = __import__(
            f"models.{cfg.model.model_name}", fromlist=["something"]
        )

        if self.cfg.ddp:
            logger.info(f"Creating model {cfg.model.g_name} on GPU {self.rank}...")
            self.model = getattr(model_module, cfg.model.g_name)(cfg).to(self.rank)
            logger.info(f"Model created on GPU {self.rank}, synchronizing all processes...")
            
            # Synchronize all processes
            if self.rank == 0:
                logger.info("Rank 0: Waiting for all ranks to create model...")
            torch.distributed.barrier()
            logger.info(f"All processes synchronized after model creation on GPU {self.rank}")
            
            logger.info(f"Creating process group on GPU {self.rank}...")
            process_group = torch.distributed.new_group()
            logger.info(f"Process group created on GPU {self.rank}, converting to SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group
            )
            logger.info(f"SyncBatchNorm converted on GPU {self.rank}, synchronizing before DDP...")
            
            if self.rank == 0:
                logger.info("Rank 0: Waiting for all ranks to convert SyncBatchNorm...")
            torch.distributed.barrier()
            logger.info(f"All processes synchronized, wrapping with DDP on GPU {self.rank}...")
            
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            logger.info(f"DDP wrapper created on GPU {self.rank}, final synchronization...")
            
            if self.rank == 0:
                logger.info("Rank 0: Waiting for all ranks to create DDP wrapper...")
            torch.distributed.barrier()
            logger.info(f"All processes synchronized after DDP wrapper on GPU {self.rank}")
        else:
            logger.info(f"Creating model with DataParallel on GPU {self.rank}...")
            self.model = torch.nn.DataParallel(
                getattr(model_module, cfg.model.g_name)(cfg), self.cfg.gpus
            ).cuda()
            logger.info(f"DataParallel model created on GPU {self.rank}")

        if self.args.mode == "train":
            if self.rank == 0:
                logger.info(self.model)
                logger.info(f"init {self.cfg.model.g_name} success")
                wandb.watch(self.model)

        ##### Optimizer and Scheduler #####
        self.opt = create_optimizer(self.cfg.solver, self.model)
        self.opt_s = create_scheduler(self.cfg.solver, self.opt)

        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Body part VQ models
        self.vq_models = self._create_body_vq_models()

        # Set all VQ models to eval mode and move to the corresponding GPU
        for model in self.vq_models.values():
            model.eval().to(self.rank)

        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = (
            self.vq_models.values()
        )

        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction="mean").to(self.rank)

        ##### Normalization #####
        mean_pose_path = getattr(cfg, 'mean_pose_path', './mean_std/beatx_2_330_mean.npy')
        std_pose_path = getattr(cfg, 'std_pose_path', './mean_std/beatx_2_330_std.npy')
        self.mean = np.load(mean_pose_path)
        self.std = np.load(std_pose_path)

        # Extract body part specific normalizations
        for part in ["upper", "hands", "lower"]:
            mask = globals()[f"{part}_body_mask"]
            setattr(self, f"mean_{part}", torch.from_numpy(self.mean[mask]).to(self.rank))
            setattr(self, f"std_{part}", torch.from_numpy(self.std[mask]).to(self.rank))

        mean_trans_path = getattr(cfg, 'mean_trans_path', './mean_std/beatx_2_trans_mean.npy')
        std_trans_path = getattr(cfg, 'std_trans_path', './mean_std/beatx_2_trans_std.npy')
        self.trans_mean = torch.from_numpy(
            np.load(mean_trans_path)
        ).to(self.rank)
        self.trans_std = torch.from_numpy(
            np.load(std_trans_path)
        ).to(self.rank)

        if self.args.checkpoint:
            try:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state_dict"
                ]
            except:
                ckpt_state_dict = torch.load(self.args.checkpoint, weights_only=False)[
                    "model_state"
                ]
            # remove 'audioEncoder' from the state_dict due to legacy issues and text_pre_encoder_body due to vocab size mismatch
            ckpt_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if "modality_encoder.audio_encoder." not in k and "modality_encoder.text_pre_encoder_body." not in k
            }
            self.model.load_state_dict(ckpt_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {self.args.checkpoint}")

    def _create_body_vq_models(self) -> Dict[str, RVQVAE]:
        """Create VQ-VAE models for body parts."""
        vq_configs = {
            "upper": {"dim_pose": 78},
            "hands": {"dim_pose": 180},
            "lower": {"dim_pose": 57},
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config["dim_pose"], part)
            vq_models[part] = model

        return vq_models

    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""

        vq_args = self.args
        vq_args.num_quantizers = 6
        vq_args.shared_codebook = False
        vq_args.quantize_dropout_prob = 0.2
        vq_args.quantize_dropout_cutoff_index = 0
        vq_args.mu = 0.99
        vq_args.beta = 1.0
        model = RVQVAE(
            vq_args,
            input_width=dim_pose,
            nb_code=1024,
            code_dim=128,
            output_emb_width=128,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        # Load pretrained weights
        checkpoint_path = getattr(self.cfg, f"vqvae_{body_part}_path")
        model.load_state_dict(torch.load(checkpoint_path)["net"])
        return model

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.rank)
        original_shape_t = torch.zeros((n, 165)).to(self.rank)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def _load_data(self, dict_data):
        facial_rep = dict_data["facial"].to(self.rank)
        beta = dict_data["beta"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank)

        # process the pose data
        tar_pose = dict_data["pose"][:, :, :165].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        tar_pose_hands = tar_pose[:, :, 25 * 3 : 55 * 3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)

        tar_pose_lower = tar_pose_leg

        tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
        tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
        tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower

        # tar_trans_v = (tar_trans_v - self.trans_mean) / self.trans_std 暂时注释，测试效果
        tar_pose_lower = torch.cat([tar_pose_lower, tar_trans_v], dim=-1)

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        

        ## TODO: Whether the latent scale is needed here?
        # latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        latent_in = (
            torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) / 5
        )

        word = dict_data.get("word", None)
        if word is not None:
            word = word.to(self.rank)

        # style feature is always None (without annotation, we never know what it is)
        style_feature = None

        audio_onset = None
        if self.cfg.data.onset_rep:
            audio_onset = dict_data["audio_onset"].to(self.rank)
        
        # Get audio name if available
        audio_name = dict_data.get("audio_name", None)

        return {
            "audio_onset": audio_onset,
            "word": word,
            "latent_in": latent_in,
            "tar_id": tar_id,
            "facial_rep": facial_rep,
            "beta": beta,
            "tar_pose": tar_pose,
            "trans": tar_trans,
            "style_feature": style_feature,
            "audio_name": audio_name,
        }

    def _g_training(self, loaded_data, mode="train", epoch=0):
        self.model.train()
        cond_ = {"y": {}}
        cond_["y"]["audio_onset"] = loaded_data["audio_onset"]
        cond_["y"]["word"] = loaded_data["word"]
        cond_["y"]["id"] = loaded_data["tar_id"]
        cond_["y"]["seed"] = loaded_data["latent_in"][:, : self.cfg.pre_frames]
        cond_["y"]["style_feature"] = loaded_data["style_feature"]
        x0 = loaded_data["latent_in"]
        x0 = x0.permute(0, 2, 1).unsqueeze(2)

        g_loss_final = self.model.module.train_forward(cond_, x0)["loss"]

        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == "train":
            return g_loss_final

    def _g_test(self, loaded_data):
        self.model.eval()
        tar_beta = loaded_data["beta"]
        tar_pose = loaded_data["tar_pose"]
        tar_exps = loaded_data["facial_rep"]
        tar_trans = loaded_data["trans"]

        audio_onset = loaded_data["audio_onset"]
        in_word = loaded_data["word"]

        in_x0 = loaded_data["latent_in"]
        in_seed = loaded_data["latent_in"]

        bs, n, j = (
            loaded_data["tar_pose"].shape[0],
            loaded_data["tar_pose"].shape[1],
            self.joints,
        )

        remain = n % 8
        if remain != 0:

            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_exps = tar_exps[:, :-remain, :]
            in_x0 = in_x0[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_seed = in_seed[
                :, : in_x0.shape[1] - (remain // self.cfg.vqvae_squeeze_scale), :
            ]
            in_word = in_word[:, :-remain]
            n = n - remain

        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.cfg.vqvae_squeeze_scale
        pre_frames_scaled = self.cfg.pre_frames * vqvae_squeeze_scale
        roundt = (n - pre_frames_scaled) // (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        remain = (n - pre_frames_scaled) % (
            self.cfg.data.pose_length - pre_frames_scaled
        )
        round_l = self.cfg.pose_length - pre_frames_scaled
        round_audio = int(round_l / 3 * 5)

        in_audio_onset_tmp = None
        in_word_tmp = None
        # Skip the last roundt iteration, adjust n accordingly
        actual_rounds = roundt - 1
        n = n - (round_l + remain)  # Adjust n to match actual generated frames
        
        for i in range(0, actual_rounds):
            if audio_onset is not None:
                # Calculate the audio slice start and end positions
                audio_start = i * (self.cfg.data.audio_sr // self.cfg.data.pose_fps * round_l)
                audio_end = (i + 1) * (self.cfg.data.audio_sr // self.cfg.data.pose_fps * round_l) + self.cfg.data.audio_sr // self.cfg.data.pose_fps * self.cfg.pre_frames * vqvae_squeeze_scale
                
                # If audio slice exceeds boundary, adjust from the end
                if audio_end > audio_onset.shape[1]:
                    audio_end = audio_onset.shape[1]
                    audio_start = max(0, audio_end - self.cfg.data.audio_sr // self.cfg.data.pose_fps * round_l - self.cfg.data.audio_sr // self.cfg.data.pose_fps * self.cfg.pre_frames * vqvae_squeeze_scale)
                
                in_audio_onset_tmp = audio_onset[:, audio_start:audio_end]
            if in_word is not None:
                in_word_tmp = in_word[
                    :,
                    i * (round_l) : (i + 1) * (round_l)
                    + self.cfg.pre_frames * vqvae_squeeze_scale,
                ]

            in_id_tmp = loaded_data["tar_id"][
                :, i * (round_l) : (i + 1) * (round_l) + self.cfg.pre_frames
            ]
            in_seed_tmp = in_seed[
                :,
                i
                * (round_l)
                // vqvae_squeeze_scale : (i + 1)
                * (round_l)
                // vqvae_squeeze_scale
                + self.cfg.pre_frames,
            ]

            if i == 0:
                in_seed_tmp = in_seed_tmp[:, : self.cfg.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.cfg.pre_frames :, :]

            cond_ = {"y": {}}
            cond_["y"]["audio_onset"] = in_audio_onset_tmp
            cond_["y"]["word"] = in_word_tmp
            cond_["y"]["id"] = in_id_tmp
            cond_["y"]["seed"] = in_seed_tmp
            cond_["y"]["style_feature"] = torch.zeros([bs, 512]).to(self.rank)

            sample = self.model(cond_)["latents"]

            sample = sample.squeeze(2).permute(0, 2, 1)

            last_sample = sample.clone()

            code_dim = self.vq_model_upper.code_dim
            rec_latent_upper = sample[..., :code_dim]
            rec_latent_hands = sample[..., code_dim : code_dim * 2]
            rec_latent_lower = sample[..., code_dim * 2 : code_dim * 3]

            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.cfg.pre_frames :])
                rec_all_hands.append(rec_latent_hands[:, self.cfg.pre_frames :])
                rec_all_lower.append(rec_latent_lower[:, self.cfg.pre_frames :])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * 5
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * 5
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * 5

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]

        rec_trans_v = rec_lower[..., -3:]
        # rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean 暂时注释，测试效果
        rec_trans = torch.zeros_like(rec_trans_v)
        rec_trans = torch.cumsum(rec_trans_v, dim=-2)
        rec_trans[..., 1] = rec_trans_v[..., 1]
        rec_lower = rec_lower[..., :-3]

        rec_upper = rec_upper * self.std_upper + self.mean_upper
        rec_hands = rec_hands * self.std_hands + self.mean_hands
        rec_lower = rec_lower * self.std_lower + self.mean_lower

        # n has already been adjusted above, no need to subtract remain again
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        if hasattr(self.cfg.model, "use_exp") and self.cfg.model.use_exp:
            rec_exps = tar_exps  # fallback to tar_exps since rec_face is not defined
        else:
            rec_exps = tar_exps

        rec_trans = tar_trans

        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n
        )
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)

        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n
        )
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n
        )
        rec_pose = (
            rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        )
        rec_pose[:, 66:69] = tar_pose.reshape(bs * n, 55 * 3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        return {
            "rec_pose": rec_pose,
            "rec_exps": rec_exps,
            "rec_trans": rec_trans,
            "tar_pose": tar_pose,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_trans": tar_trans,
        }

    def train(self, epoch):

        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start

            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, "train", epoch)

            g_loss_final.backward()
            if self.cfg.solver.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.solver.grad_norm
                )
            self.opt.step()

            mem_cost = torch.cuda.memory_cached() / 1e9
            lr_g = self.opt.param_groups[0]["lr"]

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.cfg.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.cfg.debug:
                if its == 1:
                    break
        self.opt_s.step(epoch)

    @torch.no_grad()
    def _common_test_inference(
        self, data_loader, epoch, mode="val", max_iterations=None, save_results=False
    ):
        """
        Common inference logic shared by val, test, test_clip, and test_render methods.

        Args:
            data_loader: The data loader to iterate over
            epoch: Current epoch number
            mode: Mode string for logging ("val", "test", "test_clip", "test_render")
            max_iterations: Maximum number of iterations (None for no limit)
            save_results: Whether to save result files

        Returns:
            Dictionary containing computed metrics and results
        """
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file if hasattr(self.test_data, 'selected_file') else None
        align = 0
        align_gt = 0
        latent_out = []
        latent_ori = []
        l2_all = 0
        lvel = 0
        results = []

        # Setup save path for test mode
        results_save_path = None
        if save_results:
            results_save_path = self.checkpoint_path + f"/{epoch}/"
            if mode == "test_render":
                if os.path.exists(results_save_path):
                    import shutil

                    shutil.rmtree(results_save_path)
            os.makedirs(results_save_path, exist_ok=True)

        self.model.eval()
        self.smplx.eval()
        if hasattr(self, "eval_copy"):
            self.eval_copy.eval()

        with torch.no_grad():
            iterator = enumerate(data_loader)
            if mode in ["test_clip", "test"]:
                iterator = enumerate(
                    tqdm(data_loader, desc=f"Testing {mode}", leave=True)
                )

            for its, batch_data in iterator:
                if max_iterations is not None and its > max_iterations:
                    break

                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)

                tar_pose = net_out["tar_pose"]
                rec_pose = net_out["rec_pose"]
                tar_exps = net_out["tar_exps"]
                tar_beta = net_out["tar_beta"]
                rec_trans = net_out["rec_trans"]
                tar_trans = net_out.get("tar_trans", rec_trans)
                rec_exps = net_out.get("rec_exps", tar_exps)

                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # Handle frame rate conversion
                if (30 / self.cfg.data.pose_fps) != 1:
                    assert 30 % self.cfg.data.pose_fps == 0
                    n *= int(30 / self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(
                        tar_pose.permute(0, 2, 1),
                        scale_factor=30 / self.cfg.data.pose_fps,
                        mode="linear",
                    ).permute(0, 2, 1)
                    scale_factor = (
                        30 / self.cfg.data.pose_fps
                        if mode != "test"
                        else 30 / self.cfg.pose_fps
                    )
                    rec_pose = torch.nn.functional.interpolate(
                        rec_pose.permute(0, 2, 1),
                        scale_factor=scale_factor,
                        mode="linear",
                    ).permute(0, 2, 1)

                # Calculate latent representations for evaluation
                if hasattr(self, "eval_copy") and mode != "test_render":
                    remain = n % self.cfg.vae_test_len
                    
                    if n - remain > 0:
                        latent_rec = self.eval_copy.map2latent(rec_pose[:, : n - remain])
                        latent_tar = self.eval_copy.map2latent(tar_pose[:, : n - remain])
                        
                        latent_out.append(
                            latent_rec
                            .reshape(-1, self.cfg.vae_test_len) #self.cfg.vae_length
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        latent_ori.append(
                            latent_tar
                            .reshape(-1, self.cfg.vae_test_len)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs * n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs * n, j * 3)

                # Generate SMPLX vertices and joints
                # Process in smaller chunks to reduce memory usage
                chunk_size = 1024  # Adjust based on available memory
                joints_rec_list = []
                joints_tar_list = []
                
                for i in range(0, bs * n, chunk_size):
                    end_idx = min(i + chunk_size, bs * n)
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300)[i:end_idx],
                        transl=rec_trans.reshape(bs * n, 3)[i:end_idx],
                        expression=tar_exps.reshape(bs * n, 100)[i:end_idx],
                        jaw_pose=rec_pose[:, 66:69][i:end_idx],
                        global_orient=rec_pose[:, :3][i:end_idx],
                        body_pose=rec_pose[:, 3 : 21 * 3 + 3][i:end_idx],
                        left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3][i:end_idx],
                        right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3][i:end_idx],
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72][i:end_idx],
                        reye_pose=rec_pose[:, 72:75][i:end_idx],
                    )
                    joints_rec_list.append(vertices_rec["joints"].detach().cpu().numpy())
                    
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs * n, 300)[i:end_idx],
                        transl=tar_trans.reshape(bs * n, 3)[i:end_idx],
                        expression=tar_exps.reshape(bs * n, 100)[i:end_idx],
                        jaw_pose=tar_pose[:, 66:69][i:end_idx],
                        global_orient=tar_pose[:, :3][i:end_idx],
                        body_pose=tar_pose[:, 3 : 21 * 3 + 3][i:end_idx],
                        left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3][i:end_idx],
                        right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3][i:end_idx],
                        return_joints=True,
                        leye_pose=tar_pose[:, 69:72][i:end_idx],
                        reye_pose=tar_pose[:, 72:75][i:end_idx],
                    )
                    joints_tar_list.append(vertices_tar["joints"].detach().cpu().numpy())
                    
                    del vertices_rec, vertices_tar
                    torch.cuda.empty_cache()
                
                # Concatenate results
                joints_rec = np.concatenate(joints_rec_list, axis=0)
                joints_rec = joints_rec.reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]
                
                joints_tar = np.concatenate(joints_tar_list, axis=0)
                joints_tar = joints_tar.reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]

                # Calculate L1 diversity for generated poses
                if self.l1_calculator is not None:
                    _ = self.l1_calculator.run(joints_rec)

                # Calculate L1 diversity for ground truth poses
                if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None:
                    _ = self.l1_calculator_gt.run(joints_tar)

                # Calculate alignment for single batch
                if (
                    hasattr(self, "alignmenter")
                    and self.alignmenter is not None
                    and bs == 1
                    and mode != "test_render"
                ):
                    if test_seq_list is not None:
                        audio_path = self.cfg.data.data_path + "wave16k/" + test_seq_list.iloc[its]["id"] + ".wav"
                    else:
                        audio_path = loaded_data["audio_name"][0]
                    in_audio_eval, sr = librosa.load(audio_path)
                    in_audio_eval = librosa.resample(
                        in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr
                    )
                    a_offset = int(
                        self.align_mask
                        * (self.cfg.data.audio_sr / self.cfg.data.pose_fps)
                    )
                    onset_bt = self.alignmenter.load_audio(
                        in_audio_eval[
                            : int(self.cfg.data.audio_sr / self.cfg.data.pose_fps * n)
                        ],
                        a_offset,
                        len(in_audio_eval) - a_offset,
                        True,
                    )
                    beat_vel = self.alignmenter.load_pose(
                        joints_rec, self.align_mask, n - self.align_mask, 30, True
                    )
                    align += self.alignmenter.calculate_align(
                        onset_bt, beat_vel, 30
                    ) * (n - 2 * self.align_mask)
                    
                    beat_vel_gt = self.alignmenter.load_pose(
                        joints_tar, self.align_mask, n - self.align_mask, 30, True
                    )
                    align_gt += self.alignmenter.calculate_align(
                        onset_bt, beat_vel_gt, 30
                    ) * (n - 2 * self.align_mask)

                # Calculate face vertices for loss computation
                if mode == "test" or mode == "val":
                    chunk_size = 1024
                    vertices_rec_face_list = []
                    vertices_tar_face_list = []
                    
                    for i in range(0, bs * n, chunk_size):
                        end_idx = min(i + chunk_size, bs * n)
                        
                        vertices_rec_face = self.smplx(
                            betas=tar_beta.reshape(bs * n, 300)[i:end_idx],
                            transl=rec_trans.reshape(bs * n, 3)[i:end_idx],
                            expression=rec_exps.reshape(bs * n, 100)[i:end_idx],
                            jaw_pose=rec_pose[:, 66:69].reshape(bs * n, 3)[i:end_idx],
                            global_orient=rec_pose[:, :3].reshape(bs * n, 3)[i:end_idx],
                            body_pose=rec_pose[:, 3 : 21 * 3 + 3].reshape(bs * n, 63)[i:end_idx],
                            left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3].reshape(bs * n, 45)[i:end_idx],
                            right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3].reshape(bs * n, 45)[i:end_idx],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=rec_pose[:, 69:72].reshape(bs * n, 3)[i:end_idx],
                            reye_pose=rec_pose[:, 72:75].reshape(bs * n, 3)[i:end_idx],
                        )
                        
                        vertices_tar_face = self.smplx(
                            betas=tar_beta.reshape(bs * n, 300)[i:end_idx],
                            transl=tar_trans.reshape(bs * n, 3)[i:end_idx],
                            expression=tar_exps.reshape(bs * n, 100)[i:end_idx],
                            jaw_pose=tar_pose[:, 66:69].reshape(bs * n, 3)[i:end_idx],
                            global_orient=tar_pose[:, :3].reshape(bs * n, 3)[i:end_idx],
                            body_pose=tar_pose[:, 3 : 21 * 3 + 3].reshape(bs * n, 63)[i:end_idx],
                            left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3].reshape(bs * n, 45)[i:end_idx],
                            right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3].reshape(bs * n, 45)[i:end_idx],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=tar_pose[:, 69:72].reshape(bs * n, 3)[i:end_idx],
                            reye_pose=tar_pose[:, 72:75].reshape(bs * n, 3)[i:end_idx],
                        )
                        
                        vertices_rec_face_list.append(vertices_rec_face["vertices"].detach().cpu())
                        vertices_tar_face_list.append(vertices_tar_face["vertices"].detach().cpu())
                        del vertices_rec_face, vertices_tar_face
                        torch.cuda.empty_cache()
                    
                    vertices_rec_face = {"vertices": torch.cat(vertices_rec_face_list, dim=0)}
                    vertices_tar_face = {"vertices": torch.cat(vertices_tar_face_list, dim=0)}

                    facial_rec = (
                        vertices_rec_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    facial_tar = (
                        vertices_tar_face["vertices"].reshape(1, n, -1)[0, :n].cpu()
                    )
                    face_vel_loss = self.vel_loss(
                        facial_rec[1:, :] - facial_tar[:-1, :],
                        facial_tar[1:, :] - facial_tar[:-1, :],
                    )
                    l2 = self.reclatent_loss(facial_rec, facial_tar)
                    l2_all += l2.item() * n
                    lvel += face_vel_loss.item() * n

                # Save results if needed
                if save_results:
                    if mode == "test":
                        # Save NPZ files for test mode
                        tar_pose_np = tar_pose.detach().cpu().numpy()
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_exp_np = (
                            tar_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )
                        tar_trans_np = (
                            tar_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )

                        # Get file ID for saving results
                        if test_seq_list is not None:
                            file_id = test_seq_list.iloc[its]["id"]
                        else:
                            # Extract filename from audio path
                            file_id = os.path.basename(loaded_data["audio_name"][0]).split(".")[0]
                        
                        # Try to load gt_npz from the default path, fallback to demo example if not found
                        try:
                            gt_npz = np.load(
                                self.cfg.data.data_path
                                + self.cfg.data.pose_rep
                                + "/"
                                + file_id
                                + ".npz",
                                allow_pickle=True,
                            )
                        except (FileNotFoundError, IOError):
                            # Fallback to demo example for seamless dataset
                            gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                        np.savez(
                            results_save_path
                            + "gt_"
                            + file_id
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=tar_pose_np,
                            expressions=tar_exp_np,
                            trans=tar_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )
                        np.savez(
                            results_save_path
                            + "res_"
                            + file_id
                            + ".npz",
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                    elif mode == "test_render":
                        # Save results and render for test_render mode
                        audio_name = loaded_data["audio_name"][0]
                        rec_pose_np = rec_pose.detach().cpu().numpy()
                        rec_trans_np = (
                            rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                        )
                        rec_exp_np = (
                            rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                        )

                        gt_npz = np.load(
                            "./demo/examples/2_scott_0_1_1.npz", allow_pickle=True
                        )
                        file_name = audio_name.split("/")[-1].split(".")[0]
                        results_npz_file_save_path = (
                            results_save_path + f"result_{file_name}.npz"
                        )

                        np.savez(
                            results_npz_file_save_path,
                            betas=gt_npz["betas"],
                            poses=rec_pose_np,
                            expressions=rec_exp_np,
                            trans=rec_trans_np,
                            model="smplx2020",
                            gender="neutral",
                            mocap_frame_rate=30,
                        )

                        render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                            results_npz_file_save_path,
                            results_save_path,
                            audio_name,
                            self.cfg.data_path_1 + "smplx_models/",
                            use_matplotlib=False,
                            args=self.cfg,
                        )

                total_length += n

        return {
            "total_length": total_length,
            "align": align,
            "align_gt": align_gt,
            "latent_out": latent_out,
            "latent_ori": latent_ori,
            "l2_all": l2_all,
            "lvel": lvel,
            "start_time": start_time,
        }

    def val(self, epoch):
        self.tracker.reset()
        
        if self.l1_calculator is not None:
            self.l1_calculator.reset()
        if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None:
            self.l1_calculator_gt.reset()

        results = self._common_test_inference(
            self.test_loader, epoch, mode="val", max_iterations=self.max_iterations
        )

        total_length = results["total_length"]
        align = results["align"]
        align_gt = results["align_gt"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)

        # 计算实际处理的样本数
        actual_samples = min(self.max_iterations, len(self.test_loader)) if self.max_iterations is not None else len(self.test_loader)
        
        # 使用实际处理的样本数计算分母
        denominator = total_length - 2 * actual_samples * self.align_mask
        
        align_avg = align / denominator
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        
        align_gt_avg = align_gt / denominator
        logger.info(f"align_gt score: {align_gt_avg}")
        self.tracker.update_meter("align_gt", "val", align_gt_avg)

        l1div = self.l1_calculator.avg() if self.l1_calculator is not None else 0.0
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
        
        l1div_gt = self.l1_calculator_gt.avg() if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None else 0.0
        logger.info(f"l1div_gt score: {l1div_gt}")
        self.tracker.update_meter("l1div_gt", "val", l1div_gt)

        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_clip(self, epoch):
        self.tracker.reset()
        
        if self.l1_calculator is not None:
            self.l1_calculator.reset()
        if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None:
            self.l1_calculator_gt.reset()

        # Test on CLIP dataset
        results_clip = self._common_test_inference(
            self.test_clip_loader, epoch, mode="test_clip", max_iterations=self.max_iterations
        )

        total_length_clip = results_clip["total_length"]
        latent_out_clip = results_clip["latent_out"]
        latent_ori_clip = results_clip["latent_ori"]
        start_time = results_clip["start_time"]

        latent_out_all_clip = np.concatenate(latent_out_clip, axis=0)
        latent_ori_all_clip = np.concatenate(latent_ori_clip, axis=0)

        fgd_clip = data_tools.FIDCalculator.frechet_distance(
            latent_out_all_clip, latent_ori_all_clip
        )
        logger.info(f"test_clip fgd score: {fgd_clip}")
        self.tracker.update_meter("test_clip_fgd", "val", fgd_clip)

        current_time = time.time()
        test_clip_time = current_time - start_time
        logger.info(
            f"total test_clip inference time: {int(test_clip_time)} s for {int(total_length_clip/self.cfg.data.pose_fps)} s motion"
        )

        # Test on regular test dataset for recording
        if self.l1_calculator is not None:
            self.l1_calculator.reset()
        if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None:
            self.l1_calculator_gt.reset()
        
        results_test = self._common_test_inference(
            self.test_loader, epoch, mode="test_clip", max_iterations=self.max_iterations
        )

        total_length = results_test["total_length"]
        align = results_test["align"]
        align_gt = results_test["align_gt"]
        latent_out = results_test["latent_out"]
        latent_ori = results_test["latent_ori"]

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)

        # 计算实际处理的样本数
        actual_samples = min(self.max_iterations, len(self.test_loader)) if self.max_iterations is not None else len(self.test_loader)
        
        # 使用实际处理的样本数计算分母
        denominator = total_length - 2 * actual_samples * self.align_mask
        
        align_avg = align / denominator
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        
        align_gt_avg = align_gt / denominator
        logger.info(f"align_gt score: {align_gt_avg}")
        self.tracker.update_meter("align_gt", "val", align_gt_avg)

        l1div = self.l1_calculator.avg() if self.l1_calculator is not None else 0.0
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
        
        l1div_gt = self.l1_calculator_gt.avg() if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None else 0.0
        logger.info(f"l1div_gt score: {l1div_gt}")
        self.tracker.update_meter("l1div_gt", "val", l1div_gt)

        self.val_recording(epoch)

        end_time = time.time() - current_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)
        
        if self.l1_calculator is not None:
            self.l1_calculator.reset()
        if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None:
            self.l1_calculator_gt.reset()

        results = self._common_test_inference(
            self.test_loader, epoch, mode="test", max_iterations=self.max_iterations, save_results=True
        )

        total_length = results["total_length"]
        align = results["align"]
        align_gt = results["align_gt"]
        latent_out = results["latent_out"]
        latent_ori = results["latent_ori"]
        l2_all = results["l2_all"]
        lvel = results["lvel"]
        start_time = results["start_time"]

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)

        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.test_recording("fgd", fgd, epoch)

        # 计算实际处理的样本数
        actual_samples = min(self.max_iterations, len(self.test_loader)) if self.max_iterations is not None else len(self.test_loader)
        
        # 使用实际处理的样本数计算分母
        denominator = total_length - 2 * actual_samples * self.align_mask
        
        align_avg = align / denominator
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)
        
        align_gt_avg = align_gt / denominator
        logger.info(f"align_gt score: {align_gt_avg}")
        self.test_recording("align_gt", align_gt_avg, epoch)

        l1div = self.l1_calculator.avg() if self.l1_calculator is not None else 0.0
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)
        
        l1div_gt = self.l1_calculator_gt.avg() if hasattr(self, "l1_calculator_gt") and self.l1_calculator_gt is not None else 0.0
        logger.info(f"l1div_gt score: {l1div_gt}")
        self.test_recording("l1div_gt", l1div_gt, epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test_render(self, epoch):
        import platform

        if platform.system() == "Linux":
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        """
        input audio and text, output motion
        do not calculate loss and metric
        save video
        """
        results = self._common_test_inference(
            self.test_loader, epoch, mode="test_render", max_iterations=self.max_iterations, save_results=True
        )

    def load_checkpoint(self, checkpoint):
        # checkpoint is already a dict, do NOT call torch.load again!
        try:
            ckpt_state_dict = checkpoint["model_state_dict"]
        except:
            ckpt_state_dict = checkpoint["model_state"]
        # remove 'audioEncoder' from the state_dict due to legacy issues
        ckpt_state_dict = {
            k: v
            for k, v in ckpt_state_dict.items()
            if "modality_encoder.audio_encoder." not in k
        }
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        try:
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        except:
            print("No optimizer loaded!")
        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.opt_s.load_state_dict(checkpoint["scheduler_state_dict"])
        if "val_best" in checkpoint:
            self.val_best = checkpoint["val_best"]
        logger.info("Checkpoint loaded successfully.")
