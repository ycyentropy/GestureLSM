import os
import pprint
import random
import sys
import time
import warnings
from typing import Dict
from datetime import datetime

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloaders import data_tools
from dataloaders.data_tools import joints_list
from dataloaders.dyadic_feedback_dataset import DyadicFeedbackDataset, collate_fn, collate_fn_single
from loguru import logger
from models.interactive_gesture_lsm import InteractiveGestureLSM, OnlineListenerLoader
from models.vq.model import RVQVAE
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import (
    data_transfer,
    logger_tools,
    metric,
    other_tools,
    other_tools_hf,
    rotation_conversions as rc,
)
from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask


class InteractiveGestureTrainer:
    """
    Trainer for InteractiveGestureLSM with DDP support
    Supports both single-GPU and multi-GPU training
    """

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        if cfg.ddp:
            self.rank = dist.get_rank()
            logger.info(f"[DEBUG] Got rank from dist.get_rank(): {self.rank}")
        else:
            self.rank = 0
        self.checkpoint_path = os.path.join(cfg.output_dir, cfg.exp_name)
        self.joints = 55

        self.val_best = {
            "fgd": {"value": float('inf'), "epoch": 0},
            "l1div": {"value": float('-inf'), "epoch": 0},
        }

        self.ori_joint_list = joints_list["beat_smplx_joints"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

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
            ["fgd", "l1div", "predict_x0_loss"],
            [True, True, True],
        )
        
        self.inference_mode = getattr(cfg.model, "inference_mode", "rdla")
        logger.info(f"Inference mode: {self.inference_mode}")

        self._init_dataloaders()
        self._init_model()
        self._init_vq_models()
        self._init_normalization()
        self._init_eval_model()

        if self.args.checkpoint:
            self._load_checkpoint_from_path(self.args.checkpoint)

    def _init_dataloaders(self):
        logger.info("Creating datasets...")
        
        self.train_dataset = DyadicFeedbackDataset(self.cfg, split='train', build_cache=True)
        self.val_dataset = DyadicFeedbackDataset(self.cfg, split='val', build_cache=True)

        if self.cfg.ddp:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
        else:
            self.train_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_bs,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_single,
        )

        self.train_length = len(self.train_loader)
        logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

        if self.args.mode == "train":
            if self.rank == 0:
                run_time = datetime.now().strftime("%Y%m%d-%H%M")
                run_name = self.cfg.exp_name + "_" + run_time
                if hasattr(self.cfg, 'resume_from_checkpoint') and self.cfg.resume_from_checkpoint:
                    run_name += f"_resumed"

                wandb.init(
                    project=self.cfg.wandb_project,
                    name=run_name,
                    entity=self.cfg.wandb_entity,
                    dir=self.cfg.wandb_log_dir,
                    config=OmegaConf.to_container(self.cfg)
                )

    def _init_model(self):
        model_module = __import__(
            f"models.{self.cfg.model.model_name}", fromlist=["something"]
        )

        if self.cfg.ddp:
            logger.info(f"Creating model {self.cfg.model.g_name} on GPU {self.rank}...")
            self.model = getattr(model_module, self.cfg.model.g_name)(self.cfg).to(self.rank)
            logger.info(f"Model created on GPU {self.rank}, synchronizing all processes...")

            dist.barrier()
            logger.info(f"All processes synchronized after model creation on GPU {self.rank}")

            process_group = dist.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group
            )

            dist.barrier()
            logger.info(f"Wrapping with DDP on GPU {self.rank}...")

            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            logger.info(f"DDP wrapper created on GPU {self.rank}")
        else:
            logger.info(f"Creating model on GPU...")
            self.model = getattr(model_module, self.cfg.model.g_name)(self.cfg).cuda()
            logger.info(f"Model created")

        if self.args.mode == "train":
            if self.rank == 0:
                logger.info(self.model)
                logger.info(f"init {self.cfg.model.g_name} success")
                wandb.watch(self.model)

        self.opt = create_optimizer(self.cfg.solver, self.model)
        self.opt_s = create_scheduler(self.cfg.solver, self.opt)

    def _init_vq_models(self):
        self.vq_models = self._create_body_vq_models()
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = (
            self.vq_models.values()
        )

    def _create_body_vq_models(self) -> Dict[str, RVQVAE]:
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
        checkpoint_path = getattr(self.cfg, f"vqvae_{body_part}_path")
        model.load_state_dict(torch.load(checkpoint_path)["net"])
        return model

    def _init_normalization(self):
        mean_pose_path = getattr(self.cfg, 'mean_pose_path', './mean_std/beatx_2_330_mean.npy')
        std_pose_path = getattr(self.cfg, 'std_pose_path', './mean_std/beatx_2_330_std.npy')
        self.mean = np.load(mean_pose_path)
        self.std = np.load(std_pose_path)

        for part in ["upper", "hands", "lower"]:
            mask = globals()[f"{part}_body_mask"]
            setattr(self, f"mean_{part}", torch.from_numpy(self.mean[mask]).to(self.rank))
            setattr(self, f"std_{part}", torch.from_numpy(self.std[mask]).to(self.rank))

        mean_trans_path = getattr(self.cfg, 'mean_trans_path', './mean_std/beatx_2_trans_mean.npy')
        std_trans_path = getattr(self.cfg, 'std_trans_path', './mean_std/beatx_2_trans_std.npy')
        self.trans_mean = torch.from_numpy(np.load(mean_trans_path)).to(self.rank)
        self.trans_std = torch.from_numpy(np.load(std_trans_path)).to(self.rank)

    def _init_eval_model(self):
        eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        eval_args = type('Args', (), {})()
        eval_args.vae_layer = 4
        eval_args.vae_length = 240
        eval_args.vae_test_dim = 330
        eval_args.variational = False
        eval_args.data_path_1 = "./datasets/hub/"
        eval_args.vae_grow = [1, 1, 2, 1]

        self.eval_copy = getattr(eval_model_module, 'VAESKConv')(eval_args).to(self.rank)
        logger.info(f"VAESKConv model created on GPU {self.rank}, loading checkpoints...")
        other_tools.load_checkpoints(
            self.eval_copy,
            './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/AESKConv_240_100.bin',
            'VAESKConv'
        )
        self.eval_copy.eval()
        logger.info(f"VAESKConv checkpoints loaded successfully on GPU {self.rank}")

        self.l1_calculator = metric.L1div() if self.rank == 0 else None

    def _load_checkpoint_from_path(self, checkpoint_path):
        try:
            ckpt_state_dict = torch.load(checkpoint_path, weights_only=False)[
                "model_state_dict"
            ]
        except:
            ckpt_state_dict = torch.load(checkpoint_path, weights_only=False)[
                "model_state"
            ]
        ckpt_state_dict = {
            k: v
            for k, v in ckpt_state_dict.items()
            if "modality_encoder.audio_encoder." not in k
            and "modality_encoder.text_pre_encoder_body." not in k
        }
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _load_data(self, dict_data):
        speaker_pose = dict_data['speaker']['pose'].to(self.rank)
        speaker_trans_v = dict_data['speaker']['trans_v'].to(self.rank)
        listener_pose = dict_data['listener']['pose'].to(self.rank)
        listener_trans_v = dict_data['listener']['trans_v'].to(self.rank)

        speaker_latents = self._pose_to_latent(speaker_pose, speaker_trans_v)
        listener_latents = self._pose_to_latent(listener_pose, listener_trans_v)

        word = dict_data['speaker'].get('word', None)
        if word is not None:
            word = word.to(self.rank)

        audio_onset = None
        if self.cfg.data.onset_rep:
            audio_onset = dict_data['speaker']['audio'].to(self.rank)

        tar_id_raw = dict_data['speaker']['id']
        if isinstance(tar_id_raw, torch.Tensor):
            tar_id = tar_id_raw.to(self.rank)
        else:
            tar_id = torch.tensor([tar_id_raw], device=self.rank)
        audio_name = dict_data['speaker'].get('audio_name', None)

        return {
            "audio_onset": audio_onset,
            "word": word,
            "speaker_latents": speaker_latents,
            "listener_latents": listener_latents,
            "tar_id": tar_id,
            "audio_name": audio_name,
        }

    def _pose_to_latent(self, pose_data, trans_v):
        bs, n = pose_data.shape[0], pose_data.shape[1]

        tar_pose_hands = pose_data[:, :, 25 * 3 : 55 * 3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)

        tar_pose_upper = pose_data[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)

        tar_pose_leg = pose_data[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)

        tar_pose_lower = tar_pose_leg

        tar_pose_upper = (tar_pose_upper - self.mean_upper) / (self.std_upper + 1e-8)
        tar_pose_hands = (tar_pose_hands - self.mean_hands) / (self.std_hands + 1e-8)
        tar_pose_lower = (tar_pose_lower - self.mean_lower) / (self.std_lower + 1e-8)

        use_trans = self.cfg.get('use_trans', True)
        if use_trans:
            tar_pose_lower = torch.cat([tar_pose_lower, trans_v], dim=-1)
        else:
            zero_trans = torch.zeros(bs, n, 3, device=pose_data.device, dtype=pose_data.dtype)
            tar_pose_lower = torch.cat([tar_pose_lower, zero_trans], dim=-1)

        with torch.no_grad():
            latent_upper = self.vq_model_upper.map2latent(tar_pose_upper)
            latent_hands = self.vq_model_hands.map2latent(tar_pose_hands)
            latent_lower = self.vq_model_lower.map2latent(tar_pose_lower)

        latent_in = torch.cat([latent_upper, latent_hands, latent_lower], dim=2) / 5
        return latent_in

    def _latent_to_pose(self, latents):
        latent_in = latents * 5
        code_dim = self.vq_model_upper.code_dim

        latent_upper = latent_in[..., :code_dim]
        latent_hands = latent_in[..., code_dim:code_dim*2]
        latent_lower = latent_in[..., code_dim*2:code_dim*3]

        rec_upper = self.vq_model_upper.latent2origin(latent_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(latent_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(latent_lower)[0]

        rec_lower = rec_lower[..., :-3]

        rec_upper = rec_upper * self.std_upper + self.mean_upper
        rec_hands = rec_hands * self.std_hands + self.mean_hands
        rec_lower = rec_lower * self.std_lower + self.mean_lower

        bs, n = rec_upper.shape[0], rec_upper.shape[1]
        j = 55

        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)

        rec_pose_lower = rec_lower.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)

        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)

        rec_pose_upper_recover = self._inverse_selection_tensor(
            rec_pose_upper, self.joint_mask_upper, bs * n
        )
        rec_pose_lower_recover = self._inverse_selection_tensor(
            rec_pose_lower, self.joint_mask_lower, bs * n
        )
        rec_pose_hands_recover = self._inverse_selection_tensor(
            rec_pose_hands, self.joint_mask_hands, bs * n
        )

        rec_pose = (
            rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        )

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)

        return rec_pose

    def _inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.rank)
        original_shape_t = torch.zeros((n, 165)).to(self.rank)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _convert_6d_to_axis_angle(self, pose_6d):
        bs, n, dim = pose_6d.shape
        j = dim // 6
        
        pose = pose_6d[0].reshape(n, j, 6)
        pose = rc.rotation_6d_to_matrix(pose)
        pose = rc.matrix_to_axis_angle(pose).reshape(n, j * 3)
        
        return pose.detach().cpu().numpy()
    
    def _load_gt_npz(self, file_id):
        npz_path = os.path.join(
            self.cfg.data.data_path, 
            self.cfg.data.pose_rep, 
            f"{file_id}.npz"
        )
        try:
            return np.load(npz_path, allow_pickle=True)
        except (FileNotFoundError, IOError):
            return {"betas": np.zeros(300, dtype=np.float32)}
    
    def _save_test_results(self, rec_pose, tar_pose, loaded_data, results_save_path, sample_idx):
        bs, n_pose, _ = rec_pose.shape
        
        rec_pose_axis = self._convert_6d_to_axis_angle(rec_pose)
        tar_pose_axis = self._convert_6d_to_axis_angle(tar_pose)
        
        audio_name = loaded_data.get("audio_name", None)
        if audio_name is not None:
            base_name = os.path.basename(audio_name).replace('.wav', '')
            file_id = f"{base_name}_seg{sample_idx}"
        else:
            file_id = f"sample_{sample_idx}"
        
        gt_npz = self._load_gt_npz(file_id)
        betas = gt_npz.get("betas", np.zeros(300, dtype=np.float32))
        if isinstance(betas, np.ndarray) and betas.ndim > 1:
            betas = betas.flatten()[:300]
        if len(betas) < 300:
            betas = np.pad(betas, (0, 300 - len(betas)), mode='constant')
        
        betas_expanded = np.tile(betas, (n_pose, 1))
        
        expressions = np.zeros((n_pose, 100), dtype=np.float32)
        trans = np.zeros((n_pose, 3), dtype=np.float32)
        
        np.savez(
            os.path.join(results_save_path, f"gt_{file_id}.npz"),
            betas=betas_expanded.astype(np.float32),
            poses=tar_pose_axis.astype(np.float32),
            expressions=expressions,
            trans=trans,
            model="smplx2020",
            gender="neutral",
            mocap_frame_rate=30,
        )
        
        np.savez(
            os.path.join(results_save_path, f"res_{file_id}.npz"),
            betas=betas_expanded.astype(np.float32),
            poses=rec_pose_axis.astype(np.float32),
            expressions=expressions,
            trans=trans,
            model="smplx2020",
            gender="neutral",
            mocap_frame_rate=30,
        )

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g):
        metrics = {}

        for name, states in self.tracker.loss_meters.items():
            metric_obj = states['train']
            if metric_obj.count > 0:
                value = metric_obj.avg
                metrics[name] = value
                metrics[f"train/{name}"] = value

        metrics.update({
            "train/learning_rate": lr_g,
            "train/data_time_ms": t_data * 1000,
            "train/train_time_ms": t_train * 1000,
        })

        if self.rank == 0:
            wandb.log(metrics, step=epoch * self.train_length + its)

        pstr = f"[{epoch:03d}][{its:03d}/{self.train_length:03d}]  "
        pstr += " ".join([f"{k}: {v:.3f}" for k, v in metrics.items() if "train/" not in k])
        logger.info(pstr)

    def val_recording(self, epoch):
        metrics = {}

        for name, states in self.tracker.loss_meters.items():
            metric_obj = states['val']
            if metric_obj.count > 0:
                value = float(metric_obj.avg) if metric_obj.count > 0 else float(metric_obj.sum)
                metrics[f"val/{name}"] = value

                if name in self.val_best:
                    current_best = self.val_best[name]["value"]
                    if name in ["fgd"]:
                        is_better = value < current_best
                    elif name in ["l1div"]:
                        is_better = value > current_best
                    else:
                        is_better = value < current_best

                    if is_better:
                        self.val_best[name] = {
                            "value": float(value),
                            "epoch": int(epoch)
                        }
                        self.save_checkpoint(
                            epoch=epoch,
                            iteration=epoch * len(self.train_loader),
                            is_best=True,
                            best_metric_name=name
                        )

                    metrics[f"best_{name}"] = float(self.val_best[name]["value"])
                    metrics[f"best_{name}_epoch"] = int(self.val_best[name]["epoch"])

        self.save_checkpoint(
            epoch=epoch,
            iteration=epoch * len(self.train_loader),
            is_best=False,
            best_metric_name=None
        )

        if self.rank == 0:
            try:
                wandb.log(metrics, step=epoch * len(self.train_loader))
            except:
                logger.info("WANDB not initialized!")

        pstr = "Validation Results >>>> "
        pstr += " ".join([
            f"{k.split('/')[-1]}: {v:.3f}"
            for k, v in metrics.items()
            if k.startswith("val/")
        ])
        logger.info(pstr)

        pstr = "Best Results >>>> "
        pstr += " ".join([
            f"{k}: {v['value']:.3f} (epoch {v['epoch']})"
            for k, v in self.val_best.items()
        ])
        logger.info(pstr)

    def save_checkpoint(self, epoch, iteration, is_best=False, best_metric_name=None):
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.opt_s.state_dict() if self.opt_s else None,
            'val_best': self.val_best,
        }

        if epoch % 20 == 0:
            checkpoint_path = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_path, "ckpt.pth"))

        if is_best and best_metric_name:
            best_path = os.path.join(self.checkpoint_path, f"best_{best_metric_name}")
            os.makedirs(best_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(best_path, "ckpt.pth"))

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()

        for its, batch_data in enumerate(self.train_loader):
            if batch_data is None:
                continue

            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start

            self.opt.zero_grad()

            condition_dict = {
                "y": {
                    "audio_onset": loaded_data["audio_onset"],
                    "word": loaded_data["word"],
                    "id": loaded_data["tar_id"],
                    "seed": loaded_data["speaker_latents"][:, : self.cfg.pre_frames],
                }
            }

            if self.cfg.ddp:
                losses = self.model.module.train_forward(
                    condition_dict,
                    loaded_data["speaker_latents"],
                    loaded_data["listener_latents"]
                )
            else:
                losses = self.model.train_forward(
                    condition_dict,
                    loaded_data["speaker_latents"],
                    loaded_data["listener_latents"]
                )

            g_loss_final = losses["loss"]
            self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

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

    def val(self, epoch):
        self.tracker.reset()

        if self.l1_calculator is not None:
            self.l1_calculator.reset()

        start_time = time.time()
        total_length = 0
        latent_out_list = []
        latent_ori_list = []

        self.model.eval()
        self.eval_copy.eval()

        max_val_samples = getattr(self.cfg, 'max_val_samples', None)
        val_count = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                if batch_data is None:
                    continue

                if max_val_samples is not None and val_count >= max_val_samples:
                    break
                val_count += 1

                loaded_data = self._load_data(batch_data)

                condition_dict = {
                    "y": {
                        "audio_onset": loaded_data["audio_onset"],
                        "word": loaded_data["word"],
                        "id": loaded_data["tar_id"],
                        "seed": loaded_data["speaker_latents"][:, : self.cfg.pre_frames],
                    }
                }

                if self.cfg.ddp:
                    val_target_length = getattr(self.cfg.model.modality_encoder.params, 'val_target_length', 
                                                self.cfg.model.modality_encoder.params.get('val_target_length', None))
                    audio_features = self.model.module.modality_encoder(
                        loaded_data["audio_onset"],
                        loaded_data["word"],
                        target_length=val_target_length
                    )
                    listener_loader = OnlineListenerLoader(
                        loaded_data["listener_latents"],
                        context_size=self.model.module.context_size
                    )
                    if self.inference_mode == "rflav":
                        generated_latents = self.model.module.generate_online_rflav(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                    else:
                        generated_latents = self.model.module.generate_online_rdla(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                else:
                    val_target_length = getattr(self.cfg.model.modality_encoder.params, 'val_target_length',
                                                self.cfg.model.modality_encoder.params.get('val_target_length', None))
                    audio_features = self.model.modality_encoder(
                        loaded_data["audio_onset"],
                        loaded_data["word"],
                        target_length=val_target_length
                    )
                    listener_loader = OnlineListenerLoader(
                        loaded_data["listener_latents"],
                        context_size=self.model.context_size
                    )
                    if self.inference_mode == "rflav":
                        generated_latents = self.model.generate_online_rflav(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                    else:
                        generated_latents = self.model.generate_online_rdla(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )

                bs, n_latent = loaded_data["speaker_latents"].shape[0], loaded_data["speaker_latents"].shape[1]
                pre_frames = self.cfg.pre_frames
                
                rec_pose = self._latent_to_pose(generated_latents)
                tar_pose = self._latent_to_pose(loaded_data["speaker_latents"])
                
                n_pose = rec_pose.shape[1]
                pre_frames_pose = pre_frames * self.cfg.vqvae_squeeze_scale
                total_length += (n_pose - pre_frames_pose)
                
                if val_count <= 3:
                    logger.info(f"[DEBUG] generated_latents shape: {generated_latents.shape}, n_latent: {n_latent}, n_pose: {n_pose}")
                    logger.info(f"[DEBUG] rec_pose shape: {rec_pose.shape}, tar_pose shape: {tar_pose.shape}")
                    logger.info(f"[DEBUG] pre_frames_pose: {pre_frames_pose}")

                vae_test_len = self.cfg.vae_test_len
                n_gen = n_pose - pre_frames_pose
                remain = n_gen % vae_test_len
                if val_count <= 3:
                    logger.info(f"[DEBUG] n_gen: {n_gen}, remain: {remain}, n_gen - remain: {n_gen - remain}")
                if n_gen - remain > 0:
                    latent_rec = self.eval_copy.map2latent(rec_pose[:, pre_frames_pose : pre_frames_pose + n_gen - remain])
                    latent_tar = self.eval_copy.map2latent(tar_pose[:, pre_frames_pose : pre_frames_pose + n_gen - remain])

                    latent_out_list.append(
                        latent_rec.reshape(-1, vae_test_len).detach().cpu().numpy()
                    )
                    latent_ori_list.append(
                        latent_tar.reshape(-1, vae_test_len).detach().cpu().numpy()
                    )

                if self.l1_calculator is not None:
                    joints_rec = rec_pose.reshape(bs, n_pose, -1).detach().cpu().numpy()
                    _ = self.l1_calculator.run(joints_rec[0, :n_pose, : 55 * 3])

        if len(latent_out_list) > 0:
            latent_out_all = np.concatenate(latent_out_list, axis=0)
            latent_ori_all = np.concatenate(latent_ori_list, axis=0)
            fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        else:
            fgd = 0.0

        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)

        if self.l1_calculator is not None and self.l1_calculator.counter > 0:
            l1div = self.l1_calculator.avg()
        else:
            l1div = 0.0
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)

        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def test(self, epoch, save_results=True):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)

        if self.l1_calculator is not None:
            self.l1_calculator.reset()

        start_time = time.time()
        total_length = 0
        latent_out_list = []
        latent_ori_list = []

        self.model.eval()
        self.eval_copy.eval()

        max_val_samples = getattr(self.cfg, 'max_val_samples', None)
        test_count = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Testing"):
                if batch_data is None:
                    continue

                if max_val_samples is not None and test_count >= max_val_samples:
                    break
                test_count += 1

                loaded_data = self._load_data(batch_data)

                condition_dict = {
                    "y": {
                        "audio_onset": loaded_data["audio_onset"],
                        "word": loaded_data["word"],
                        "id": loaded_data["tar_id"],
                        "seed": loaded_data["speaker_latents"][:, : self.cfg.pre_frames],
                    }
                }

                if self.cfg.ddp:
                    val_target_length = getattr(self.cfg.model.modality_encoder.params, 'val_target_length', 
                                                self.cfg.model.modality_encoder.params.get('val_target_length', None))
                    audio_features = self.model.module.modality_encoder(
                        loaded_data["audio_onset"],
                        loaded_data["word"],
                        target_length=val_target_length
                    )
                    listener_loader = OnlineListenerLoader(
                        loaded_data["listener_latents"],
                        context_size=self.model.module.context_size
                    )
                    if self.inference_mode == "rflav":
                        generated_latents = self.model.module.generate_online_rflav(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                    else:
                        generated_latents = self.model.module.generate_online_rdla(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                else:
                    val_target_length = getattr(self.cfg.model.modality_encoder.params, 'val_target_length',
                                                self.cfg.model.modality_encoder.params.get('val_target_length', None))
                    audio_features = self.model.modality_encoder(
                        loaded_data["audio_onset"],
                        loaded_data["word"],
                        target_length=val_target_length
                    )
                    listener_loader = OnlineListenerLoader(
                        loaded_data["listener_latents"],
                        context_size=self.model.context_size
                    )
                    if self.inference_mode == "rflav":
                        generated_latents = self.model.generate_online_rflav(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )
                    else:
                        generated_latents = self.model.generate_online_rdla(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            listener_loader=listener_loader,
                            guidance_scale=self.cfg.model.guidance_scale
                        )

                bs, n_latent = loaded_data["speaker_latents"].shape[0], loaded_data["speaker_latents"].shape[1]
                pre_frames = self.cfg.pre_frames
                
                rec_pose = self._latent_to_pose(generated_latents)
                tar_pose = self._latent_to_pose(loaded_data["speaker_latents"])
                
                n_pose = rec_pose.shape[1]
                pre_frames_pose = pre_frames * self.cfg.vqvae_squeeze_scale
                total_length += (n_pose - pre_frames_pose)

                vae_test_len = self.cfg.vae_test_len
                n_gen = n_pose - pre_frames_pose
                remain = n_gen % vae_test_len
                if n_gen - remain > 0:
                    latent_rec = self.eval_copy.map2latent(rec_pose[:, pre_frames_pose : pre_frames_pose + n_gen - remain])
                    latent_tar = self.eval_copy.map2latent(tar_pose[:, pre_frames_pose : pre_frames_pose + n_gen - remain])

                    latent_out_list.append(
                        latent_rec.reshape(-1, vae_test_len).detach().cpu().numpy()
                    )
                    latent_ori_list.append(
                        latent_tar.reshape(-1, vae_test_len).detach().cpu().numpy()
                    )

                if self.l1_calculator is not None:
                    joints_rec = rec_pose.reshape(bs, n_pose, -1).detach().cpu().numpy()
                    _ = self.l1_calculator.run(joints_rec[0, :n_pose, : 55 * 3])
                
                if save_results:
                    self._save_test_results(
                        rec_pose, tar_pose, loaded_data, 
                        results_save_path, test_count
                    )

        if len(latent_out_list) > 0:
            latent_out_all = np.concatenate(latent_out_list, axis=0)
            latent_ori_all = np.concatenate(latent_ori_list, axis=0)
            fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        else:
            fgd = 0.0

        logger.info(f"fgd score: {fgd}")

        if self.l1_calculator is not None and self.l1_calculator.counter > 0:
            l1div = self.l1_calculator.avg()
        else:
            l1div = 0.0
        logger.info(f"l1div score: {l1div}")

        end_time = time.time() - start_time
        logger.info(
            f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion"
        )

    def load_checkpoint(self, checkpoint):
        try:
            ckpt_state_dict = checkpoint["model_state_dict"]
        except:
            ckpt_state_dict = checkpoint["model_state"]
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
