import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import numpy as np

from dataloaders.dyadic_feedback_dataset import DyadicFeedbackDataset, collate_fn, collate_fn_single
from dataloaders import data_tools
from dataloaders.data_tools import joints_list
from models.interactive_gesture_lsm import InteractiveGestureLSM, OnlineListenerLoader
from models.vq.model import RVQVAE
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from utils import metric


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/interactive_gesture_lsm.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def create_vqvae_model(cfg, dim_pose, body_part):
    vq_args = cfg
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
    
    checkpoint_path = getattr(cfg, f"vqvae_{body_part}_path")
    model.load_state_dict(torch.load(checkpoint_path)["net"])
    return model


def load_model(cfg, checkpoint_path=None):
    model = InteractiveGestureLSM(cfg)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)["model"])
    return model


def pose_to_latent(pose_data, trans_v, vq_upper, vq_hands, vq_lower, 
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower, use_trans=True, vqvae_latent_scale=5,
                    train_body_part=None):
    bs, n = pose_data.shape[0], pose_data.shape[1]
    
    tar_pose_hands = pose_data[:, :, 25 * 3 : 55 * 3]
    tar_pose_hands = axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
    tar_pose_hands = matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)
    
    tar_pose_upper = pose_data[:, :, joint_mask_upper.astype(bool)]
    tar_pose_upper = axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
    tar_pose_upper = matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)
    
    tar_pose_leg = pose_data[:, :, joint_mask_lower.astype(bool)]
    tar_pose_leg = axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
    tar_pose_leg = matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)
    
    tar_pose_hands = (tar_pose_hands - mean_hands) / (std_hands + 1e-8)
    tar_pose_upper = (tar_pose_upper - mean_upper) / (std_upper + 1e-8)
    tar_pose_lower = (tar_pose_leg - mean_lower) / (std_lower + 1e-8)
    
    if use_trans:
        tar_pose_lower = torch.cat([tar_pose_lower, trans_v], dim=-1)
    else:
        zero_trans = torch.zeros(bs, n, 3, device=pose_data.device, dtype=pose_data.dtype)
        tar_pose_lower = torch.cat([tar_pose_lower, zero_trans], dim=-1)
    
    with torch.no_grad():
        latent_upper = vq_upper.map2latent(tar_pose_upper)
        latent_hands = vq_hands.map2latent(tar_pose_hands)
        latent_lower = vq_lower.map2latent(tar_pose_lower)
    
    if train_body_part == "upper":
        latent_hands = torch.zeros_like(latent_hands)
        latent_lower = torch.zeros_like(latent_lower)
    elif train_body_part == "hands":
        latent_upper = torch.zeros_like(latent_upper)
        latent_lower = torch.zeros_like(latent_lower)
    elif train_body_part == "lower":
        latent_upper = torch.zeros_like(latent_upper)
        latent_hands = torch.zeros_like(latent_hands)
    
    latent_in = torch.cat([latent_upper, latent_hands, latent_lower], dim=2) / vqvae_latent_scale
    return latent_in


def latent_to_pose(latents, vq_upper, vq_hands, vq_lower,
                   mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                   joint_mask_upper, joint_mask_lower, joint_mask_hands, vqvae_latent_scale=5,
                   gt_latents=None, train_body_part=None):
    from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d
    
    latent_in = latents * vqvae_latent_scale
    code_dim = vq_upper.code_dim
    
    latent_upper = latent_in[..., :code_dim]
    latent_hands = latent_in[..., code_dim:code_dim*2]
    latent_lower = latent_in[..., code_dim*2:code_dim*3]
    
    if gt_latents is not None and train_body_part is not None:
        gt_latent_in = gt_latents * vqvae_latent_scale
        gt_latent_upper = gt_latent_in[..., :code_dim]
        gt_latent_hands = gt_latent_in[..., code_dim:code_dim*2]
        gt_latent_lower = gt_latent_in[..., code_dim*2:code_dim*3]
        
        if train_body_part == "upper":
            latent_hands = gt_latent_hands
            latent_lower = gt_latent_lower
        elif train_body_part == "hands":
            latent_upper = gt_latent_upper
            latent_lower = gt_latent_lower
        elif train_body_part == "lower":
            latent_upper = gt_latent_upper
            latent_hands = gt_latent_hands
    
    rec_upper = vq_upper.latent2origin(latent_upper)[0]
    rec_hands = vq_hands.latent2origin(latent_hands)[0]
    rec_lower = vq_lower.latent2origin(latent_lower)[0]
    
    rec_lower = rec_lower[..., :-3]
    
    rec_upper = rec_upper * std_upper + mean_upper
    rec_hands = rec_hands * std_hands + mean_hands
    rec_lower = rec_lower * std_lower + mean_lower
    
    bs, n = rec_upper.shape[0], rec_upper.shape[1]
    j = 55  # total joints
    
    # Convert 6D to axis-angle and map back to full pose
    rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
    rec_pose_upper = rotation_6d_to_matrix(rec_pose_upper)
    rec_pose_upper = matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
    
    rec_pose_lower = rec_lower.reshape(bs, n, 9, 6)
    rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
    rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
    
    rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
    rec_pose_hands = rotation_6d_to_matrix(rec_pose_hands)
    rec_pose_hands = matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
    
    # Inverse selection to map back to full pose
    rec_pose_upper_recover = inverse_selection(rec_pose_upper, joint_mask_upper, bs * n)
    rec_pose_lower_recover = inverse_selection(rec_pose_lower, joint_mask_lower, bs * n)
    rec_pose_hands_recover = inverse_selection(rec_pose_hands, joint_mask_hands, bs * n)
    
    rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    
    # Convert back to 6D for FGD calculation
    rec_pose = torch.from_numpy(rec_pose).reshape(bs * n, j, 3).float().to(latents.device)
    rec_pose = axis_angle_to_matrix(rec_pose)
    rec_pose = matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
    
    return rec_pose


def inverse_selection(filtered_t, selection_array, n):
    """Map filtered pose back to original joint positions."""
    # Convert to numpy if tensor
    if isinstance(filtered_t, torch.Tensor):
        filtered_t = filtered_t.detach().cpu().numpy()
    
    original_shape_t = np.zeros((n, selection_array.size))
    selected_indices = np.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t


def main():
    args = get_args_parser()
    cfg = OmegaConf.load(args.config)
    
    if args.debug:
        cfg.debug = True
    
    logger.add(os.path.join(cfg.output_dir, 'train.log'), rotation='500 MB')
    logger.info(f"Config: {cfg}")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    
    writer = SummaryWriter(os.path.join(cfg.output_dir, 'logs'))
    
    tar_joint_list_upper = joints_list["beat_smplx_upper"]
    tar_joint_list_lower = joints_list["beat_smplx_lower"]
    tar_joint_list_hands = joints_list["beat_smplx_hands"]
    
    joint_mask_upper = np.zeros(len(list(joints_list["beat_smplx_joints"].keys())) * 3)
    for joint_name in tar_joint_list_upper:
        joint_mask_upper[
            joints_list["beat_smplx_joints"][joint_name][1]
            - joints_list["beat_smplx_joints"][joint_name][0] : joints_list["beat_smplx_joints"][joint_name][1]
        ] = 1
    
    joint_mask_lower = np.zeros(len(list(joints_list["beat_smplx_joints"].keys())) * 3)
    for joint_name in tar_joint_list_lower:
        joint_mask_lower[
            joints_list["beat_smplx_joints"][joint_name][1]
            - joints_list["beat_smplx_joints"][joint_name][0] : joints_list["beat_smplx_joints"][joint_name][1]
        ] = 1
    
    joint_mask_hands = np.zeros(len(list(joints_list["beat_smplx_joints"].keys())) * 3)
    for joint_name in tar_joint_list_hands:
        joint_mask_hands[
            joints_list["beat_smplx_joints"][joint_name][1]
            - joints_list["beat_smplx_joints"][joint_name][0] : joints_list["beat_smplx_joints"][joint_name][1]
        ] = 1
    
    from utils.joints import hands_body_mask, lower_body_mask, upper_body_mask
    
    # Load mean and std from files
    mean_pose_path = getattr(cfg, 'mean_pose_path', './mean_std/beatx_2_330_mean.npy')
    std_pose_path = getattr(cfg, 'std_pose_path', './mean_std/beatx_2_330_std.npy')
    mean = np.load(mean_pose_path)
    std = np.load(std_pose_path)
    
    # Extract body part specific normalizations
    mean_upper = torch.from_numpy(mean[upper_body_mask]).cuda()
    std_upper = torch.from_numpy(std[upper_body_mask]).cuda()
    mean_hands = torch.from_numpy(mean[hands_body_mask]).cuda()
    std_hands = torch.from_numpy(std[hands_body_mask]).cuda()
    mean_lower = torch.from_numpy(mean[lower_body_mask]).cuda()
    std_lower = torch.from_numpy(std[lower_body_mask]).cuda()
    
    logger.info("Creating model...")
    model = InteractiveGestureLSM(cfg)
    model = model.cuda()
    logger.info(model)
    
    logger.info("Loading VQ-VAE models...")
    vq_upper = create_vqvae_model(cfg, 78, "upper").cuda()
    vq_hands = create_vqvae_model(cfg, 180, "hands").cuda()
    vq_lower = create_vqvae_model(cfg, 57, "lower").cuda()
    
    vq_upper.eval()
    vq_hands.eval()
    vq_lower.eval()
    
    logger.info("Loading eval_copy model for FGD...")
    from utils import other_tools
    eval_model_module = __import__("models.motion_representation", fromlist=["something"])
    eval_args = type('Args', (), {})()
    eval_args.vae_layer = 4
    eval_args.vae_length = 240
    eval_args.vae_test_dim = 330
    eval_args.variational = False
    eval_args.data_path_1 = "./datasets/hub/"
    eval_args.vae_grow = [1, 1, 2, 1]
    
    eval_copy = getattr(eval_model_module, 'VAESKConv')(eval_args).cuda()
    other_tools.load_checkpoints(
        eval_copy,
        './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/AESKConv_240_100.bin',
        'VAESKConv'
    )
    eval_copy.eval()
    logger.info("eval_copy model loaded")
    
    vae_test_len = 32
    vqvae_latent_scale = cfg.get('vqvae_latent_scale', 5)
    train_body_part = cfg.get('train_body_part', None)
    if train_body_part:
        logger.info(f"Training only body part: {train_body_part}")
    
    logger.info("Creating datasets...")
    train_dataset = DyadicFeedbackDataset(cfg, split='train', build_cache=True)
    val_dataset = DyadicFeedbackDataset(cfg, split='val', build_cache=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_single,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.solver.lr_base,
        weight_decay=cfg.solver.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.solver.epochs,
        eta_min=cfg.solver.lr_min,
    )
    
    save_dir = cfg.output_dir
    
    logger.info("Testing forward pass...")
    model.train()
    for batch_data in train_loader:
        if batch_data is None:
            continue
        
        # 保持 batch 维度
        speaker_pose = batch_data['speaker']['pose'].cuda()
        speaker_trans_v = batch_data['speaker']['trans_v'].cuda()
        listener_pose = batch_data['listener']['pose'].cuda()
        listener_trans_v = batch_data['listener']['trans_v'].cuda()
        
        speaker_latents = pose_to_latent(
            speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
            joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
            vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
        )
        
        listener_latents = pose_to_latent(
            listener_pose, listener_trans_v, vq_upper, vq_hands, vq_lower,
            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
            joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
            vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
        )
        
        condition_dict = {
            "y": {
                "audio_onset": batch_data['speaker']['audio'].cuda(),
                "word": batch_data['speaker']['word'].cuda(),
                "id": batch_data['speaker']['id'].cuda() if isinstance(batch_data['speaker']['id'], torch.Tensor) else torch.tensor([batch_data['speaker']['id']]).cuda(),
                "seed": speaker_latents[:, :cfg.get('pre_frames', 4)],
            }
        }
        
        losses = model.train_forward(condition_dict, speaker_latents, listener_latents)
        logger.info(f"Test forward pass - Loss: {losses['loss'].item():.4f}")
        break
    
    logger.info("Starting initial validation...")
    model.eval()
    
    l1_calculator = metric.L1div()
    latent_out_list = []
    latent_ori_list = []
    total_length = 0
    start_time = time.time()
    
    # 限制验证数量
    max_val_samples = getattr(cfg, 'max_val_samples', None)
    val_count = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Initial Validation"):
            if batch_data is None:
                continue
            
            # 检查是否达到最大验证数量
            if max_val_samples is not None and val_count >= max_val_samples:
                break
            val_count += 1
            
            speaker_pose = batch_data['speaker']['pose'].cuda()
            speaker_trans_v = batch_data['speaker']['trans_v'].cuda()
            listener_pose = batch_data['listener']['pose'].cuda()
            listener_trans_v = batch_data['listener']['trans_v'].cuda()
            
            speaker_latents_full = pose_to_latent(
                speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                vqvae_latent_scale=vqvae_latent_scale, train_body_part=None
            )
            
            speaker_latents_seed = pose_to_latent(
                speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
            )
            
            listener_latents = pose_to_latent(
                listener_pose, listener_trans_v, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                vqvae_latent_scale=vqvae_latent_scale, train_body_part=None
            )
            
            condition_dict = {
                "y": {
                    "audio_onset": batch_data['speaker']['audio'].cuda(),
                    "word": batch_data['speaker']['word'].cuda(),
                    "id": batch_data['speaker']['id'].cuda() if isinstance(batch_data['speaker']['id'], torch.Tensor) else torch.tensor([batch_data['speaker']['id']]).cuda(),
                    "seed": speaker_latents_seed[:, :cfg.get('pre_frames', 4)],
                }
            }
            
            audio_features = model.modality_encoder(
                batch_data['speaker']['audio'].cuda(),
                batch_data['speaker']['word'].cuda()
            )
            
            # 使用在线交互模式生成（模拟真实在线交互）
            listener_loader = OnlineListenerLoader(
                listener_latents, 
                context_size=model.context_size
            )
            
            generated_latents = model.generate_online(
                condition_dict=condition_dict,
                audio_features=audio_features,
                listener_loader=listener_loader,
                num_steps=cfg.model.n_steps,
                guidance_scale=cfg.model.guidance_scale
            )
            
            bs, n = speaker_pose.shape[0], speaker_pose.shape[1]
            total_length += n
            
            rec_pose = latent_to_pose(
                generated_latents, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, joint_mask_hands,
                vqvae_latent_scale=vqvae_latent_scale,
                gt_latents=speaker_latents_full, train_body_part=train_body_part
            )
            tar_pose = latent_to_pose(
                speaker_latents_full, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, joint_mask_hands,
                vqvae_latent_scale=vqvae_latent_scale
            )
            
            remain = n % vae_test_len
            if n - remain > 0:
                latent_rec = eval_copy.map2latent(rec_pose[:, :n - remain])
                latent_tar = eval_copy.map2latent(tar_pose[:, :n - remain])
                
                latent_out_list.append(
                    latent_rec.reshape(-1, vae_test_len).detach().cpu().numpy()
                )
                latent_ori_list.append(
                    latent_tar.reshape(-1, vae_test_len).detach().cpu().numpy()
                )
    
    if len(latent_out_list) > 0:
        latent_out_all = np.concatenate(latent_out_list, axis=0)
        latent_ori_all = np.concatenate(latent_ori_list, axis=0)
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
    else:
        fgd = 0.0
    
    l1div = l1_calculator.avg() if l1_calculator.counter > 0 else 0.0
    
    end_time = time.time() - start_time
    logger.info(f"Initial validation - FGD: {fgd:.6f}, L1 Div: {l1div:.6f}")
    logger.info(f"Initial validation time: {int(end_time)}s")
    
    # Early stopping 初始化
    early_stopping_enabled = cfg.solver.get("early_stopping", {}).get("enabled", False)
    if early_stopping_enabled:
        patience = cfg.solver.early_stopping.get("patience", 5)
        delta = cfg.solver.early_stopping.get("delta", 0.001)
        best_fgd = float('inf')
        patience_counter = 0
        early_stopped = False
        logger.info(f"Early stopping enabled with patience={patience}, delta={delta}")
    else:
        best_fgd = float('inf')
        patience_counter = 0
        early_stopped = False
    
    for epoch in range(cfg.solver.epochs):
        # 检查是否触发了 early stopping
        if early_stopping_enabled and early_stopped:
            break
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader), ncols=80)
        
        for batch_idx, batch_data in pbar:
            if batch_data is None:
                continue
            
            speaker_pose = batch_data['speaker']['pose'].cuda()
            speaker_trans_v = batch_data['speaker']['trans_v'].cuda()
            listener_pose = batch_data['listener']['pose'].cuda()
            listener_trans_v = batch_data['listener']['trans_v'].cuda()
            
            speaker_latents = pose_to_latent(
                speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
            )
            
            listener_latents = pose_to_latent(
                listener_pose, listener_trans_v, vq_upper, vq_hands, vq_lower,
                mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
            )
            
            condition_dict = {
                "y": {
                    "audio_onset": batch_data['speaker']['audio'].cuda(),
                    "word": batch_data['speaker']['word'].cuda(),
                    "id": batch_data['speaker']['id'].cuda() if isinstance(batch_data['speaker']['id'], torch.Tensor) else torch.tensor([batch_data['speaker']['id']]).cuda(),
                    "seed": speaker_latents[:, :cfg.get('pre_frames', 4)],
                }
            }
            
            losses = model.train_forward(condition_dict, speaker_latents, listener_latents)
            
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
            
            batch_loss = losses["loss"].item()
            epoch_loss += batch_loss
            num_batches += 1
            
            pbar.set_postfix(loss=f'{batch_loss:.3f}')
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} - Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
        
        if (epoch + 1) % cfg.val_period == 0:
            logger.info(f"Epoch {epoch}: Starting validation...")
            model.eval()
            
            l1_calculator = metric.L1div()
            latent_out_list = []
            latent_ori_list = []
            total_length = 0
            start_time = time.time()
            
            # 限制验证数量
            val_count = 0
            
            with torch.no_grad():
                for batch_data in tqdm(val_loader, desc="Validating"):
                    if batch_data is None:
                        continue
                    
                    # 检查是否达到最大验证数量
                    if max_val_samples is not None and val_count >= max_val_samples:
                        break
                    val_count += 1
                    
                    speaker_pose = batch_data['speaker']['pose'].cuda()
                    speaker_trans_v = batch_data['speaker']['trans_v'].cuda()
                    listener_pose = batch_data['listener']['pose'].cuda()
                    listener_trans_v = batch_data['listener']['trans_v'].cuda()
                    
                    speaker_latents_full = pose_to_latent(
                        speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
                        mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                        joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                        vqvae_latent_scale=vqvae_latent_scale, train_body_part=None
                    )
                    
                    speaker_latents_seed = pose_to_latent(
                        speaker_pose, speaker_trans_v, vq_upper, vq_hands, vq_lower,
                        mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                        joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                        vqvae_latent_scale=vqvae_latent_scale, train_body_part=train_body_part
                    )
                    
                    listener_latents = pose_to_latent(
                        listener_pose, listener_trans_v, vq_upper, vq_hands, vq_lower,
                        mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                        joint_mask_upper, joint_mask_lower, use_trans=cfg.get('use_trans', True),
                        vqvae_latent_scale=vqvae_latent_scale, train_body_part=None
                    )
                    
                    condition_dict = {
                        "y": {
                            "audio_onset": batch_data['speaker']['audio'].cuda(),
                            "word": batch_data['speaker']['word'].cuda(),
                            "id": batch_data['speaker']['id'].cuda() if isinstance(batch_data['speaker']['id'], torch.Tensor) else torch.tensor([batch_data['speaker']['id']]).cuda(),
                            "seed": speaker_latents_seed[:, :cfg.get('pre_frames', 4)],
                        }
                    }
                    
                    audio_features = model.modality_encoder(
                        batch_data['speaker']['audio'].cuda(),
                        batch_data['speaker']['word'].cuda()
                    )
                    
                    # 使用在线交互模式生成（模拟真实在线交互）
                    listener_loader = OnlineListenerLoader(
                        listener_latents, 
                        context_size=model.context_size
                    )
                    
                    generated_latents = model.generate_online(
                        condition_dict=condition_dict,
                        audio_features=audio_features,
                        listener_loader=listener_loader,
                        num_steps=cfg.model.n_steps,
                        guidance_scale=cfg.model.guidance_scale
                    )
                    
                    bs, n = speaker_pose.shape[0], speaker_pose.shape[1]
                    total_length += n
                    
                    rec_pose = latent_to_pose(
                        generated_latents, vq_upper, vq_hands, vq_lower,
                        mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                        joint_mask_upper, joint_mask_lower, joint_mask_hands,
                        vqvae_latent_scale=vqvae_latent_scale,
                        gt_latents=speaker_latents_full, train_body_part=train_body_part
                    )
                    tar_pose = latent_to_pose(
                        speaker_latents_full, vq_upper, vq_hands, vq_lower,
                        mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                        joint_mask_upper, joint_mask_lower, joint_mask_hands,
                        vqvae_latent_scale=vqvae_latent_scale
                    )
                    
                    remain = n % vae_test_len
                    if n - remain > 0:
                        latent_rec = eval_copy.map2latent(rec_pose[:, :n - remain])
                        latent_tar = eval_copy.map2latent(tar_pose[:, :n - remain])
                        
                        latent_out_list.append(
                            latent_rec.reshape(-1, vae_test_len).detach().cpu().numpy()
                        )
                        latent_ori_list.append(
                            latent_tar.reshape(-1, vae_test_len).detach().cpu().numpy()
                        )
            
            if len(latent_out_list) > 0:
                latent_out_all = np.concatenate(latent_out_list, axis=0)
                latent_ori_all = np.concatenate(latent_ori_list, axis=0)
                fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
            else:
                fgd = 0.0
            
            l1div = l1_calculator.avg() if l1_calculator.counter > 0 else 0.0
            
            end_time = time.time() - start_time
            logger.info(f"Epoch {epoch} Validation - FGD: {fgd:.6f}, L1 Div: {l1div:.6f}")
            logger.info(f"Validation time: {int(end_time)}s for {int(total_length/cfg.data.pose_fps)}s motion")
            
            writer.add_scalar('Val/FGD', fgd, epoch)
            writer.add_scalar('Val/L1Div', l1div, epoch)
            
            # Early stopping 检查
            if early_stopping_enabled:
                if fgd < best_fgd - delta:
                    # 有改进，保存 checkpoint
                    best_fgd = fgd
                    patience_counter = 0
                    ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_best.pth")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'fgd': fgd,
                    }, ckpt_path)
                    logger.info(f"Best checkpoint saved: {ckpt_path} (FGD improved)")
                else:
                    # 没有改进，增加计数器
                    patience_counter += 1
                    logger.info(f"Early stopping counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        early_stopped = True
                        logger.info(f"Early stopping triggered after epoch {epoch}")
                        break
            else:
                # 没有 early stopping，每次都保存
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_best.pth")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'fgd': fgd,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
    
    logger.info("Training completed!")


def test_model(args, cfg):
    from trainer.interactive_gesture_trainer import InteractiveGestureTrainer
    
    class TestArgs:
        def __init__(self):
            self.checkpoint = args.checkpoint
            self.resume = None
            self.debug = args.debug
            self.mode = "test"
    
    test_args = TestArgs()
    
    trainer = InteractiveGestureTrainer(cfg, test_args)
    
    if args.checkpoint:
        trainer._load_checkpoint_from_path(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    trainer.test(epoch=999, save_results=True)
    logger.info("Testing completed!")


if __name__ == "__main__":
    args = get_args_parser()
    cfg = OmegaConf.load(args.config)
    
    if args.mode == "test":
        if args.checkpoint is None:
            logger.error("--checkpoint is required for test mode")
            sys.exit(1)
        test_model(args, cfg)
    else:
        main()
