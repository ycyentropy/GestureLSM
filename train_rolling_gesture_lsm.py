import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from loguru import logger
from dataloaders.seamless_interaction_dataset import CustomDataset
from dataloaders import data_tools
from models.rolling_gesture_lsm import RollingGestureLSM
from models.vq.model import RVQVAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle, rotation_6d_to_matrix
from dataloaders.data_tools import joints_list
from utils.joints import upper_body_mask, lower_body_mask, hands_body_mask
from utils import metric
from tqdm import tqdm
import time

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/rolling_gesture_lsm.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()

def create_vqvae_model(cfg, dim_pose, body_part):
    """Create a single RVQVAE model with specified configuration."""
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

def pose_to_latent(pose_data, trans_v, vq_upper, vq_hands, vq_lower, 
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower):
    """Convert pose data to latent representation."""
    bs, n = pose_data.shape[0], pose_data.shape[1]
    
    # 手部：使用固定索引（关节25-54）
    tar_pose_hands = pose_data[:, :, 25 * 3 : 55 * 3]
    tar_pose_hands = axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
    tar_pose_hands = matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)
    
    # 上肢：使用关节掩码
    tar_pose_upper = pose_data[:, :, joint_mask_upper.astype(bool)]
    tar_pose_upper = axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
    tar_pose_upper = matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)
    
    # 下肢：使用关节掩码
    tar_pose_leg = pose_data[:, :, joint_mask_lower.astype(bool)]
    tar_pose_leg = axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
    tar_pose_leg = matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)
    
    tar_pose_lower = tar_pose_leg
    
    tar_pose_upper = (tar_pose_upper - mean_upper) / std_upper
    tar_pose_hands = (tar_pose_hands - mean_hands) / std_hands
    tar_pose_lower = (tar_pose_lower - mean_lower) / std_lower
    
    # 拼接平移速度到下肢（在归一化后拼接，与generative_trainer.py一致）
    tar_pose_lower = torch.cat([tar_pose_lower, trans_v], dim=-1)
    
    with torch.no_grad():
        latent_upper = vq_upper.map2latent(tar_pose_upper)
        latent_hands = vq_hands.map2latent(tar_pose_hands)
        latent_lower = vq_lower.map2latent(tar_pose_lower)
    
    latent_in = torch.cat([latent_upper, latent_hands, latent_lower], dim=2) / 5
    return latent_in

def latent_to_pose(latents, vq_upper, vq_hands, vq_lower,
                   mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                   joint_mask_upper, joint_mask_lower, joint_mask_hands):
    """从 latent 解码回姿态 (rotation_6d 格式，用于 eval_copy)
    输出: [bs, n, 330] (55 joints * 6d)
    """
    latent_in = latents * 5  # 反归一化
    code_dim = vq_upper.code_dim
    
    latent_upper = latent_in[..., :code_dim]
    latent_hands = latent_in[..., code_dim:code_dim*2]
    latent_lower = latent_in[..., code_dim*2:code_dim*3]
    
    # 解码
    rec_upper = vq_upper.latent2origin(latent_upper)[0]
    rec_hands = vq_hands.latent2origin(latent_hands)[0]
    rec_lower = vq_lower.latent2origin(latent_lower)[0]
    
    # 分离平移速度和姿态
    rec_lower = rec_lower[..., :-3]
    
    # 反归一化
    rec_upper = rec_upper * std_upper + mean_upper
    rec_hands = rec_hands * std_hands + mean_hands
    rec_lower = rec_lower * std_lower + mean_lower
    
    bs, n = rec_upper.shape[0], rec_upper.shape[1]
    
    # 转换 6d -> matrix -> axis_angle，然后映射到完整空间
    # upper: 13 joints, lower: 9 joints, hands: 30 joints
    rec_upper_aa = rotation_6d_to_matrix(rec_upper.reshape(bs * n, 13, 6))
    rec_upper_aa = matrix_to_axis_angle(rec_upper_aa).reshape(bs * n, 39)
    rec_pose_upper_recover = inverse_selection(rec_upper_aa, joint_mask_upper, bs * n)
    
    rec_lower_aa = rotation_6d_to_matrix(rec_lower.reshape(bs * n, 9, 6))
    rec_lower_aa = matrix_to_axis_angle(rec_lower_aa).reshape(bs * n, 27)
    rec_pose_lower_recover = inverse_selection(rec_lower_aa, joint_mask_lower, bs * n)
    
    rec_hands_aa = rotation_6d_to_matrix(rec_hands.reshape(bs * n, 30, 6))
    rec_hands_aa = matrix_to_axis_angle(rec_hands_aa).reshape(bs * n, 90)
    rec_pose_hands_recover = inverse_selection(rec_hands_aa, joint_mask_hands, bs * n)
    
    # 合并轴角格式
    rec_pose_aa = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    rec_pose_aa = rec_pose_aa.reshape(bs * n, 55, 3)
    
    # 转换为 6d 格式 (55 * 6 = 330)
    rec_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(rec_pose_aa))
    rec_pose_6d = rec_pose_6d.reshape(bs, n, 330)
    
    return rec_pose_6d

def inverse_selection(data, joint_mask, batch_size):
    """将部分关节数据恢复到完整关节空间"""
    full_dim = len(joint_mask)
    recovered = torch.zeros(batch_size, full_dim, device=data.device, dtype=data.dtype)
    joint_indices = torch.where(torch.tensor(joint_mask) == 1)[0]
    for i, idx in enumerate(joint_indices):
        if i < data.shape[1]:
            recovered[:, idx] = data[:, i]
    return recovered

def main():
    args = get_args_parser()
    
    cfg = OmegaConf.load(args.config)
    
    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = CustomDataset(cfg, "train", build_cache=True)
    test_dataset = CustomDataset(cfg, "val", build_cache=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # 改为0，避免多进程LMDB连接问题
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    model = RollingGestureLSM(cfg).cuda()
    logger.info(model)
    
    # 加载 VQ-VAE 模型
    vq_upper = create_vqvae_model(cfg, 78, "upper").cuda()
    vq_hands = create_vqvae_model(cfg, 180, "hands").cuda()
    vq_lower = create_vqvae_model(cfg, 57, "lower").cuda()
    
    # 设置 VQ 模型为 eval 模式
    vq_upper.eval()
    vq_hands.eval()
    vq_lower.eval()
    
    # 加载 eval_copy 模型用于 FGD 计算
    logger.info("加载 eval_copy 模型用于 FGD 计算...")
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
    logger.info("eval_copy 模型加载完成")
    
    vae_test_len = 32 #240  # 与 eval_copy 的 vae_length 一致
    
    # 加载归一化参数
    mean_pose_path = getattr(cfg, 'mean_pose_path', './mean_std/beatx_2_330_mean.npy')
    std_pose_path = getattr(cfg, 'std_pose_path', './mean_std/beatx_2_330_std.npy')
    mean_pose = np.load(mean_pose_path)
    std_pose = np.load(std_pose_path)
    
    # 创建关节掩码（针对轴角表示，3D）
    ori_joint_list = joints_list["beat_smplx_joints"]
    tar_joint_list_upper = joints_list["beat_smplx_upper"]
    tar_joint_list_lower = joints_list["beat_smplx_lower"]
    
    joint_mask_upper = np.zeros(len(list(ori_joint_list.keys())) * 3)
    for joint_name in tar_joint_list_upper:
        joint_mask_upper[
            ori_joint_list[joint_name][1]
            - ori_joint_list[joint_name][0] : ori_joint_list[joint_name][1]
        ] = 1
    
    joint_mask_lower = np.zeros(len(list(ori_joint_list.keys())) * 3)
    for joint_name in tar_joint_list_lower:
        joint_mask_lower[
            ori_joint_list[joint_name][1]
            - ori_joint_list[joint_name][0] : ori_joint_list[joint_name][1]
        ] = 1
    
    # 使用关节掩码提取归一化参数（针对6D旋转表示）
    mean_upper = torch.from_numpy(mean_pose[upper_body_mask]).cuda()
    std_upper = torch.from_numpy(std_pose[upper_body_mask]).cuda()
    mean_hands = torch.from_numpy(mean_pose[hands_body_mask]).cuda()
    std_hands = torch.from_numpy(std_pose[hands_body_mask]).cuda()
    mean_lower = torch.from_numpy(mean_pose[lower_body_mask]).cuda()
    std_lower = torch.from_numpy(std_pose[lower_body_mask]).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.solver.lr_base)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30000, 40000], gamma=0.1
    )
    
    writer = SummaryWriter(save_dir)
    
    if args.mode == "train":
        logger.info("开始训练...")
        
        logger.info("获取第一个批次进行测试...")
        batch_data = next(iter(train_loader))
        logger.info(f"第一个批次获取成功，pose shape: {batch_data['pose'].shape}")
        
        # 快速测试一个批次
        logger.info("测试完整的前向传播...")
        pose_data = batch_data['pose'].cuda()
        trans_v = batch_data['trans_v'].cuda()
        logger.info("数据已移到 GPU")
        
        latents = pose_to_latent(
            pose_data, trans_v, vq_upper, vq_hands, vq_lower,
            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
            joint_mask_upper, joint_mask_lower
        )
        logger.info(f"pose_to_latent 完成，latents shape: {latents.shape}")
        
        condition_dict = {
            "y": {
                "audio_onset": batch_data["audio_onset"].cuda(),
                "word": batch_data["word"].cuda(),
                "id": batch_data["id"].cuda(),
                "seed": latents[:, :cfg.pre_frames],
                "style_feature": batch_data.get("style_feature", None)
            }
        }
        logger.info("调用 model.train_forward...")
        losses = model.train_forward(condition_dict, latents)
        logger.info(f"train_forward 完成，loss: {losses['loss'].item()}")
        
        logger.info("单批次测试通过！现在开始训练循环...")
        
        # 训练开始前进行一次初始评估 (Epoch -1)
        logger.info("开始初始验证评估 (Epoch -1)...")
        model.eval()
        
        # 初始化评估指标计算器
        l1_calculator = metric.L1div()
        l1_calculator_gt = metric.L1div()
        latent_out_list = []
        latent_ori_list = []
        total_length = 0
        start_time = time.time()
        
        # 创建 hands 关节掩码
        tar_joint_list_hands = joints_list["beat_smplx_hands"]
        joint_mask_hands = np.zeros(len(list(joints_list["beat_smplx_joints"].keys())) * 3)
        for joint_name in tar_joint_list_hands:
            joint_mask_hands[
                joints_list["beat_smplx_joints"][joint_name][1]
                - joints_list["beat_smplx_joints"][joint_name][0] : joints_list["beat_smplx_joints"][joint_name][1]
            ] = 1
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Initial Validation"):
                pose_data = batch_data['pose'].cuda()
                trans_v = batch_data['trans_v'].cuda()
                
                latents = pose_to_latent(
                    pose_data, trans_v, vq_upper, vq_hands, vq_lower,
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower
                )
                
                condition_dict = {
                    "y": {
                        "audio_onset": batch_data["audio_onset"].cuda(),
                        "word": batch_data["word"].cuda(),
                        "id": batch_data["id"].cuda(),
                        "seed": latents[:, :cfg.pre_frames],
                        "style_feature": batch_data.get("style_feature", None)
                    }
                }
                
                # 生成姿态
                audio_features = model.modality_encoder(
                    batch_data["audio_onset"].cuda(),
                    batch_data["word"].cuda()
                )
                
                generated_latents = model.generate(
                    condition_dict=condition_dict,
                    audio_features=audio_features,
                    num_steps=cfg.model.n_steps,
                    guidance_scale=cfg.model.guidance_scale
                )
                
                bs, n = pose_data.shape[0], pose_data.shape[1]
                total_length += n
                
                # 从 latent 解码回姿态 (rotation_6d 格式)
                rec_pose = latent_to_pose(
                    generated_latents, vq_upper, vq_hands, vq_lower,
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower, joint_mask_hands
                )
                tar_pose = latent_to_pose(
                    latents, vq_upper, vq_hands, vq_lower,
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower, joint_mask_hands
                )
                
                # 使用 eval_copy 计算 latent 用于 FGD
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
        
        # 计算 FGD
        if len(latent_out_list) > 0:
            latent_out_all = np.concatenate(latent_out_list, axis=0)
            latent_ori_all = np.concatenate(latent_ori_list, axis=0)
            fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        else:
            fgd = 0.0
        
        # 计算 L1 Diversity (需要姿态数据，暂时跳过)
        l1div = l1_calculator.avg() if l1_calculator.counter > 0 else 0.0
        l1div_gt = l1_calculator_gt.avg() if l1_calculator_gt.counter > 0 else 0.0
        
        end_time = time.time() - start_time
        logger.info(f"初始验证完成 (Epoch -1) - FGD: {fgd:.6f}")
        logger.info(f"Initial validation time: {int(end_time)}s for {int(total_length/cfg.pose_fps)}s motion")
        
        for epoch in range(cfg.solver.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # 使用 tqdm 显示训练进度
            pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader), ncols=80)
            
            for batch_idx, batch_data in pbar:
                pose_data = batch_data['pose'].cuda()
                trans_v = batch_data['trans_v'].cuda()
                
                # 从当前批次计算 latents
                latents = pose_to_latent(
                    pose_data, trans_v, vq_upper, vq_hands, vq_lower,
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower
                )
                
                condition_dict = {
                    "y": {
                        "audio_onset": batch_data["audio_onset"].cuda(),
                        "word": batch_data["word"].cuda(),
                        "id": batch_data["id"].cuda(),
                        "seed": latents[:, :cfg.pre_frames],
                        "style_feature": batch_data.get("style_feature", None)
                    }
                }
                
                losses = model.train_forward(condition_dict, latents)
                
                optimizer.zero_grad()
                losses['loss'].backward()
                optimizer.step()
                
                batch_loss = losses["loss"].item()
                epoch_loss += batch_loss
                num_batches += 1
                
                # 更新进度条描述
                pbar.set_postfix(loss=f'{batch_loss:.3f}')
            
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} 完成 - Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            writer.add_scalar('Train/LR', current_lr, epoch)
            
            # 保存 checkpoint
            if (epoch + 1) % 100 == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
            
            if (epoch + 1) % cfg.validation.test_period == 0:
                logger.info(f"Epoch {epoch}: 开始验证...")
                model.eval()
                
                # 初始化评估指标计算器
                l1_calculator = metric.L1div()
                l1_calculator_gt = metric.L1div()
                latent_out_list = []
                latent_ori_list = []
                total_length = 0
                start_time = time.time()
                
                with torch.no_grad():
                    for batch_data in tqdm(test_loader, desc="Validating"):
                        pose_data = batch_data['pose'].cuda()
                        trans_v = batch_data['trans_v'].cuda()
                        
                        # 将姿态数据转换为 latent
                        latents = pose_to_latent(
                            pose_data, trans_v, vq_upper, vq_hands, vq_lower,
                            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                            joint_mask_upper, joint_mask_lower
                        )
                        
                        condition_dict = {
                            "y": {
                                "audio_onset": batch_data["audio_onset"].cuda(),
                                "word": batch_data["word"].cuda(),
                                "id": batch_data["id"].cuda(),
                                "seed": latents[:, :cfg.pre_frames],
                                "style_feature": batch_data.get("style_feature", None)
                            }
                        }
                        
                        # 生成姿态
                        audio_features = model.modality_encoder(
                            batch_data["audio_onset"].cuda(),
                            batch_data["word"].cuda()
                        )
                        
                        generated_latents = model.generate(
                            condition_dict=condition_dict,
                            audio_features=audio_features,
                            num_steps=cfg.model.n_steps,
                            guidance_scale=cfg.model.guidance_scale
                        )
                        
                        # 从 latent 解码回姿态
                        # TODO: 添加 latent 到姿态的解码和评估指标计算
                        
                        bs, n = pose_data.shape[0], pose_data.shape[1]
                        total_length += n
                        
                        # 从 latent 解码回姿态 (rotation_6d 格式)
                        rec_pose = latent_to_pose(
                            generated_latents, vq_upper, vq_hands, vq_lower,
                            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                            joint_mask_upper, joint_mask_lower, joint_mask_hands
                        )
                        tar_pose = latent_to_pose(
                            latents, vq_upper, vq_hands, vq_lower,
                            mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                            joint_mask_upper, joint_mask_lower, joint_mask_hands
                        )
                        
                        # 使用 eval_copy 计算 latent 用于 FGD
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
                
                # 计算 FGD
                if len(latent_out_list) > 0:
                    latent_out_all = np.concatenate(latent_out_list, axis=0)
                    latent_ori_all = np.concatenate(latent_ori_list, axis=0)
                    fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
                else:
                    fgd = 0.0
                
                # 计算 L1 Diversity (需要姿态数据，暂时跳过)
                l1div = l1_calculator.avg() if l1_calculator.counter > 0 else 0.0
                l1div_gt = l1_calculator_gt.avg() if l1_calculator_gt.counter > 0 else 0.0
                
                # 记录指标
                writer.add_scalar('Val/FGD', fgd, epoch)
                
                end_time = time.time() - start_time
                logger.info(f"Epoch {epoch}: FGD={fgd:.6f}")
                logger.info(f"Validation time: {int(end_time)}s for {int(total_length/cfg.pose_fps)}s motion")
            
            if (epoch + 1) % 100 == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
    
    elif args.mode == "test":
        logger.info("开始测试...")
        model.eval()
        
        os.makedirs(os.path.join(save_dir, "test_results"), exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                pose_data = batch_data['pose'].cuda()
                trans_v = batch_data['trans_v'].cuda()
                
                # 将姿态数据转换为 latent
                latents = pose_to_latent(
                    pose_data, trans_v, vq_upper, vq_hands, vq_lower,
                    mean_upper, std_upper, mean_hands, std_hands, mean_lower, std_lower,
                    joint_mask_upper, joint_mask_lower
                )
                
                condition_dict = {
                    "y": {
                        "audio_onset": batch_data["audio_onset"].cuda(),
                        "word": batch_data["word"].cuda(),
                        "id": batch_data["id"].cuda(),
                        "seed": latents[:, :cfg.pre_frames],
                        "style_feature": batch_data.get("style_feature", None)
                    }
                }
                
                audio_features = model.modality_encoder(
                    batch_data["audio_onset"].cuda(),
                    batch_data["word"].cuda()
                )
                
                generated_latents = model.generate(
                    condition_dict=condition_dict,
                    audio_features=audio_features,
                    num_steps=cfg.n_steps,
                    guidance_scale=cfg.guidance_scale
                )
                
                output_path = os.path.join(save_dir, "test_results", f"sample_{batch_idx}.pth")
                torch.save(generated_latents, output_path)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1} samples")
    
    logger.info("完成！")

if __name__ == "__main__":
    main()
