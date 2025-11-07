#!/usr/bin/env python3
"""
带进度显示的多GPU训练脚本
在数据加载过程中显示详细的进度信息
"""

import os
import sys
import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lazy_window_dataset_progress import create_progress_dataset
from models.vq.model import RVQVAE
# from utils.config import load_config  # 不需要配置文件，直接使用命令行参数
# from utils.media import Pipe  # 不需要Pipe类

# 忽略一些不重要的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_logging(log_dir, local_rank):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{local_rank}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if local_rank == 0 else logging.NullHandler()
        ]
    )
    return logging.getLogger(f"Training_{local_rank}")


def create_model(args, input_dim, device):
    """创建RVQVAE模型"""
    model = RVQVAE(
        args,
        input_dim,
        args.nb_code,
        args.code_dim,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    )

    model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, total_params, trainable_params


def create_optimizer_and_scheduler(model, args):
    """创建优化器和学习率调度器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_scheduler,
        gamma=args.gamma
    )

    return optimizer, scheduler


def create_datasets(args):
    """创建数据集"""
    print(f"🏗️  创建数据集...")
    print(f"📂 数据路径: {args.data_path}")
    print(f"🎯 训练集分割: {args.split}")
    print(f"📏 窗口大小: {args.window_size}, 步长: {args.window_stride}")
    print(f"🔄 多长度训练比例: {args.multi_length_training}")
    print(f"📊 最大样本数: {args.max_samples}")

    dataset_start = time.time()

    # 训练集
    train_dataset = create_progress_dataset(
        data_path=args.data_path,
        split="train",
        window_size=args.window_size,
        window_stride=args.window_stride,
        multi_length_training=args.multi_length_training,
        load_video=False,
        load_audio=False,
        max_samples=args.max_samples,
        cache_path=args.cache_train,
        show_progress=True,
        progress_interval=500  # 每500个窗口显示一次详细进度
    )

    # 验证集
    val_dataset = create_progress_dataset(
        data_path=args.data_path,
        split="val",
        window_size=args.window_size,
        window_stride=args.window_stride,
        multi_length_training=args.multi_length_training,
        load_video=False,
        load_audio=False,
        max_samples=args.max_samples,
        cache_path=args.cache_val,
        show_progress=True,
        progress_interval=200
    )

    dataset_time = time.time() - dataset_start
    print(f"✅ 数据集创建完成，总耗时: {dataset_time:.2f}秒")
    print(f"📊 训练集窗口数: {len(train_dataset):,}")
    print(f"📊 验证集窗口数: {len(val_dataset):,}")

    return train_dataset, val_dataset


def collate_fn(batch):
    """批次整理函数"""
    max_len = max(item['pose'].shape[0] for item in batch)
    batch_size = len(batch)
    pose_dim = batch[0]['pose'].shape[1]

    poses = torch.zeros(batch_size, max_len, pose_dim)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        pose = item['pose']
        length = pose.shape[0]
        poses[i, :length] = pose
        masks[i, :length] = True

    return {'pose': poses, 'mask': masks}


def train_epoch(model, dataloader, optimizer, device, epoch, logger, writer, args):
    """训练一个epoch"""
    model.train()

    epoch_loss = 0.0
    epoch_motion_loss = 0.0
    epoch_commit_loss = 0.0
    epoch_perplexity = 0.0
    num_batches = 0

    # 创建进度条
    if dist.get_rank() == 0:
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True
        )
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        # 数据移到设备
        gt_motion = batch['pose'].to(device, non_blocking=True)
        batch_mask = batch['mask'].to(device, non_blocking=True)

        # 数据标准化
        mean_pose = args.mean_pose.to(device)
        std_pose = args.std_pose.to(device)
        gt_motion = (gt_motion - mean_pose) / std_pose

        # 前向传播
        optimizer.zero_grad()

        output = model(gt_motion, mask=batch_mask)
        pred_motion = output["x_rec"]
        loss_commit = output["commit_loss"]
        perplexity = output["perplexity"]

        # 计算损失
        if args.recons_loss == "l1_smooth":
            loss_motion = torch.nn.functional.l1_loss(pred_motion, gt_motion, reduction='mean')
        else:
            loss_motion = torch.nn.functional.mse_loss(pred_motion, gt_motion, reduction='mean')

        loss = loss_motion + args.commit * loss_commit

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # 统计损失
        epoch_loss += loss.item()
        epoch_motion_loss += loss_motion.item()
        epoch_commit_loss += loss_commit.item()
        epoch_perplexity += perplexity.item()
        num_batches += 1

        # 更新进度条
        if dist.get_rank() == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Motion': f'{loss_motion.item():.4f}',
                'Commit': f'{loss_commit.item():.4f}',
                'PPL': f'{perplexity.item():.1f}'
            })

            # 记录到tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/TotalLoss', loss.item(), global_step)
            writer.add_scalar('Train/MotionLoss', loss_motion.item(), global_step)
            writer.add_scalar('Train/CommitLoss', loss_commit.item(), global_step)
            writer.add_scalar('Train/Perplexity', perplexity.item(), global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

    # 平均损失
    avg_loss = epoch_loss / num_batches
    avg_motion_loss = epoch_motion_loss / num_batches
    avg_commit_loss = epoch_commit_loss / num_batches
    avg_perplexity = epoch_perplexity / num_batches

    return avg_loss, avg_motion_loss, avg_commit_loss, avg_perplexity


def validate(model, dataloader, device, logger, args):
    """验证模型"""
    model.eval()

    val_loss = 0.0
    val_motion_loss = 0.0
    val_commit_loss = 0.0
    val_perplexity = 0.0
    num_batches = 0

    with torch.no_grad():
        # 创建进度条
        if dist.get_rank() == 0:
            pbar = tqdm(
                dataloader,
                desc="Validation",
                leave=False,
                dynamic_ncols=True
            )
        else:
            pbar = dataloader

        for batch in pbar:
            # 数据移到设备
            gt_motion = batch['pose'].to(device, non_blocking=True)
            batch_mask = batch['mask'].to(device, non_blocking=True)

            # 数据标准化
            mean_pose = args.mean_pose.to(device)
            std_pose = args.std_pose.to(device)
            gt_motion = (gt_motion - mean_pose) / std_pose

            # 前向传播
            output = model(gt_motion, mask=batch_mask)
            pred_motion = output["x_rec"]
            loss_commit = output["commit_loss"]
            perplexity = output["perplexity"]

            # 计算损失
            if args.recons_loss == "l1_smooth":
                loss_motion = torch.nn.functional.l1_loss(pred_motion, gt_motion, reduction='mean')
            else:
                loss_motion = torch.nn.functional.mse_loss(pred_motion, gt_motion, reduction='mean')

            loss = loss_motion + args.commit * loss_commit

            # 统计损失
            val_loss += loss.item()
            val_motion_loss += loss_motion.item()
            val_commit_loss += loss_commit.item()
            val_perplexity += perplexity.item()
            num_batches += 1

            # 更新进度条
            if dist.get_rank() == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Motion': f'{loss_motion.item():.4f}',
                    'PPL': f'{perplexity.item():.1f}'
                })

    # 平均损失
    avg_loss = val_loss / num_batches
    avg_motion_loss = val_motion_loss / num_batches
    avg_commit_loss = val_commit_loss / num_batches
    avg_perplexity = val_perplexity / num_batches

    return avg_loss, avg_motion_loss, avg_commit_loss, avg_perplexity


def train_worker(local_rank, args):
    """训练工作进程"""
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')

    # 设置设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # 设置随机种子
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # 创建日志记录器
    logger = setup_logging(args.out_dir, local_rank)

    if dist.get_rank() == 0:
        logger.info(f"开始训练，参数配置: {vars(args)}")
        logger.info(f"使用 {dist.get_world_size()} 个GPU，每个GPU的batch_size为 {args.batch_size}")
        logger.info(f"总有效batch_size为 {args.batch_size * dist.get_world_size()}")

    # 加载均值和标准差
    args.mean_pose = torch.from_numpy(np.load('mean_std/seamless_smplh_mean.npy')[:args.pose_dim]).float()
    args.std_pose = torch.from_numpy(np.load('mean_std/seamless_smplh_std.npy')[:args.pose_dim]).float()
    args.mean_pose = args.mean_pose.unsqueeze(0).unsqueeze(0)
    args.std_pose = args.std_pose.unsqueeze(0).unsqueeze(0)

    # 创建数据集
    train_dataset, val_dataset = create_datasets(args)

    # 创建数据采样器和加载器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # 创建模型
    if dist.get_rank() == 0:
        logger.info("创建模型...")

    model, total_params, trainable_params = create_model(args, args.pose_dim, device)

    # 包装为分布式模型
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)

    # 创建tensorboard写入器
    if dist.get_rank() == 0:
        writer = SummaryWriter(os.path.join(args.out_dir, 'tensorboard'))
        logger.info(f"模型参数量: {total_params/1e6:.1f}M (可训练: {trainable_params/1e6:.1f}M)")
        logger.info(f"训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")
        logger.info("即将进入训练循环...")

    # 训练循环
    best_val_loss = float('inf')
    start_epoch = 0

    if dist.get_rank() == 0:
        logger.info("开始训练，总共 %d 次迭代", args.total_iter)

    # 训练进度条
    global_step = 0
    epoch_pbar = tqdm(range(start_epoch, args.total_iter),
                     desc="训练进度",
                     disable=dist.get_rank() != 0)

    for epoch in epoch_pbar:
        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)

        # 训练
        train_loss, train_motion_loss, train_commit_loss, train_perplexity = train_epoch(
            model, train_loader, optimizer, device, epoch, logger, writer, args
        )

        # 学习率调度
        scheduler.step()

        # 验证
        if epoch % args.eval_iter == 0 or epoch == args.total_iter - 1:
            val_loss, val_motion_loss, val_commit_loss, val_perplexity = validate(
                model, val_loader, device, logger, args
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if dist.get_rank() == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'args': args
                    }, os.path.join(args.out_dir, 'best_model.pth'))
                    logger.info(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

            # 记录验证损失
            if dist.get_rank() == 0:
                writer.add_scalar('Val/TotalLoss', val_loss, global_step)
                writer.add_scalar('Val/MotionLoss', val_motion_loss, global_step)
                writer.add_scalar('Val/CommitLoss', val_commit_loss, global_step)
                writer.add_scalar('Val/Perplexity', val_perplexity, global_step)

                logger.info(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")

        # 保存定期检查点
        if epoch % 1000 == 0 and dist.get_rank() == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args
            }, os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pth'))

        global_step += len(train_loader)

        # 更新进度条
        if dist.get_rank() == 0:
            epoch_pbar.set_postfix({
                'Train': f'{train_loss:.4f}',
                'Val': f'{val_loss if epoch % args.eval_iter == 0 else "N/A"}',
                'Best': f'{best_val_loss:.4f}'
            })

    # 训练完成
    if dist.get_rank() == 0:
        writer.close()
        logger.info("训练完成!")

    # 清理
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0)

    # 数据参数
    parser.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--window_stride', type=int, default=20)
    parser.add_argument('--multi_length_training', type=float, nargs='+', default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--cache_train', type=str, default=None)
    parser.add_argument('--cache_val', type=str, default=None)
    parser.add_argument('--use_cache', action='store_true')

    # 模型参数
    parser.add_argument('--code_dim', type=int, default=128)
    parser.add_argument('--nb_code', type=int, default=1024)
    parser.add_argument('--num_quantizers', type=int, default=8)
    parser.add_argument('--down_t', type=int, default=2)
    parser.add_argument('--stride_t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation_growth_rate', type=int, default=3)
    parser.add_argument('--vq_act', type=str, default='relu')
    parser.add_argument('--vq_norm', type=str, default=None)
    parser.add_argument('--commit', type=float, default=0.02)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.5)
    parser.add_argument('--recons_loss', type=str, default='l1_smooth')
    parser.add_argument('--loss_vel', type=float, default=0.0)
    parser.add_argument('--shared_codebook', action='store_true')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=int, nargs='+', default=[5000, 8000])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--total_iter', type=int, default=10000)
    parser.add_argument('--eval_iter', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pose_dim', type=int, default=156)

    # 输出参数
    parser.add_argument('--out_dir', type=str, default='experiments/rvq_seamless_progress')
    parser.add_argument('--exp_name', type=str, default='rvq_seamless_progress')
    parser.add_argument('--print_iter', type=int, default=100)

    # 添加对torch.distributed.launch传递的--local-rank参数的支持
    # 注意：torch.distributed.launch使用--local-rank（带连字符），而不是--local_rank（带下划线）
    known_args, unknown_args = parser.parse_known_args()

    # 处理未知参数，特别是--local-rank
    local_rank = 0
    for i, arg in enumerate(unknown_args):
        if arg == '--local-rank' and i + 1 < len(unknown_args):
            local_rank = int(unknown_args[i + 1])
            break
        elif arg.startswith('--local-rank='):
            local_rank = int(arg.split('=')[1])
            break

    # 设置local_rank
    known_args.local_rank = local_rank
    args = known_args

    # 自动启用缓存模式
    if args.cache_train is not None or args.cache_val is not None:
        args.use_cache = True
        print(f"🔧 检测到缓存路径，自动启用缓存模式")
        print(f"📂 缓存路径 - 训练: {args.cache_train}, 验证: {args.cache_val}")

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存配置
    import json
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 启动训练
    print("🚀 启动分布式训练...")
    mp.spawn(train_worker, args=(args,), nprocs=torch.cuda.device_count(), join=True)


if __name__ == '__main__':
    main()