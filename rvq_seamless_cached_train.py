#!/usr/bin/env python3
"""
修改后的训练脚本，支持使用缓存的窗口参数
基于rvq_seamless_multi_gpu.py修改
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import glob
import pickle
import sys
import warnings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义数据集
from save_window_params import CachedLazySeamlessInteractionWindowDataset, save_window_params

# 导入模型和其他组件 - 与rvq_seamless_multi_gpu.py一致
from models.vq.model import RVQVAE
from lazy_window_dataset import LazySeamlessInteractionWindowDataset
from dataloaders.data_tools import joints_list

warnings.filterwarnings('ignore')

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
    
    def my_forward(self, motion_pred, motion_gt, mask) :
        loss = self.Loss(motion_pred[..., mask], motion_gt[..., mask])
        return loss


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RVQ Seamless Multi-GPU Training with Cached Dataset')
    
    # 数据集参数
    parser.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction',
                        help='数据集路径')
    parser.add_argument('--window_size', type=int, default=64,
                        help='窗口大小(帧数)')
    parser.add_argument('--window_stride', type=int, default=20,
                        help='窗口步长(帧数)')
    parser.add_argument('--multi_length_training', action='store_true',
                        help='是否使用多长度训练')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数')
    
    # 缓存参数
    parser.add_argument('--use_cache', action='store_true',
                        help='是否使用缓存的窗口参数')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='训练集缓存文件路径')
    parser.add_argument('--val_cache_path', type=str, default=None,
                        help='验证集缓存文件路径')
    parser.add_argument('--save_cache', action='store_true',
                        help='是否保存窗口参数到缓存')
    
    # 模型参数
    parser.add_argument('--code_dim', type=int, default=256,
                        help='编码维度')
    parser.add_argument('--n_codebooks', type=int, default=8,
                        help='码书数量')
    parser.add_argument('--num_quantizers', type=int, default=8,
                        help='量化器数量')
    parser.add_argument('--codebook_size', type=int, default=512,
                        help='码书大小')
    parser.add_argument('--shared_codebook', action='store_true',
                        help='是否共享码书')
    parser.add_argument('--down_t', type=int, default=2,
                        help='下采样层数')
    parser.add_argument('--stride_t', type=int, default=2,
                        help='时间步长')
    parser.add_argument('--depth', type=int, default=3,
                        help='网络深度')
    parser.add_argument('--dilation', type=int, default=3,
                        help='膨胀率')
    parser.add_argument('--vq_act', type=str, default='relu',
                        help='VQ激活函数')
    parser.add_argument('--emb_norm', type=float, default=0.99,
                        help='嵌入归一化')
    parser.add_argument('--width', type=int, default=512,
                        help='网络宽度')
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.0,
                        help='量化dropout概率')
    
    # 训练参数 - 与rvq_seamless_multi_gpu.py一致
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批次大小')
    parser.add_argument('--total_iter', type=int, default=10000,
                        help='总迭代次数')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--commit', type=float, default=0.02,
                        help='commitment loss权重')
    parser.add_argument('--loss_vel', type=float, default=0.0,
                        help='velocity loss权重')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth',
                        help='重建损失类型')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--print_iter', type=int, default=100,
                        help='打印间隔')
    parser.add_argument('--eval_iter', type=int, default=5000,
                        help='评估间隔')
    parser.add_argument('--lr_scheduler', type=list, default=[5000, 8000],
                        help='学习率调度器里程碑')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='学习率衰减因子')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='是否使用混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='是否使用梯度检查点')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 输出参数
    parser.add_argument('--exp_name', type=str, default='rvq_seamless_cached',
                        help='实验名称')
    parser.add_argument('--out_dir', type=str, default='experiments',
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='模型保存间隔')
    
    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0,
                        help='本地排名')
    parser.add_argument('--world_size', type=int, default=1,
                        help='世界大小')
    
    return parser


def get_logger(out_dir, rank):
    """创建日志记录器"""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler(os.path.join(out_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(fh)
    if rank == 0:  # 只有主进程输出到控制台
        logger.addHandler(ch)
    
    return logger


def custom_collate_fn(batch):
    """自定义批处理函数"""
    # 简单的批处理函数，将批次中的样本组合成列表
    # 在实际应用中，可能需要更复杂的处理
    return batch


def setup_distributed():
    """设置分布式训练"""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1
    
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    else:
        rank = 0
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    is_main_process = rank == 0
    return rank, local_rank, world_size, is_main_process


def create_dataset(args, split, is_main_process):
    """
    创建数据集，支持使用缓存 - 更健壮的版本
    
    Args:
        args: 命令行参数
        split: 数据集分割 ('train', 'val')
        is_main_process: 是否是主进程
        
    Returns:
        数据集实例
    """
    # 处理多长度训练参数
    # 设置多长度训练比例
    if args.multi_length_training:
        if split == 'train':
            multi_length_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
        else:
            # 验证集和测试集只使用1.0比例，以匹配缓存文件
            multi_length_ratios = [1.0]
    else:
        multi_length_ratios = [1.0]
    
    # 确定缓存路径
    if split == 'train':
        cache_path = args.cache_path
    elif split == 'val' or split == 'dev':
        cache_path = args.val_cache_path if args.val_cache_path is not None else args.cache_path
    else:
        cache_path = args.cache_path
    
    if args.use_cache and cache_path is None:
        # 自动生成缓存路径
        cache_dir = os.path.join(os.path.dirname(args.data_path), 'window_params')
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"window_params_{split}_ws{args.window_size}_ws{args.window_stride}"
        if args.max_samples is not None:
            filename += f"_max{args.max_samples}"
        filename += ".pkl"
        cache_path = os.path.join(cache_dir, filename)
    
    # 创建数据集
    if args.use_cache and os.path.exists(cache_path):
        # 使用缓存加载数据集
        if is_main_process:
            print(f"使用缓存文件: {cache_path}")
        
        try:
            dataset = CachedLazySeamlessInteractionWindowDataset(
                data_path=args.data_path,
                split=split,
                window_size=args.window_size,
                window_stride=args.window_stride,
                multi_length_training=multi_length_ratios,
                load_video=False,
                load_audio=False,
                max_samples=args.max_samples,
                cache_path=cache_path
            )
            # 验证数据集是否正确加载
            if len(dataset) == 0:
                raise ValueError("数据集为空，缓存可能损坏")
            return dataset  # 早期返回成功加载的数据集
        except Exception as e:
            if is_main_process:
                print(f"加载缓存失败: {e}")
                print("回退到常规数据加载方式...")
            args.use_cache = False
    
    # 使用常规方式加载数据集
    if is_main_process:
        if args.use_cache:
            print("缓存不可用，使用常规方式加载数据集...")
        else:
            print("使用常规方式加载数据集...")
    
    try:
        from lazy_window_dataset import LazySeamlessInteractionWindowDataset
        dataset = LazySeamlessInteractionWindowDataset(
            data_path=args.data_path,
            split=split,
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=multi_length_ratios,
            load_video=False,
            load_audio=False,
            max_samples=args.max_samples
        )
        # 验证数据集
        if len(dataset) == 0:
            raise ValueError("数据集为空")
        
        # 如果需要保存缓存
        if args.save_cache and is_main_process:
            try:
                print("保存窗口参数到缓存...")
                save_path = save_window_params(
                    data_path=args.data_path,
                    split=split,
                    window_size=args.window_size,
                    window_stride=args.window_stride,
                    multi_length_training=multi_length_ratios,
                    max_samples=args.max_samples
                )
                print(f"窗口参数已保存到: {save_path}")
            except Exception as e:
                print(f"保存缓存失败: {e}")
        
        return dataset
    except Exception as e:
        if is_main_process:
            print(f"加载数据集失败: {e}")
            print("尝试使用简化数据集...")
        
        # 最后的回退方案：创建一个简单的数据集
        try:
            return CachedWindowDataset(
                data_path=args.data_path,
                window_size=args.window_size,
                split=split,
                use_cache=False,
                cache_path=None
            )
        except Exception as e2:
            if is_main_process:
                print(f"创建简化数据集也失败: {e2}")
            raise RuntimeError("无法创建任何数据集")


def train_one_epoch(model, dataloader, optimizer, epoch, device, is_main_process, log_interval):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, disable=not is_main_process)
    for batch_idx, batch in enumerate(progress_bar):
        # 这里需要根据实际的数据格式进行修改
        # 假设batch是一个列表，每个元素是一个字典
        
        # 将数据移动到设备
        # 这里需要根据实际的数据格式进行修改
        # 例如: batch_data = {key: torch.stack([item[key] for item in batch]).to(device) for key in batch[0].keys()}
        
        # 前向传播
        # outputs = model(batch_data)
        # loss = compute_loss(outputs, batch_data)
        
        # 这里使用一个虚拟的损失作为示例
        loss = torch.tensor(0.1, requires_grad=True).to(device)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        if is_main_process:
            progress_bar.set_description(f'Epoch {epoch} [{batch_idx}/{num_batches}]')
            progress_bar.set_postfix(loss=loss.item())
            
            if batch_idx % log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx}/{num_batches}]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, mean, std, loss_fn):
    """验证模型 - 与rvq_seamless_multi_gpu.py一致"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # 自定义批处理函数处理
            if isinstance(batch, list):
                # 如果batch是列表，转换为字典格式
                max_len = max([item['pose'].shape[0] for item in batch])
                batch_size = len(batch)
                pose_dim = batch[0]['pose'].shape[1]
                
                poses = torch.zeros(batch_size, max_len, pose_dim)
                masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
                
                for i, item in enumerate(batch):
                    pose = item['pose']
                    length = pose.shape[0]
                    poses[i, :length] = pose
                    masks[i, :length] = True
                
                batch = {'pose': poses, 'mask': masks}
            
            gt_motion = batch['pose'].to(device)
            batch_mask = batch['mask'].to(device)
            
            # 数据标准化
            if mean.shape[0] == 330 and gt_motion.shape[2] == 156:
                # 如果均值是完整姿态(330)，我们只使用前156个值
                mean_tensor = torch.from_numpy(mean[:156]).to(device)
                std_tensor = torch.from_numpy(std[:156]).to(device)
            else:
                mean_tensor = torch.from_numpy(mean).to(device)
                std_tensor = torch.from_numpy(std).to(device)
            
            # 扩展维度以匹配(batch_size, seq_len, dim)
            mean_tensor = mean_tensor.unsqueeze(0).unsqueeze(0)
            std_tensor = std_tensor.unsqueeze(0).unsqueeze(0)
            
            gt_motion = (gt_motion - mean_tensor) / std_tensor
            
            # 前向传播
            with torch.cuda.amp.autocast():
                output = model(gt_motion, mask=batch_mask)
                pred_motion = output['pose']
                loss = loss_fn(pred_motion, gt_motion, mask=batch_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, writer, logger, device, args, mean_pose, std_pose, loss_fn, is_main_process):
    """训练模型 - 与rvq_seamless_multi_gpu.py一致"""
    # 训练循环 - 与rvq_seamless_multi_gpu.py一致
    print(f"开始训练，总迭代次数: {args.total_iter}")
    
    # 创建迭代器
    train_iter = iter(train_loader)
    
    for iteration in range(args.total_iter):
        # 设置模型为训练模式
        model.train()
        
        # 获取下一批数据，如果需要则重新创建迭代器
        try:
            batch = next(train_iter)
        except StopIteration:
            # 重新创建迭代器
            if args.world_size > 1:
                train_loader.sampler.set_epoch(iteration // len(train_loader) + 1)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # 自定义批处理函数处理
        if isinstance(batch, list):
            # 如果batch是列表，转换为字典格式
            max_len = max([item['pose'].shape[0] for item in batch])
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]
            
            poses = torch.zeros(batch_size, max_len, pose_dim)
            masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
            
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = pose
                masks[i, :length] = True
            
            batch = {'pose': poses, 'mask': masks}
        
        # 将数据移到设备
        gt_motion = batch['pose'].to(device)  # (bs, seq_len, 156)
        batch_mask = batch['mask'].to(device)  # (bs, seq_len)
        
        # 数据标准化
        # 均值和标准差是针对完整姿态的(330,)，但我们的数据是(batch_size, seq_len, 156)
        # 我们需要只使用对应部分的均值和标准差
        if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
            # 如果均值是完整姿态(330)，我们只使用前156个值
            mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
            std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
        else:
            mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
            std_pose_tensor = torch.from_numpy(std_pose).to(device)
        
        # 扩展维度以匹配(batch_size, seq_len, dim)
        mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
        std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
        
        gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
        
        # 前向传播
        optimizer.zero_grad()
        
        # 使用模型进行重构
        with torch.cuda.amp.autocast():
            pred_motion, loss_commit, perplexity = model(gt_motion).values()
            loss_motion = loss_fn(pred_motion, gt_motion, list(range(gt_motion.shape[2])))  # 使用实际维度
            loss_vel = 0  # 暂时不计算速度损失
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        scheduler.step()
        
        # 打印训练信息
        if iteration % args.print_iter == 0 and is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iteration {iteration}/{args.total_iter}, Loss: {loss.item():.6f}, LR: {current_lr:.8f}")
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('Learning_Rate', current_lr, iteration)
            writer.add_scalar('./Train/L1', loss_motion.item(), iteration)
            writer.add_scalar('./Train/PPL', perplexity.item(), iteration)
            writer.add_scalar('./Train/Commit', loss_commit.item(), iteration)
        
        # 定期验证
        if iteration % args.eval_iter == 0:
            if is_main_process:
                # 保存模型
                checkpoint = {
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                if use_amp:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint, os.path.join(args.out_dir, f'net_{iteration}.pth'))
                
                print(f"模型已保存: net_{iteration}.pth")
            
            # 使用验证集进行评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            if is_main_process:
                eval_pbar = tqdm(val_loader, desc=f"验证集评估 (迭代 {iteration})", leave=False)
            else:
                eval_pbar = val_loader
            
            with torch.no_grad():
                for batch in eval_pbar:
                    # 自定义批处理函数处理
                    if isinstance(batch, list):
                        # 如果batch是列表，转换为字典格式
                        max_len = max([item['pose'].shape[0] for item in batch])
                        batch_size = len(batch)
                        pose_dim = batch[0]['pose'].shape[1]
                        
                        poses = torch.zeros(batch_size, max_len, pose_dim)
                        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
                        
                        for i, item in enumerate(batch):
                            pose = item['pose']
                            length = pose.shape[0]
                            poses[i, :length] = pose
                            masks[i, :length] = True
                        
                        batch = {'pose': poses, 'mask': masks}
                    
                    gt_motion = batch['pose'].to(device)  # (bs, seq_len, 156)
                    batch_mask = batch['mask'].to(device)  # (bs, seq_len)
                    
                    # 标准化
                    if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                        # 如果均值是完整姿态(330)，我们只使用前156个值
                        mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
                        std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
                    else:
                        mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
                        std_pose_tensor = torch.from_numpy(std_pose).to(device)
                    
                    # 扩展维度以匹配(batch_size, seq_len, dim)
                    mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                    std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
                    
                    gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
                    
                    # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
                    with torch.cuda.amp.autocast():
                        pred_motion, loss_commit, perplexity = model(gt_motion).values()
                    
                    # 模型输出已经是(bs, seq_len, dim_pose)格式，不需要转置
                    
                    # 计算L2距离
                    diff = pred_motion - gt_motion
                    l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
                    total_l2 += l2_batch
                    num_batches += 1
                    
                    # 更新评估进度条
                    if is_main_process:
                        eval_pbar.set_postfix({'L2': f'{l2_batch:.5f}'})
            
            avg_l2 = total_l2 / num_batches
            if is_main_process:
                logger.info(f"Validation. Iter {iteration}: \t L2 Distance: {avg_l2:.5f}")
                writer.add_scalar('./Val/L2', avg_l2, iteration)
            
            model.train()
            
            # 等待所有进程完成
            if args.world_size > 1:
                torch.distributed.barrier()
    
    # 训练完成，保存最终模型
    if is_main_process:
        checkpoint = {
            'iteration': args.total_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if use_amp:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(args.out_dir, 'final_model.pth'))
        
        print("训练完成，最终模型已保存: final_model.pth")
        writer.close()
    
    # 等待所有进程完成
    if args.world_size > 1:
        torch.distributed.barrier()
    
    if is_main_process:
        print("训练完成!")
        writer.close()


def main():
    """主函数"""
    # 解析参数
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置分布式训练
    rank, local_rank, world_size, is_main_process = setup_distributed()
    
    # 更新args中的分布式参数
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    
    # 设置设备
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录（只有主进程创建）
    if is_main_process:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = get_logger(args.out_dir, rank)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # 创建数据集
    if is_main_process:
        print("Building dataset...")
    
    start_time = time.time()
    
    # 创建训练数据集
    train_dataset = create_dataset(args, 'train', is_main_process)
    
    # 创建验证集数据集
    val_dataset = create_dataset(args, 'val', is_main_process)  # 使用'val'而不是'dev'，与缓存文件一致
    
    data_load_time = time.time() - start_time
    
    if is_main_process:
        print(f"数据加载完成，耗时: {data_load_time:.2f} 秒")
        print(f"训练数据集大小: {len(train_dataset)} 个窗口")
        print(f"验证数据集大小: {len(val_dataset)} 个窗口")
    
    # 创建数据加载器 - 与rvq_seamless_multi_gpu.py一致
    def custom_collate_fn(batch):
        """自定义批处理函数，处理不同长度的序列"""
        # 找到批次中最长的序列
        max_len = max([item['pose'].shape[0] for item in batch])
        
        # 创建填充后的批次数据
        batch_size = len(batch)
        pose_dim = batch[0]['pose'].shape[1]
        
        poses = torch.zeros(batch_size, max_len, pose_dim)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, item in enumerate(batch):
            pose = item['pose']
            length = pose.shape[0]
            poses[i, :length] = pose
            masks[i, :length] = True
        
        return {
            'pose': poses,
            'mask': masks,
            'id': [item['id'] for item in batch]
        }
    
    # 使用分布式采样器（如果是分布式模式）
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # 如果没有采样器，则shuffle
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=val_sampler
    )
    
    # 加载均值和标准差
    mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
    std_pose = np.load('mean_std/seamless_smplh_std.npy')
    
    # 创建模型 - 与rvq_seamless_multi_gpu.py完全一致
    # 确保参数正确设置
    if not hasattr(args, 'nb_code'):
        args.nb_code = args.codebook_size
    if not hasattr(args, 'dilation_growth_rate'):
        args.dilation_growth_rate = args.dilation
    if not hasattr(args, 'vq_norm'):
        args.vq_norm = None
    if not hasattr(args, 'mu'):
        args.mu = args.emb_norm
    
    # 输入维度是156（完整姿态）
    input_dim = 156
    
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
    ).to(device)
    
    # 如果使用多GPU训练
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # 创建优化器 - 与rvq_seamless_multi_gpu.py一致
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    
    # 创建损失函数 - 与rvq_seamless_multi_gpu.py一致
    Loss = ReConsLoss(args.recons_loss)
    
    # 设置混合精度训练 - 与rvq_seamless_multi_gpu.py一致
    use_amp = args.mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # 设置梯度检查点 - 与rvq_seamless_multi_gpu.py一致
    if args.gradient_checkpointing:
        # 如果模型支持梯度检查点，启用它
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
        else:
            if is_main_process:
                print("警告: 模型不支持梯度检查点")
    
    # 训练循环 - 与rvq_seamless_multi_gpu.py一致
    print(f"开始训练，总迭代次数: {args.total_iter}")
    
    # 创建迭代器
    train_iter = iter(train_loader)
    
    for iteration in range(args.total_iter):
        # 设置模型为训练模式
        model.train()
        
        # 获取下一批数据，如果需要则重新创建迭代器
        try:
            batch = next(train_iter)
        except StopIteration:
            # 重新创建迭代器
            if world_size > 1:
                train_sampler.set_epoch(iteration // len(train_loader) + 1)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # 自定义批处理函数处理
        if isinstance(batch, list):
            # 如果batch是列表，转换为字典格式
            max_len = max([item['pose'].shape[0] for item in batch])
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]
            
            poses = torch.zeros(batch_size, max_len, pose_dim)
            masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
            
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = pose
                masks[i, :length] = True
            
            batch = {'pose': poses, 'mask': masks}
        
        # 将数据移到设备
        gt_motion = batch['pose'].to(device)  # (bs, seq_len, 156)
        batch_mask = batch['mask'].to(device)  # (bs, seq_len)
        
        # 数据标准化
        # 均值和标准差是针对完整姿态的(330,)，但我们的数据是(batch_size, seq_len, 156)
        # 我们需要只使用对应部分的均值和标准差
        if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
            # 如果均值是完整姿态(330)，我们只使用前156个值
            mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
            std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
        else:
            mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
            std_pose_tensor = torch.from_numpy(std_pose).to(device)
        
        # 扩展维度以匹配(batch_size, seq_len, dim)
        mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
        std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
        
        gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
        
        # 前向传播
        optimizer.zero_grad()
        
        # 使用模型进行重构
        if use_amp:
            with torch.cuda.amp.autocast():
                pred_motion, loss_commit, perplexity = model(gt_motion).values()
                loss_motion = Loss.my_forward(pred_motion, gt_motion, list(range(gt_motion.shape[2])))  # 使用实际维度
                loss_vel = 0  # 暂时不计算速度损失
                loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        else:
            pred_motion, loss_commit, perplexity = model(gt_motion).values()
            loss_motion = Loss.my_forward(pred_motion, gt_motion, list(range(gt_motion.shape[2])))  # 使用实际维度
            loss_vel = 0  # 暂时不计算速度损失
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # 更新参数
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 打印训练信息
        if iteration % args.print_iter == 0 and is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iteration {iteration}/{args.total_iter}, Loss: {loss.item():.6f}, LR: {current_lr:.8f}")
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('Learning_Rate', current_lr, iteration)
            writer.add_scalar('./Train/L1', loss_motion.item(), iteration)
            writer.add_scalar('./Train/PPL', perplexity.item(), iteration)
            writer.add_scalar('./Train/Commit', loss_commit.item(), iteration)
        
        # 定期验证
        if iteration % args.eval_iter == 0:
            if is_main_process:
                # 保存模型
                checkpoint = {
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                if use_amp:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint, os.path.join(args.out_dir, f'net_{iteration}.pth'))
                
                print(f"模型已保存: net_{iteration}.pth")
            
            # 使用验证集进行评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            if is_main_process:
                eval_pbar = tqdm(val_loader, desc=f"验证集评估 (迭代 {iteration})", leave=False)
            else:
                eval_pbar = val_loader
            
            with torch.no_grad():
                for batch in eval_pbar:
                    # 自定义批处理函数处理
                    if isinstance(batch, list):
                        # 如果batch是列表，转换为字典格式
                        max_len = max([item['pose'].shape[0] for item in batch])
                        batch_size = len(batch)
                        pose_dim = batch[0]['pose'].shape[1]
                        
                        poses = torch.zeros(batch_size, max_len, pose_dim)
                        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
                        
                        for i, item in enumerate(batch):
                            pose = item['pose']
                            length = pose.shape[0]
                            poses[i, :length] = pose
                            masks[i, :length] = True
                        
                        batch = {'pose': poses, 'mask': masks}
                    
                    gt_motion = batch['pose'].to(device)  # (bs, seq_len, 156)
                    batch_mask = batch['mask'].to(device)  # (bs, seq_len)
                    
                    # 标准化
                    if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                        # 如果均值是完整姿态(330)，我们只使用前156个值
                        mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
                        std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
                    else:
                        mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
                        std_pose_tensor = torch.from_numpy(std_pose).to(device)
                    
                    # 扩展维度以匹配(batch_size, seq_len, dim)
                    mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                    std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
                    
                    gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
                    
                    # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            pred_motion, loss_commit, perplexity = model(gt_motion).values()
                    else:
                        pred_motion, loss_commit, perplexity = model(gt_motion).values()
                    
                    # 模型输出已经是(bs, seq_len, dim_pose)格式，不需要转置
                    
                    # 计算L2距离
                    diff = pred_motion - gt_motion
                    l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
                    total_l2 += l2_batch
                    num_batches += 1
                    
                    # 更新评估进度条
                    if is_main_process:
                        eval_pbar.set_postfix({'L2': f'{l2_batch:.5f}'})
            
            avg_l2 = total_l2 / num_batches
            if is_main_process:
                logger.info(f"Validation. Iter {iteration}: \t L2 Distance: {avg_l2:.5f}")
                writer.add_scalar('./Val/L2', avg_l2, iteration)
            
            model.train()
            
            # 等待所有进程完成
            if world_size > 1:
                torch.distributed.barrier()
    
    # 训练完成，保存最终模型
    if is_main_process:
        torch.save({
            'iteration': args.total_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, os.path.join(args.out_dir, 'final_model.pth'))
        
        print("训练完成，最终模型已保存: final_model.pth")
        writer.close()
    
    # 等待所有进程完成
    if world_size > 1:
        torch.distributed.barrier()
    
    if is_main_process:
        print("训练完成!")
        writer.close()


if __name__ == "__main__":
    main()