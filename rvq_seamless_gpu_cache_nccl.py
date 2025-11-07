import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import json
import random
from collections import defaultdict
import time
import gc
import psutil
import threading
import multiprocessing as mp
from functools import partial
import signal

# 导入必要的模块
from model.vqvae import RVQVAE
from data.seamless_interaction import SeamlessInteractionDataset
from data.lazy_window_dataset import LazySeamlessInteractionWindowDataset, CachedLazySeamlessInteractionWindowDataset
from utils.recons_loss import ReConsLoss
from utils.utils import print_log

def custom_collate_fn(batch):
    """
    自定义批次整理函数，处理不同长度的序列
    """
    # 找出批次中最长的序列长度
    max_len = max([item['pose'].shape[0] for item in batch])
    
    # 初始化批次数据
    batch_size = len(batch)
    pose_dim = batch[0]['pose'].shape[1]
    
    # 创建填充后的批次数据
    pose_batch = torch.zeros(batch_size, max_len, pose_dim, dtype=torch.float32)
    mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # 填充数据
    for i, item in enumerate(batch):
        seq_len = item['pose'].shape[0]
        pose_batch[i, :seq_len] = item['pose']
        mask_batch[i, :seq_len] = True  # 有效位置为True
    
    return {
        'pose': pose_batch,
        'mask': mask_batch
    }

class GPUMemoryDataset(Dataset):
    """
    将数据预加载到显存中的数据集包装器
    """
    def __init__(self, base_dataset, device='cuda', cache_size=None):
        self.base_dataset = base_dataset
        self.device = device
        self.cache_size = cache_size if cache_size else len(base_dataset)
        
        # 初始化缓存
        self.cache = {}
        self.cache_order = []
        
        # 预加载部分数据到显存
        self._preload_data()
    
    def _preload_data(self):
        """预加载数据到显存"""
        print(f"开始预加载数据到显存，缓存大小: {self.cache_size}")
        
        # 随机选择要缓存的数据索引
        if self.cache_size < len(self.base_dataset):
            cache_indices = random.sample(range(len(self.base_dataset)), self.cache_size)
        else:
            cache_indices = list(range(len(self.base_dataset)))
        
        # 预加载数据
        for i in tqdm(cache_indices, desc="预加载数据到显存"):
            data = self.base_dataset[i]
            # 将数据移动到显存
            gpu_data = {
                'pose': torch.tensor(data['pose'], device=self.device),
                'mask': torch.tensor(data['mask'], device=self.device)
            }
            self.cache[i] = gpu_data
            self.cache_order.append(i)
        
        print(f"成功预加载 {len(self.cache)} 个样本到显存")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 如果数据在缓存中，直接返回
        if idx in self.cache:
            return self.cache[idx]
        
        # 否则从基础数据集加载并移动到显存
        data = self.base_dataset[idx]
        gpu_data = {
            'pose': torch.tensor(data['pose'], device=self.device),
            'mask': torch.tensor(data['mask'], device=self.device)
        }
        
        # 如果缓存未满，添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = gpu_data
            self.cache_order.append(idx)
        
        return gpu_data

class PrefetchDataLoader:
    """
    预取数据加载器，提前将数据移动到显存
    """
    def __init__(self, loader, device='cuda', prefetch_factor=2):
        self.loader = loader
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.loader_iter = iter(loader)
        self.prefetch_queue = []
        self._prefetch_initial()
    
    def _prefetch_initial(self):
        """初始预取数据"""
        for _ in range(self.prefetch_factor):
            try:
                batch = next(self.loader_iter)
                # 将数据移动到显存
                gpu_batch = {
                    'pose': batch['pose'].to(self.device, non_blocking=True),
                    'mask': batch['mask'].to(self.device, non_blocking=True)
                }
                self.prefetch_queue.append(gpu_batch)
            except StopIteration:
                break
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.prefetch_queue:
            try:
                batch = next(self.loader_iter)
                # 将数据移动到显存
                gpu_batch = {
                    'pose': batch['pose'].to(self.device, non_blocking=True),
                    'mask': batch['mask'].to(self.device, non_blocking=True)
                }
                self.prefetch_queue.append(gpu_batch)
            except StopIteration:
                raise StopIteration
        
        # 返回预取的数据
        result = self.prefetch_queue.pop(0)
        
        # 预取下一个批次
        try:
            batch = next(self.loader_iter)
            # 将数据移动到显存
            gpu_batch = {
                'pose': batch['pose'].to(self.device, non_blocking=True),
                'mask': batch['mask'].to(self.device, non_blocking=True)
            }
            self.prefetch_queue.append(gpu_batch)
        except StopIteration:
            pass
        
        return result
    
    def __len__(self):
        return len(self.loader)

def get_memory_info():
    """获取内存和显存使用情况"""
    # 获取系统内存
    memory = psutil.virtual_memory()
    sys_mem_used = memory.used / (1024 ** 3)  # GB
    sys_mem_total = memory.total / (1024 ** 3)  # GB
    sys_mem_percent = memory.percent
    
    # 获取GPU显存
    gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    
    return {
        'sys_mem_used': sys_mem_used,
        'sys_mem_total': sys_mem_total,
        'sys_mem_percent': sys_mem_percent,
        'gpu_mem_allocated': gpu_mem_allocated,
        'gpu_mem_reserved': gpu_mem_reserved,
        'gpu_mem_total': gpu_mem_total
    }

def print_memory_info(prefix=""):
    """打印内存和显存使用情况"""
    mem_info = get_memory_info()
    print(f"{prefix}系统内存: {mem_info['sys_mem_used']:.2f}GB / {mem_info['sys_mem_total']:.2f}GB ({mem_info['sys_mem_percent']:.1f}%)")
    print(f"{prefix}GPU显存已分配: {mem_info['gpu_mem_allocated']:.2f}GB / {mem_info['gpu_mem_total']:.2f}GB ({mem_info['gpu_mem_allocated']/mem_info['gpu_mem_total']*100:.1f}%)")
    print(f"{prefix}GPU显存已预留: {mem_info['gpu_mem_reserved']:.2f}GB / {mem_info['gpu_mem_total']:.2f}GB ({mem_info['gpu_mem_reserved']/mem_info['gpu_mem_total']*100:.1f}%)")

def is_main_process():
    """检查是否是主进程"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = local_rank
        
        print(f"初始化分布式训练: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # 设置NCCL超时和调试参数
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 阻塞等待
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 异步错误处理
        
        # 网络优化设置
        os.environ['NCCL_SOCKET_TIMEOUT'] = '60000'
        os.environ['NCCL_NET_RETRY_COUNT'] = '10'
        os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand（如果不需要）
        os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P（如果GPU之间通信有问题）
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # 强制使用树形算法
        
        # PyTorch分布式设置
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，便于调试
        
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=30))
        torch.cuda.set_device(device)
        
        return True, rank, world_size, local_rank
    else:
        # 单GPU训练
        print("使用单GPU训练")
        return False, 0, 1, 0

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def signal_handler(sig, frame):
    """信号处理函数，确保正确清理资源"""
    print("接收到中断信号，正在清理资源...")
    cleanup_distributed()
    sys.exit(0)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用显存缓存的RVQ训练（NCCL优化版）')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='datasets/train', help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='datasets/val', help='验证数据路径')
    parser.add_argument('--window_size', type=int, default=64, help='窗口大小')
    parser.add_argument('--window_stride', type=int, default=20, help='窗口步长')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--multi_length_training', action='store_true', help='是否使用多长度训练')
    
    # 模型参数
    parser.add_argument('--nb_code', type=int, default=512, help='码本大小')
    parser.add_argument('--code_dim', type=int, default=128, help='码本维度')
    parser.add_argument('--down_t', type=int, default=2, help='下采样层数')
    parser.add_argument('--stride_t', type=int, default=2, help='时间步长')
    parser.add_argument('--width', type=int, default=512, help='模型宽度')
    parser.add_argument('--depth', type=int, default=3, help='模型深度')
    parser.add_argument('--dilation_growth_rate', type=int, default=3, help='膨胀增长率')
    parser.add_argument('--vq_act', type=str, default='relu', help='VQ激活函数')
    parser.add_argument('--vq_norm', type=str, default=None, help='VQ归一化')
    parser.add_argument('--mu', type=float, default=0.99, help='动量系数')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--total_iter', type=int, default=300000, help='总迭代次数')
    parser.add_argument('--lr_scheduler', type=list, default=[100000, 200000, 250000], help='学习率调度器里程碑')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--commit', type=float, default=0.02, help='commitment损失权重')
    parser.add_argument('--loss_vel', type=float, default=0.1, help='速度损失权重')
    parser.add_argument('--recons_loss', type=str, default='l1', help='重构损失类型')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名')
    parser.add_argument('--world_size', type=int, default=1, help='世界大小')
    
    # 其他参数
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='是否使用梯度检查点')
    parser.add_argument('--print_every', type=int, default=100, help='打印间隔')
    parser.add_argument('--eval_every', type=int, default=5000, help='评估间隔')
    parser.add_argument('--save_every', type=int, default=10000, help='保存间隔')
    parser.add_argument('--exp_name', type=str, default='rvq_gpu_cache_nccl', help='实验名称')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='输出目录')
    
    # GPU缓存参数
    parser.add_argument('--use_gpu_cache', action='store_true', help='是否使用GPU缓存数据集')
    parser.add_argument('--gpu_cache_size', type=int, default=5000, help='GPU缓存数据集大小')
    parser.add_argument('--use_prefetch', action='store_true', help='是否使用预取数据加载器')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='预取因子')
    
    # DataLoader参数
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader工作进程数')
    parser.add_argument('--pin_memory', action='store_true', help='是否使用固定内存')
    parser.add_argument('--persistent_workers', action='store_true', help='是否保持工作进程活跃')
    
    # NCCL优化参数
    parser.add_argument('--nccl_timeout', type=int, default=1800, help='NCCL超时时间（秒）')
    parser.add_argument('--sync_batch_norm', action='store_true', help='是否使用同步批归一化')
    parser.add_argument('--find_unused_parameters', action='store_true', help='是否查找未使用的参数')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    # 设置分布式训练环境
    is_distributed, rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank
    args.world_size = world_size
    
    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 打印内存信息
    if is_main_process():
        print("初始内存状态:")
        print_memory_info()
    
    # 创建数据集
    if is_main_process():
        print("创建数据集...")
    
    # 检查是否存在缓存文件
    cache_file_train = f"datasets/window_params/window_params_train_ws{args.window_size}_ws{args.window_stride}_fixed.pkl"
    cache_file_val = f"datasets/window_params/window_params_val_ws{args.window_size}_ws{args.window_stride}_fixed.pkl"
    
    if os.path.exists(cache_file_train) and os.path.exists(cache_file_val):
        if is_main_process():
            print("使用缓存的数据集...")
        
        train_dataset = CachedLazySeamlessInteractionWindowDataset(
            args.train_data_path,
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=args.multi_length_training,
            cache_file=cache_file_train
        )
        
        val_dataset = CachedLazySeamlessInteractionWindowDataset(
            args.val_data_path,
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=args.multi_length_training,
            cache_file=cache_file_val
        )
    else:
        if is_main_process():
            print("创建新的数据集...")
        
        train_dataset = LazySeamlessInteractionWindowDataset(
            args.train_data_path,
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=args.multi_length_training
        )
        
        val_dataset = LazySeamlessInteractionWindowDataset(
            args.val_data_path,
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=args.multi_length_training
        )
    
    # 使用GPU缓存数据集
    if args.use_gpu_cache:
        if is_main_process():
            print(f"使用GPU缓存数据集，缓存大小: {args.gpu_cache_size}")
        
        train_dataset = GPUMemoryDataset(
            train_dataset,
            device=device,
            cache_size=args.gpu_cache_size
        )
        
        val_dataset = GPUMemoryDataset(
            val_dataset,
            device=device,
            cache_size=min(args.gpu_cache_size // 4, len(val_dataset))  # 验证集缓存较小
        )
    
    # 使用分布式采样器（如果是分布式模式）
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # 如果没有采样器，则shuffle
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=train_sampler,
        persistent_workers=args.persistent_workers and args.num_workers > 0,  # 保持工作进程活跃
        pin_memory=args.pin_memory and not args.use_gpu_cache,  # 如果使用GPU缓存，不需要pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=val_sampler,
        persistent_workers=args.persistent_workers and args.num_workers > 0,  # 保持工作进程活跃
        pin_memory=args.pin_memory and not args.use_gpu_cache,  # 如果使用GPU缓存，不需要pin_memory
    )
    
    # 使用预取数据加载器
    if args.use_prefetch:
        if is_main_process():
            print(f"使用预取数据加载器，预取因子: {args.prefetch_factor}")
        
        train_loader = PrefetchDataLoader(
            train_loader,
            device=device,
            prefetch_factor=args.prefetch_factor
        )
        
        val_loader = PrefetchDataLoader(
            val_loader,
            device=device,
            prefetch_factor=args.prefetch_factor
        )
    
    # 打印内存信息
    if is_main_process():
        print("数据加载后内存状态:")
        print_memory_info()
    
    # 加载均值和标准差
    mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
    std_pose = np.load('mean_std/seamless_smplh_std.npy')
    
    # 创建模型
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
    )
    
    # 设置设备
    model.cuda(device)
    
    # 使用分布式数据并行（如果是分布式模式）
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=args.find_unused_parameters, output_device=local_rank)
    
    model.train()
    
    # 打印内存信息
    if is_main_process():
        print("模型加载后内存状态:")
        print_memory_info()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    
    # 创建损失函数
    Loss = ReConsLoss(args.recons_loss)
    
    # 设置混合精度训练
    use_amp = args.mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # 设置梯度检查点
    if args.gradient_checkpointing:
        # 如果模型支持梯度检查点，启用它
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
        else:
            if is_main_process:
                print("警告: 模型不支持梯度检查点")
    
    # 训练循环
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    
    if is_main_process:
        print("开始训练循环...")
        print(f"开始训练，总共 {args.total_iter} 次迭代")
        print(f"使用 {world_size} 个GPU，每个GPU的batch_size为 {args.batch_size}")
        print(f"总有效batch_size为 {args.batch_size * world_size}")
        
        # 创建进度条
        pbar = tqdm(range(1, args.total_iter + 1), desc="训练进度", 
                    bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}]',
                    mininterval=1.0,  # 最小更新间隔为1秒
                    miniters=1)  # 最小迭代次数为1
    else:
        pbar = range(1, args.total_iter + 1)
    
    # 记录最佳损失
    best_recons_loss = float('inf')
    
    if is_main_process:
        print("即将进入训练循环...")
    
    # 创建数据迭代器
    train_iter = iter(train_loader)
    
    # 定期打印内存信息
    memory_print_interval = 1000
    
    # 定期同步检查
    sync_check_interval = 100
    
    for nb_iter in pbar:
        # 设置分布式采样器的epoch（如果是分布式模式）
        if is_distributed:
            train_sampler.set_epoch(nb_iter)

        try:
            # 获取数据批次
            try:
                batch = next(train_iter)
            except StopIteration:
                # 数据迭代器耗尽，重新创建
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # 如果使用预取数据加载器，数据已经在GPU上
            if args.use_prefetch or args.use_gpu_cache:
                gt_motion = batch['pose']  # 已经在GPU上
                batch_mask = batch['mask']  # 已经在GPU上
            else:
                gt_motion = batch['pose'].to(device)  # (bs, seq_len, 156)
                batch_mask = batch['mask'].to(device)  # (bs, seq_len)

            # 标准化
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

            # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
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

            # 反向传播和优化
            if use_amp:
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # 梯度裁剪
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()

            scheduler.step()

            avg_recons += loss_motion.item()
            avg_perplexity += perplexity.item()
            avg_commit += loss_commit.item()

            # 打印训练信息
            if is_main_process and nb_iter % args.print_every == 0:
                avg_recons /= args.print_every
                avg_perplexity /= args.print_every
                avg_commit /= args.print_every
                
                pbar.set_description(f"训练进度 | 损失: {avg_recons:.5f} | 困惑度: {avg_perplexity:.2f} | Commit: {avg_commit:.5f}")
                
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
            
            # 定期打印内存信息
            if is_main_process and nb_iter % memory_print_interval == 0:
                print(f"\n迭代 {nb_iter} 内存状态:")
                print_memory_info()
                
                # 清理未使用的缓存
                torch.cuda.empty_cache()
                gc.collect()
                
                print("清理后内存状态:")
                print_memory_info()
            
            # 定期同步检查（防止NCCL超时）
            if is_distributed and nb_iter % sync_check_interval == 0:
                try:
                    # 使用barrier进行同步检查
                    dist.barrier()
                except Exception as e:
                    if is_main_process():
                        print(f"同步检查失败: {e}")
                    # 继续训练，不中断
            
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 清理资源
    cleanup_distributed()
    
    if is_main_process:
        print("训练完成!")
        print_memory_info()

if __name__ == "__main__":
    main()