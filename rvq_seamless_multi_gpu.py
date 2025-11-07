#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
import warnings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型和数据集
from models.vq.model import RVQVAE
from lazy_window_dataset import LazySeamlessInteractionWindowDataset, CachedLazySeamlessInteractionWindowDataset
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

def get_logger(log_dir, rank):
    """创建日志记录器"""
    logger = logging.getLogger(f'Training_{rank}')
    logger.setLevel(logging.INFO)
    
    # 只有主进程创建文件处理器和控制台输出
    if rank == 0:
        # 创建文件处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_args_parser():
    """获取参数解析器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RVQ-VAE无缝交互多GPU训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction', help='数据路径')
    parser.add_argument('--body_part', type=str, default='whole', choices=['upper', 'lower', 'hands', 'whole'], help='身体部位')
    
    # 模型参数
    parser.add_argument('--nb_code', type=int, default=1024, help='码本大小')
    parser.add_argument('--code_dim', type=int, default=128, help='码本维度')
    parser.add_argument('--down_t', type=int, default=2, help='下采样层数')
    parser.add_argument('--stride_t', type=int, default=2, help='时间步长')
    parser.add_argument('--width', type=int, default=512, help='网络宽度')
    parser.add_argument('--depth', type=int, default=3, help='网络深度')
    parser.add_argument('--dilation_growth_rate', type=int, default=3, help='膨胀率增长')
    parser.add_argument('--vq_act', type=str, default='relu', help='VQ激活函数')
    parser.add_argument('--vq_norm', type=str, default=None, help='VQ归一化方法')
    parser.add_argument('--mu', type=float, default=0.99, help='EMA动量')
    parser.add_argument('--num_quantizers', type=int, default=6, help='量化器数量')
    parser.add_argument('--shared_codebook', action='store_true', help='是否共享码本')
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.5, help='量化器dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024, help='每个GPU的批次大小')
    parser.add_argument('--total_iter', type=int, default=10000, help='总迭代次数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lr_scheduler', type=list, default=[5000, 8000], help='学习率调度器里程碑')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--commit', type=float, default=0.02, help='commitment loss权重')
    parser.add_argument('--loss_vel', type=float, default=0.01, help='速度损失权重')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='重建损失类型')
    
    # 数据参数
    parser.add_argument('--window_size', type=int, default=64, help='窗口大小(帧数)')
    parser.add_argument('--window_stride', type=int, default=20, help='窗口步长(帧数)')
    parser.add_argument('--multi_length_training', type=float, nargs='+', default=None, help='多长度训练比例列表，如: 0.5 0.75 1.0 1.25 1.5')
    parser.add_argument('--max_samples', type=int, default=None, nargs='?', const=None, help='最大样本数，设置为None使用整个数据集')
    parser.add_argument('--use_cache', action='store_true', help='是否使用缓存数据')
    parser.add_argument('--cache_path', type=str, default=None, help='训练集缓存路径')
    parser.add_argument('--val_cache_path', type=str, default=None, help='验证集缓存路径')
    parser.add_argument('--cache_train', type=str, default=None, help='训练集缓存路径（别名）')
    parser.add_argument('--cache_val', type=str, default=None, help='验证集缓存路径（别名）')
    
    # 输出参数
    parser.add_argument('--out_dir', type=str, default='experiments', help='输出目录')
    parser.add_argument('--exp_name', type=str, default='rvq_seamless_multi_gpu', help='实验名称')
    parser.add_argument('--print_iter', type=int, default=200, help='打印间隔')
    parser.add_argument('--eval_iter', type=int, default=2000, help='评估间隔')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名')
    
    # 显存优化参数
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='是否使用梯度检查点')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    return parser

def main():
    """主函数"""
    parser = get_args_parser()
    
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
    
    # 检查是否在分布式环境中运行
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    # 初始化is_main_process
    is_main_process = True  # 默认值，稍后会被更新
    
    if is_distributed:
        # 初始化分布式训练环境
        dist.init_process_group(backend='nccl')
        
        # 确保每个进程使用不同的GPU
        torch.cuda.set_device(args.local_rank)
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 更新is_main_process
        is_main_process = rank == 0
        
        # 打印GPU分配信息
        if is_main_process:
            print(f"分布式训练模式: 使用 {world_size} 个GPU")
            print(f"当前进程 (rank {rank}) 使用 GPU {args.local_rank}")
    else:
        # 非分布式模式
        rank = 0
        world_size = 1
        is_main_process = True
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print("警告: 未检测到分布式环境，运行在单GPU模式")
    
    # 只有主进程打印信息
    is_main_process = rank == 0
    
    torch.manual_seed(args.seed)
    
    # 创建输出目录（只有主进程创建）
    if is_main_process:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = get_logger(args.out_dir, rank)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # 创建数据集
    if is_main_process:
        print("Building dataset...")
    
    # 处理多长度训练参数
    split = 'train'  # 定义split变量
    if args.multi_length_training is not None:
        if split == 'train':
            multi_length_ratios = args.multi_length_training
            if is_main_process:
                print(f"使用多长度训练比例: {multi_length_ratios}")
        else:
            # 验证集和测试集只使用1.0比例，以匹配缓存文件
            multi_length_ratios = [1.0]
    else:
        multi_length_ratios = [1.0]
        if is_main_process:
            print("使用单长度训练比例: [1.0]")
    
    # 处理缓存路径别名
    if args.cache_train is not None:
        args.cache_path = args.cache_train
    if args.cache_val is not None:
        args.val_cache_path = args.cache_val

    # 如果提供了缓存路径，自动启用缓存
    if args.cache_path is not None or args.val_cache_path is not None or args.cache_train is not None or args.cache_val is not None:
        args.use_cache = True
        if is_main_process:
            print(f"检测到缓存路径，自动启用缓存模式")
            print(f"缓存路径 - 训练: {args.cache_path or args.cache_train}, 验证: {args.val_cache_path or args.cache_val}")

    # 确保缓存路径正确设置
    if args.use_cache:
        if args.cache_path is None and args.cache_train is not None:
            args.cache_path = args.cache_train
        if args.val_cache_path is None and args.cache_val is not None:
            args.val_cache_path = args.cache_val

    # 确定缓存路径
    if args.use_cache:
        train_cache_path = args.cache_path
        val_cache_path = args.val_cache_path if args.val_cache_path is not None else args.cache_path

        if train_cache_path is None or val_cache_path is None:
            # 自动生成缓存路径
            cache_dir = os.path.join(os.path.dirname(args.data_path), 'window_params')
            os.makedirs(cache_dir, exist_ok=True)
            train_cache_path = os.path.join(cache_dir, f"window_params_train_ws{args.window_size}_ws{args.window_stride}.pkl")
            val_cache_path = os.path.join(cache_dir, f"window_params_val_ws{args.window_size}_ws{args.window_stride}.pkl")
    else:
        train_cache_path = None
        val_cache_path = None
    
    # 创建训练数据集
    if args.use_cache and train_cache_path is not None and os.path.exists(train_cache_path):
        # 使用缓存加载训练数据集
        if is_main_process:
            print(f"使用训练集缓存文件: {train_cache_path}")
        
        try:
            from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
            train_dataset = CachedLazySeamlessInteractionWindowDataset(
                data_path=args.data_path,
                split='train',
                window_size=args.window_size,
                window_stride=args.window_stride,
                multi_length_training=multi_length_ratios,
                load_video=False,
                load_audio=False,
                max_samples=args.max_samples,
                cache_path=train_cache_path
            )
            if len(train_dataset) == 0:
                raise ValueError("训练数据集为空，缓存可能损坏")
        except Exception as e:
            if is_main_process:
                print(f"加载训练集缓存失败: {e}")
                print("回退到常规数据加载方式...")
            args.use_cache = False
    
    if not args.use_cache or train_cache_path is None or not os.path.exists(train_cache_path):
        # 使用常规方式加载训练数据集
        if is_main_process:
            print("使用常规方式加载训练数据集...")
        
        train_dataset = LazySeamlessInteractionWindowDataset(
            data_path=args.data_path,
            split='train',
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=multi_length_ratios,
            load_video=False,
            load_audio=False,
            max_samples=args.max_samples
        )
    
    # 创建验证数据集
    if args.use_cache and val_cache_path is not None and os.path.exists(val_cache_path):
        # 使用缓存加载验证数据集
        if is_main_process:
            print(f"使用验证集缓存文件: {val_cache_path}")
        
        try:
            from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
            val_dataset = CachedLazySeamlessInteractionWindowDataset(
                data_path=args.data_path,
                split='val',
                window_size=args.window_size,
                window_stride=args.window_stride,
                multi_length_training=[1.0],  # 验证集只使用1.0比例
                load_video=False,
                load_audio=False,
                max_samples=args.max_samples,
                cache_path=val_cache_path
            )
            if len(val_dataset) == 0:
                raise ValueError("验证数据集为空，缓存可能损坏")
        except Exception as e:
            if is_main_process:
                print(f"加载验证集缓存失败: {e}")
                print("回退到常规数据加载方式...")
            args.use_cache = False
    
    if not args.use_cache or val_cache_path is None or not os.path.exists(val_cache_path):
        # 使用常规方式加载验证数据集
        if is_main_process:
            print("使用常规方式加载验证数据集...")
        
        val_dataset = LazySeamlessInteractionWindowDataset(
            data_path=args.data_path,
            split='val',  # 使用val作为验证集
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=[1.0],  # 验证集只使用1.0比例
            load_video=False,
            load_audio=False,
            max_samples=args.max_samples
        )
    
    if is_main_process:
        print(f"使用 {len(train_dataset)} 个训练窗口和 {len(val_dataset)} 个验证窗口进行训练")
    
    # 创建数据加载器
    def custom_collate_fn(batch):
        """自定义批处理函数，处理不同长度的序列"""
        if len(batch) == 0:
            return None

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
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # 如果没有采样器，则shuffle
        num_workers=2,  # 减少工作进程数，避免资源竞争
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=train_sampler,
        persistent_workers=True,  # 保持工作进程活跃，减少进程创建开销
        pin_memory=True,  # 使用固定内存，加速GPU传输
    )
    
      
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,  # 减少工作进程数，避免资源竞争
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=val_sampler,
        persistent_workers=True,  # 保持工作进程活跃，减少进程创建开销
        pin_memory=True,  # 使用固定内存，加速GPU传输
    )
    
    # 加载均值和标准差
    mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
    std_pose = np.load('mean_std/seamless_smplh_std.npy')
    
    # 创建模型
    # 使用传入的参数，如果没有传入则使用默认值
    if not hasattr(args, 'code_dim') or args.code_dim == 128:
        args.code_dim = 128  # 保持与参数解析器一致
    if not hasattr(args, 'width') or args.width == 512:
        args.width = 512  # 默认值
    if not hasattr(args, 'depth') or args.depth == 3:
        args.depth = 3  # 默认值
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99
    
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
    )
    
    # 设置设备
    device = torch.device(f"cuda:{args.local_rank}" if is_distributed else "cuda")
    model.cuda(device)
    
    # 使用分布式数据并行（如果是分布式模式）
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False, output_device=args.local_rank)
    
    model.train()
    
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

            # 更新最佳损失
            if loss_motion.item() < best_recons_loss:
                best_recons_loss = loss_motion.item()

        except Exception as e:
            if is_main_process:
                print(f"获取数据批次时出错: {e}")
                import traceback
                traceback.print_exc()
            break
        
        # 更新进度条显示
        if is_main_process:
            pbar.set_postfix({
                'Recons': f'{loss_motion.item():.5f}',
                'PPL': f'{perplexity.item():.2f}',
                'Commit': f'{loss_commit.item():.5f}',
                'Best': f'{best_recons_loss:.5f}'
            })
            # 不需要手动调用pbar.update(1)，因为tqdm会自动更新
        
        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            
            if is_main_process:
                writer.add_scalar('./Train/L1', avg_recons, nb_iter)
                writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
                writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
                
                logger.info(f"Train. Iter {nb_iter}: \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons. {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        
        if nb_iter % args.eval_iter == 0:
            if is_main_process:
                torch.save({'net': model.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
            
            # 使用验证集进行评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            if is_main_process:
                eval_pbar = tqdm(val_loader, desc=f"验证集评估 (迭代 {nb_iter})", leave=False)
            else:
                eval_pbar = val_loader
            
            with torch.no_grad():
                for batch in eval_pbar:
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
                logger.info(f"Validation. Iter {nb_iter}: \t L2 Distance: {avg_l2:.5f}")
                writer.add_scalar('./Val/L2', avg_l2, nb_iter)
            
            model.train()
    
    if is_main_process:
        print("训练完成!")
    
    # 清理分布式训练环境（如果是分布式模式）
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()