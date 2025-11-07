#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# 导入模型和数据集
from models.rqvae import RVQVAE
from utils.losses import ReConsLoss
from utils.seamless_dataset import LazySeamlessInteractionWindowDataset

def get_logger(log_dir):
    """创建日志记录器"""
    logger = logging.getLogger('Training')
    logger.setLevel(logging.INFO)
    
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
    
    parser = argparse.ArgumentParser(description='RVQ-VAE无缝交互训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/seamless_motions', help='数据路径')
    parser.add_argument('--body_part', type=str, default='whole', choices=['upper', 'lower', 'hands', 'whole'], help='身体部位')
    
    # 模型参数
    parser.add_argument('--nb_code', type=int, default=512, help='码本大小')
    parser.add_argument('--code_dim', type=int, default=256, help='码本维度')
    parser.add_argument('--down_t', type=int, default=2, help='下采样层数')
    parser.add_argument('--stride_t', type=int, default=2, help='时间步长')
    parser.add_argument('--width', type=int, default=512, help='网络宽度')
    parser.add_argument('--depth', type=int, default=3, help='网络深度')
    parser.add_argument('--dilation_growth_rate', type=int, default=3, help='膨胀率增长')
    parser.add_argument('--vq_act', type=str, default='relu', help='VQ激活函数')
    parser.add_argument('--vq_norm', type=str, default=None, help='VQ归一化方法')
    parser.add_argument('--mu', type=float, default=0.99, help='EMA动量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--total_iter', type=int, default=10000, help='总迭代次数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--lr_scheduler', type=list, default=[5000, 8000], help='学习率调度器里程碑')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--commit', type=float, default=0.02, help='commitment loss权重')
    parser.add_argument('--loss_vel', type=float, default=0.0, help='速度损失权重')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='重建损失类型')
    
    # 数据参数
    parser.add_argument('--window_size', type=int, default=60, help='窗口大小')
    parser.add_argument('--window_stride', type=int, default=30, help='窗口步长')
    parser.add_argument('--max_samples', type=int, default=5000, help='最大样本数')
    
    # 输出参数
    parser.add_argument('--out_dir', type=str, default='experiments', help='输出目录')
    parser.add_argument('--exp_name', type=str, default='rvq_seamless', help='实验名称')
    parser.add_argument('--print_iter', type=int, default=100, help='打印间隔')
    parser.add_argument('--eval_iter', type=int, default=1000, help='评估间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 分布式训练参数
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名')
    parser.add_argument('--rank', type=int, default=0, help='全局进程排名')
    parser.add_argument('--world_size', type=int, default=1, help='总进程数')
    
    return parser

def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    
    # 创建数据集
    train_dataset = LazySeamlessInteractionWindowDataset(
        data_path='data/seamless_motions',
        split='train',
        window_size=60,
        window_stride=30,
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=10  # 限制为10个样本
    )
    
    print(f"训练数据集大小: {len(train_dataset)}")
    
    # 创建数据加载器
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次形状: {batch['pose'].shape}")
        print(f"掩码形状: {batch['mask'].shape}")
        print(f"ID: {batch['id']}")
        break
    
    print("数据加载测试完成!")

def test_minimal_training():
    """测试最小训练流程"""
    print("测试最小训练流程...")
    
    # 创建数据集
    train_dataset = LazySeamlessInteractionWindowDataset(
        data_path='data/seamless_motions',
        split='train',
        window_size=60,
        window_stride=30,
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=10  # 限制为10个样本
    )
    
    # 创建数据加载器
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    # 获取一个批次数据
    for batch in train_loader:
        test_motion = batch['pose'].cuda()  # [2, 60, 156]
        break
    
    # 创建模型
    class Args:
        def __init__(self):
            self.nb_code = 512
            self.code_dim = 256
            self.down_t = 2
            self.stride_t = 2
            self.width = 512
            self.depth = 3
            self.dilation_growth_rate = 3
            self.vq_act = 'relu'
            self.vq_norm = None
            self.mu = 0.99  # 添加mu参数
    
    args = Args()
    pose_dim = 156  # 完整姿态维度
    
    model = RVQVAE(
        args,
        pose_dim,
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
    
    model.cuda()
    model.train()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    
    # 创建损失函数
    Loss = ReConsLoss('l1_smooth')
    
    print("开始训练循环...")
    # 简单训练循环
    for iter_idx in range(10):  # 只训练10步
        # 转置维度以适应网络输入 (bs, seq_len, dim_pose) -> (bs, dim_pose, seq_len)
        input_motion = test_motion.transpose(1, 2)  # [2, 60, 39] -> [2, 39, 60]
        
        # 前向传播
        pred_motion, loss_commit, perplexity = model(input_motion).values()
        
        # 转置回 (bs, seq_len, dim_pose)
        pred_motion = pred_motion.transpose(1, 2)
        
        # 计算损失
        loss_motion = Loss.my_forward(pred_motion, test_motion, list(range(pose_dim)))
        loss = loss_motion + 0.02 * loss_commit
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Iter {iter_idx}: Loss {loss.item():.5f}, Motion Loss {loss_motion.item():.5f}, Commit Loss {loss_commit.item():.5f}, PPL {perplexity.item():.2f}")
    
    print("最小训练测试完成!")
    return True


def main():
    import argparse
    args = get_args_parser()
    
    # 设置分布式训练环境
    is_distributed = args.world_size > 1
    if is_distributed:
        # 初始化分布式训练环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 初始化进程组
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        
        # 只有主进程打印信息
        is_main_process = args.rank == 0
    else:
        is_main_process = True
    
    torch.manual_seed(args.seed)
    
    # 创建输出目录（只有主进程创建）
    if is_main_process:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = get_logger(args.out_dir)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # 创建数据集
    if is_main_process:
        print("Building dataset...")
    
    # 创建窗口数据集
    train_dataset = LazySeamlessInteractionWindowDataset(
        data_path=args.data_path,
        split='train',
        window_size=args.window_size,
        window_stride=args.window_stride,
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=args.max_samples
    )
    test_dataset = LazySeamlessInteractionWindowDataset(
        data_path=args.data_path,
        split='test',
        window_size=args.window_size,
        window_stride=args.window_stride,
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=args.max_samples // 5  # 测试集大小为训练集的1/5
    )
    
    if is_main_process:
        print(f"使用 {len(train_dataset)} 个训练窗口和 {len(test_dataset)} 个测试窗口")
    
    # 创建数据加载器
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
    
    # 使用分布式采样器
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=train_sampler
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn,
        sampler=test_sampler
    )
    
    # 加载均值和标准差
    mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
    std_pose = np.load('mean_std/seamless_smplh_std.npy')
    
    # 创建关节掩码
    mask = []
    if args.body_part == 'upper':
        # 上半身关节点
        joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        for i in joints:
            mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    elif args.body_part == 'hands':
        # 手部关节点
        joints = list(range(25, 55))
        for i in joints:
            mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    elif args.body_part == 'lower':
        # 下半身关节点
        joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        for i in joints:
            mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    else:  # whole
        # 全身关节点
        joints = list(range(0, 22)) + list(range(25, 55))
        for i in joints:
            mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    
    # 计算姿态维度
    pose_dim = len(mask) // 6 * 3  # 6D表示转换为3D表示
    
    # 创建模型
    args.code_dim = 256
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99
    
    # 输入维度是156（完整姿态），而不是pose_dim（应用掩码后的维度）
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
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 使用分布式数据并行
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True, output_device=args.local_rank)
    
    model.train()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    
    # 创建损失函数
    Loss = ReConsLoss(args.recons_loss)
    
    # 训练循环
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    
    if is_main_process:
        print("开始训练循环...")
        print(f"开始训练，总共 {args.total_iter} 次迭代")
        
        # 创建进度条
        pbar = tqdm(range(1, args.total_iter + 1), desc="训练进度", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    else:
        pbar = range(1, args.total_iter + 1)
    
    # 记录最佳损失
    best_recons_loss = float('inf')
    
    for nb_iter in pbar:
        # 设置分布式采样器的epoch
        if is_distributed:
            train_sampler.set_epoch(nb_iter)
        
        for batch in train_loader:
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
            
            # 暂时不应用关节掩码，避免索引越界问题
            # gt_motion = gt_motion[..., mask]  # (bs, seq_len, dim_pose)
            
            # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
            pred_motion, loss_commit, perplexity = model(gt_motion).values()
            
            loss_motion = Loss.my_forward(pred_motion, gt_motion, list(range(gt_motion.shape[2])))  # 使用实际维度而不是pose_dim
            loss_vel = 0  # 暂时不计算速度损失
            
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            avg_recons += loss_motion.item()
            avg_perplexity += perplexity.item()
            avg_commit += loss_commit.item()
            
            # 更新最佳损失
            if loss_motion.item() < best_recons_loss:
                best_recons_loss = loss_motion.item()
            
            break  # 每次迭代只处理一个批次
        
        # 更新进度条显示
        if is_main_process:
            pbar.set_postfix({
                'Recons': f'{loss_motion.item():.5f}',
                'PPL': f'{perplexity.item():.2f}',
                'Commit': f'{loss_commit.item():.5f}',
                'Best': f'{best_recons_loss:.5f}'
            })
            pbar.update(1)  # 确保每次迭代都更新进度条
        
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
            
            # 评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            if is_main_process:
                eval_pbar = tqdm(test_loader, desc=f"评估 (迭代 {nb_iter})", leave=False)
            else:
                eval_pbar = test_loader
            
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
                    
                    # 暂时不应用关节掩码，避免索引越界问题
                    # gt_motion = gt_motion[..., mask]  # (bs, seq_len, dim_pose)
                    
                    # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
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
                logger.info(f"Evaluation. Iter {nb_iter}: \t L2 Distance: {avg_l2:.5f}")
                writer.add_scalar('./Eval/L2', avg_l2, nb_iter)
            
            model.train()
    
    if is_main_process:
        print("训练完成!")
    
    # 清理分布式训练环境
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()