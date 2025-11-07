#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import sys
import numpy as np
import warnings
import argparse
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vq.model import RVQVAE
from dataloaders.seamless_interaction import SeamlessInteractionDataset
from lazy_window_dataset import LazySeamlessInteractionWindowDataset
from dataloaders.data_tools import joints_list

warnings.filterwarnings('ignore')

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


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
    parser = argparse.ArgumentParser(description='RVQ-VAE training for Seamless Interaction',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='seamless_interaction', help='dataset directory')
    parser.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction', help='dataset path')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='窗口大小(帧数)')
    parser.add_argument('--window-stride', type=int, default=20, help='窗口步长(帧数)')
    parser.add_argument('--multi_length_training', action='store_true', help='是否启用多长度训练')
    parser.add_argument('--body-part', type=str, default='whole', help='body part to train')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数，设置为None使用整个数据集')
    
    ## optimization
    parser.add_argument('--total-iter', default=10000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=500, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[30000, 40000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='activation function')
    parser.add_argument('--vq-norm', type=str, default=None, help='normalization method')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'], help="quantizer type")
    parser.add_argument("--num_quantizers", type=int, default=6, help="number of quantizers")
    parser.add_argument("--shared_codebook", type=bool, default=False, help="whether to use shared codebook")
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.2, help="quantize dropout probability")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='outputs/rvqvae_seamless', help='output directory')
    parser.add_argument('--exp-name', type=str, default='RVQVAE_Seamless', help='name of the experiment')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=2000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    
    return parser.parse_args()


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def collate_fn(batch):
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


def test_data_loading():
    """测试数据加载是否正常"""
    try:
        # 尝试加载一个数据文件
        data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train/0000/0000/V00_S0039_I00000581_P0061.npz"
        print(f"尝试加载数据文件: {data_path}")
        
        if os.path.exists(data_path):
            npz_data = np.load(data_path)
            print(f"成功加载数据文件，包含的键: {list(npz_data.keys())}")
            
            # 检查关键数据
            if 'smplh:body_pose' in npz_data:
                print(f"body_pose shape: {npz_data['smplh:body_pose'].shape}")
            if 'boxes_and_keypoints:keypoints' in npz_data:
                print(f"keypoints shape: {npz_data['boxes_and_keypoints:keypoints'].shape}")
            
            return True
        else:
            print(f"数据文件不存在: {data_path}")
            return False
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return False


def test_minimal_training():
    """使用最小数据集进行训练测试"""
    print("开始最小训练测试...")
    
    # 创建简单的测试数据
    batch_size = 2
    seq_len = 60  # 2秒 * 30fps
    pose_dim = len(joints_list["beat_smplx_upper"]) * 3  # 上半身关节点数 * 3个坐标值
    
    # 生成随机姿态数据 (batch_size, pose_dim, seq_len) - 直接生成模型期望的维度顺序
    test_motion = torch.randn(batch_size, pose_dim, seq_len).cuda()  # [2, 39, 60]
    test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
    
    print(f"测试数据形状: {test_motion.shape}")
    
    # 创建网络
    args = argparse.Namespace()
    args.num_quantizers = 6
    args.shared_codebook = False
    args.quantize_dropout_prob = 0.2
    args.nb_code = 512
    args.code_dim = 256
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99  # 添加mu参数
    
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
    
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # 创建数据集
    print("Building dataset...")
    # 创建窗口数据集
    train_dataset = LazySeamlessInteractionWindowDataset(
        data_path=args.data_path,
        split='train',
        window_size=args.window_size,
        window_stride=args.window_stride,
        multi_length_training=args.multi_length_training,  # 添加多长度训练参数
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=args.max_samples  # 使用参数中的max_samples值
    )
    # 创建验证集数据集
    val_dataset = LazySeamlessInteractionWindowDataset(
        data_path=args.data_path,
        split='dev',  # 使用dev作为验证集
        window_size=args.window_size,
        window_stride=args.window_stride,
        multi_length_training=args.multi_length_training,  # 添加多长度训练参数
        load_video=False,  # 不加载视频
        load_audio=False,   # 不加载音频
        max_samples=args.max_samples  # 使用相同的max_samples值，但实际样本数量取决于dev split中的文件数量
    )
    
    print(f"使用 {len(train_dataset)} 个训练窗口和 {len(val_dataset)} 个验证窗口进行训练")
    
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    # 创建验证集数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        drop_last=True,
        collate_fn=custom_collate_fn
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
    
    model.cuda()
    model.train()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    
    # 创建损失函数
    Loss = ReConsLoss(args.recons_loss)
    
    # 训练循环
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    
    print("开始训练循环...")
    print(f"开始训练，总共 {args.total_iter} 次迭代")
    
    # 创建进度条
    pbar = tqdm(range(1, args.total_iter + 1), desc="训练进度", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    # 记录最佳损失
    best_recons_loss = float('inf')
    
    for nb_iter in pbar:
        for batch in train_loader:
            gt_motion = batch['pose'].cuda()  # (bs, seq_len, 156)
            batch_mask = batch['mask'].cuda()  # (bs, seq_len)
            
            # 标准化
            # 均值和标准差是针对完整姿态的(330,)，但我们的数据是(batch_size, seq_len, 156)
            # 我们需要只使用对应部分的均值和标准差
            if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                # 如果均值是完整姿态(330)，我们只使用前156个值
                mean_pose_tensor = torch.from_numpy(mean_pose[:156]).cuda()
                std_pose_tensor = torch.from_numpy(std_pose[:156]).cuda()
            else:
                mean_pose_tensor = torch.from_numpy(mean_pose).cuda()
                std_pose_tensor = torch.from_numpy(std_pose).cuda()
            
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
            
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter}: \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons. {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        
        if nb_iter % args.eval_iter == 0:
            torch.save({'net': model.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
            
            # 使用验证集进行评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            eval_pbar = tqdm(val_loader, desc=f"验证集评估 (迭代 {nb_iter})", leave=False)
            
            with torch.no_grad():
                for batch in eval_pbar:
                    gt_motion = batch['pose'].cuda()  # (bs, seq_len, 156)
                    batch_mask = batch['mask'].cuda()  # (bs, seq_len)
                    
                    # 标准化
                    if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                        # 如果均值是完整姿态(330)，我们只使用前156个值
                        mean_pose_tensor = torch.from_numpy(mean_pose[:156]).cuda()
                        std_pose_tensor = torch.from_numpy(std_pose[:156]).cuda()
                    else:
                        mean_pose_tensor = torch.from_numpy(mean_pose).cuda()
                        std_pose_tensor = torch.from_numpy(std_pose).cuda()
                    
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
                    eval_pbar.set_postfix({'L2': f'{l2_batch:.5f}'})
            
            avg_l2 = total_l2 / num_batches
            logger.info(f"Validation. Iter {nb_iter}: \t L2 Distance: {avg_l2:.5f}")
            writer.add_scalar('./Val/L2', avg_l2, nb_iter)
            
            model.train()
    
    print("训练完成!")


if __name__ == "__main__":
    main()