#!/usr/bin/env python3
"""
7个4090显卡最大化显存使用训练脚本 (GPU 1-7)
使用方法: python train_7x4090.py
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='8个4090显卡最大化显存使用训练脚本')
    parser.add_argument('--data_path', type=str, default='/datasets/seamless_interaction/improvised', help='数据路径')
    parser.add_argument('--out_dir', type=str, default='experiments', help='输出目录')
    parser.add_argument('--exp_name', type=str, default='rvq_seamless_8x4090', help='实验名称')
    parser.add_argument('--gpus', type=int, default=7, help='Number of GPUs to use (GPU 1-7)')
    parser.add_argument('--master_port', type=int, default=29500, help='主端口')
    
    # 模型参数 - 基于RVQ-VAE架构文档配置
    parser.add_argument('--nb_code', type=int, default=1024, help='码本大小（与架构文档一致）')
    parser.add_argument('--code_dim', type=int, default=128, help='码本维度（与架构文档一致）')
    parser.add_argument('--down_t', type=int, default=2, help='下采样层数（与架构文档一致）')
    parser.add_argument('--stride_t', type=int, default=2, help='时间步长（与架构文档一致）')
    parser.add_argument('--width', type=int, default=512, help='网络宽度（与架构文档一致）')
    parser.add_argument('--depth', type=int, default=3, help='网络深度（与架构文档一致）')
    parser.add_argument('--dilation_growth_rate', type=int, default=3, help='膨胀增长率（与架构文档一致）')
    parser.add_argument('--vq_act', type=str, default='relu', help='VQ激活函数（与架构文档一致）')
    parser.add_argument('--num_quantizers', type=int, default=6, help='量化器数量，控制RVQ的层数（与架构文档一致）')
    parser.add_argument('--shared_codebook', action='store_true', help='是否共享码本')
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.5, help='量化器dropout概率')
    
    # 显存优化参数
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='是否使用梯度检查点')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 训练参数 - 基于RVQ-VAE架构文档配置
    parser.add_argument('--batch_size', type=int, default=80, help='每个GPU的批次大小')
    parser.add_argument('--total_iter', type=int, default=300000, help='总迭代次数（与架构文档一致）')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率（与架构文档一致）')
    parser.add_argument('--lr_scheduler', type=str, default='[50000, 200000, 400000]', help='学习率调度器里程碑')
    parser.add_argument('--gamma', type=float, default=0.05, help='学习率衰减')
    parser.add_argument('--commit', type=float, default=0.02, help='提交损失（与架构文档一致）')
    parser.add_argument('--loss_vel', type=float, default=0.5, help='速度损失（与架构文档一致）')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='重建损失函数类型')
    
    # 数据参数 - 基于RVQ-VAE架构文档配置
    parser.add_argument('--window_size', type=int, default=128, help='窗口大小（与架构文档一致）')
    parser.add_argument('--window_stride', type=int, default=64, help='窗口步长（与架构文档一致）')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数，设置为None使用整个数据集')
    
    # 输出参数
    parser.add_argument('--print_iter', type=int, default=100, help='打印间隔')
    parser.add_argument('--eval_iter', type=int, default=1000, help='评估间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(1, args.gpus+1))
    env['NCCL_DEBUG'] = 'INFO'
    env['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
    
    # 运行训练命令
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.gpus}',
        f'--master_port={args.master_port}',
        'rvq_seamless_multi_gpu.py',
        '--data_path', args.data_path,
        '--out_dir', args.out_dir,
        '--exp_name', args.exp_name,
        '--nb_code', str(args.nb_code),
        '--code_dim', str(args.code_dim),
        '--down_t', str(args.down_t),
        '--stride_t', str(args.stride_t),
        '--width', str(args.width),
        '--depth', str(args.depth),
        '--dilation_growth_rate', str(args.dilation_growth_rate),
        '--vq_act', args.vq_act,
        '--num_quantizers', str(args.num_quantizers),
        '--quantize_dropout_prob', str(args.quantize_dropout_prob),
        '--batch_size', str(args.batch_size),
        '--total_iter', str(args.total_iter),
        '--lr', str(args.lr),
        '--lr_scheduler', args.lr_scheduler,
        '--gamma', str(args.gamma),
        '--commit', str(args.commit),
        '--loss_vel', str(args.loss_vel),
        '--recons_loss', args.recons_loss,
        '--window_size', str(args.window_size),
        '--window_stride', str(args.window_stride),
        '--print_iter', str(args.print_iter),
        '--eval_iter', str(args.eval_iter),
        '--seed', str(args.seed),
        '--max_grad_norm', str(args.max_grad_norm)
    ]
    
    # 添加max_samples参数（如果不为None）
    if args.max_samples is not None:
        cmd.extend(['--max_samples', str(args.max_samples)])
    
    # 添加shared_codebook参数（如果设置了）
    if args.shared_codebook:
        cmd.append('--shared_codebook')
    
    # 添加混合精度训练参数（如果设置了）
    if args.mixed_precision:
        cmd.append('--mixed_precision')
    
    # 添加梯度检查点参数（如果设置了）
    if args.gradient_checkpointing:
        cmd.append('--gradient_checkpointing')
    
    # 打印命令
    print("运行命令:")
    print(' '.join(cmd))
    print(f"总批次大小: {args.batch_size * args.gpus}")
    print(f"预计显存使用: 每个GPU约 {24 * 0.9:.1f}GB (90%利用率)")
    
    # 运行命令
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()