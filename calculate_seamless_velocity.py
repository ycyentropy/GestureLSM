#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 seamless_interaction 数据集的运动数据平均速度
仿照 final_sep.py 中 calculate_mean_velocity 方法实现
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import smplx

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Calculate mean velocity for seamless_interaction dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--smplx_model_path', type=str, 
                        default='/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models/',
                        help='Path to SMPLX models')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    return parser.parse_args()

def load_smplx_model(model_path, device):
    """加载 SMPLX 模型"""
    print(f"Loading SMPLX model from {model_path}...")
    smplx_model = smplx.create(
        model_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
    )
    return smplx_model.to(device).eval()

def find_npz_files(directory):
    """查找目录下所有 npz 文件"""
    npz_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return npz_files

def process_npz_file(file_path, smplx_model, batch_size, device):
    """处理单个 npz 文件，生成关节点数据"""
    # 加载数据
    data = np.load(file_path, allow_pickle=True)
    
    # 提取 smplh 相关数据
    body_pose = data['smplh:body_pose']
    global_orient = data['smplh:global_orient']
    left_hand_pose = data['smplh:left_hand_pose']
    right_hand_pose = data['smplh:right_hand_pose']
    translation = data['smplh:translation']
    is_valid = data['smplh:is_valid']
    
    # 使用 is_valid 过滤数据
    valid_indices = np.where(is_valid)[0]
    if len(valid_indices) == 0:
        return None
    
    # 过滤有效数据
    body_pose = body_pose[valid_indices]
    global_orient = global_orient[valid_indices]
    left_hand_pose = left_hand_pose[valid_indices]
    right_hand_pose = right_hand_pose[valid_indices]
    translation = translation[valid_indices]
    
    n_frames = len(valid_indices)
    
    # 补零处理：添加缺失的字段
    # smplh 缺少 jaw_pose, leye_pose, reye_pose
    jaw_pose = np.zeros((n_frames, 1, 3), dtype=np.float32)
    leye_pose = np.zeros((n_frames, 1, 3), dtype=np.float32)
    reye_pose = np.zeros((n_frames, 1, 3), dtype=np.float32)
    
    # 添加固定的 betas（smplh 没有提供 betas，使用零值）
    betas = np.zeros((n_frames, 300), dtype=np.float32)
    # 添加固定的 expressions（smplh 没有提供 expressions，使用零值）
    expressions = np.zeros((n_frames, 100), dtype=np.float32)
    
    # 转换为 torch 张量
    body_pose = torch.from_numpy(body_pose).to(device).float()
    global_orient = torch.from_numpy(global_orient).to(device).float()
    left_hand_pose = torch.from_numpy(left_hand_pose).to(device).float()
    right_hand_pose = torch.from_numpy(right_hand_pose).to(device).float()
    translation = torch.from_numpy(translation).to(device).float()
    jaw_pose = torch.from_numpy(jaw_pose).to(device).float()
    leye_pose = torch.from_numpy(leye_pose).to(device).float()
    reye_pose = torch.from_numpy(reye_pose).to(device).float()
    betas = torch.from_numpy(betas).to(device).float()
    expressions = torch.from_numpy(expressions).to(device).float()
    
    # 生成关节点 - 智能分批处理，根据序列长度调整batch_size
    all_joints = []
    
    # 对于长序列，使用较小的batch_size，避免显存不足
    seq_length = betas.shape[0]
    
    # # 动态调整batch_size，确保显存使用合理（更加保守）
    # if seq_length > 1024:
    #     effective_batch_size = 128
    # elif seq_length > 512:
    #     effective_batch_size = 256
    # else:
    #     effective_batch_size = min(seq_length, batch_size)
    
    # 分批次处理长序列
    for start_idx in range(0, seq_length, batch_size):
        end_idx = min(start_idx + batch_size, seq_length)
        
        with torch.no_grad():
            # 使用 smplx 模型生成关节点
            outputs = smplx_model(
                betas=betas[start_idx:end_idx],
                transl=translation[start_idx:end_idx],
                expression=expressions[start_idx:end_idx],
                jaw_pose=jaw_pose[start_idx:end_idx].reshape(-1, 3),
                global_orient=global_orient[start_idx:end_idx],
                body_pose=body_pose[start_idx:end_idx].reshape(-1, 63),
                left_hand_pose=left_hand_pose[start_idx:end_idx].reshape(-1, 45),
                right_hand_pose=right_hand_pose[start_idx:end_idx].reshape(-1, 45),
                return_verts=True,
                return_joints=True,
                leye_pose=leye_pose[start_idx:end_idx].reshape(-1, 3),
                reye_pose=reye_pose[start_idx:end_idx].reshape(-1, 3),
            )
            
            # 提取前55个关节点，形状：(batch_size, 55, 3)
            batch_joints = outputs['joints'][:, :55, :]
            all_joints.append(batch_joints.detach().cpu())  # 转移到CPU并分离梯度
            
            # 清理当前批次使用的显存
            del outputs, batch_joints
            torch.cuda.empty_cache()
    
    # 合并所有批次的关节点并保持在CPU上
    joints = torch.cat(all_joints, dim=0)
    
    return joints

def calculate_velocity(joints, fps=30, device='cuda:0'):
    """计算速度序列"""
    dt = 1.0 / fps
    
    # 将关节点数据临时转移到GPU上进行快速计算
    joints_gpu = joints.to(device)
    
    # 转换为 (55, n_frames, 3) 格式，便于计算
    joints_gpu = joints_gpu.permute(1, 0, 2)
    
    # 前向差分计算初始速度：(t+1 - t) / dt
    init_vel = (joints_gpu[:, 1:2] - joints_gpu[:, :1]) / dt
    
    # 二阶差分计算中间速度：(t+1 - t-1) / 2dt
    middle_vel = (joints_gpu[:, 2:] - joints_gpu[:, 0:-2]) / (2 * dt)
    
    # 后向差分计算最终速度：(t - t-1) / dt
    final_vel = (joints_gpu[:, -1:] - joints_gpu[:, -2:-1]) / dt
    
    # 合并速度序列，形状：(55, n_frames, 3)
    vel_seq_gpu = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    
    # 转换回 (n_frames, 55, 3) 格式并转移到CPU
    vel_seq = vel_seq_gpu.permute(1, 0, 2).cpu()
    
    # 清理GPU上的临时数据
    del joints_gpu, init_vel, middle_vel, final_vel, vel_seq_gpu
    torch.cuda.empty_cache()
    
    return vel_seq

def calculate_average_velocity(vel_seq_list):
    """计算平均速度"""
    if not vel_seq_list:
        return None
    
    # 合并所有速度序列
    all_velocities = torch.cat(vel_seq_list, dim=0)
    
    # 计算每个关节点的速度大小（欧几里得范数）
    velocity_magnitudes = torch.norm(all_velocities, dim=2)
    
    # 计算每个关节点的平均速度
    avg_velocities = torch.mean(velocity_magnitudes, dim=0)
    
    return avg_velocities

def save_results(avg_velocities, output_dir):
    """保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存平均速度
    avg_velocities_np = avg_velocities.cpu().numpy()
    np.save(os.path.join(output_dir, 'average_velocity.npy'), avg_velocities_np)
    
    # 保存统计信息
    stats = {
        'mean': float(np.mean(avg_velocities_np)),
        'std': float(np.std(avg_velocities_np)),
        'min': float(np.min(avg_velocities_np)),
        'max': float(np.max(avg_velocities_np)),
        'median': float(np.median(avg_velocities_np)),
        'total_joints': len(avg_velocities_np)
    }
    
    with open(os.path.join(output_dir, 'velocity_stats.txt'), 'w') as f:
        f.write("Joint Velocity Statistics\n")
        f.write("=" * 30 + "\n")
        for i, vel in enumerate(avg_velocities_np):
            f.write(f"Joint {i:2d}: {vel:.6f}\n")
        f.write("\nOverall Statistics:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Results saved to {output_dir}")
    print(f"Average velocity shape: {avg_velocities_np.shape}")
    print(f"Overall mean velocity: {stats['mean']:.6f}")
    print(f"Overall std velocity: {stats['std']:.6f}")

def main():
    """主函数"""
    args = parse_args()
    
    # 加载 smplx 模型
    smplx_model = load_smplx_model(args.smplx_model_path, args.device)
    
    # 查找所有 npz 文件
    npz_files = find_npz_files(args.input_dir)
    print(f"Found {len(npz_files)} npz files")
    
    # 处理所有文件，生成关节点数据
    all_vel_sequences = []
    
    for file_path in tqdm(npz_files, desc="Processing files"):
        # 处理单个文件
        joints = process_npz_file(file_path, smplx_model, args.batch_size, args.device)
        
        if joints is None:
            continue
        
        # 计算速度序列
        vel_seq = calculate_velocity(joints, device=args.device)
        all_vel_sequences.append(vel_seq)
    
    if not all_vel_sequences:
        print("No valid data found in the dataset")
        return
    
    # 计算平均速度
    avg_velocities = calculate_average_velocity(all_vel_sequences)
    
    # 保存结果
    save_results(avg_velocities, args.output_dir)

if __name__ == "__main__":
    main()
