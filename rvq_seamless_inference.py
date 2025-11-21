#!/usr/bin/env python3
"""
RVQ-VAE Seamless数据集推理脚本

该脚本用于调用训练好的RVQ-VAE模型重建Seamless数据集的NPZ运动数据。
纯粹专注于推理功能，不包含额外的评估和可视化。

使用方式:
    # 单文件推理
    python rvq_seamless_inference.py --model-path ./outputs/rvq_seamless/RVQVAE_Seamless_whole/net_best.pth --input-path ./datasets/seamless_interaction/improvised/session_0/gesture_001/frame_0000.npz --output-path ./reconstructed_motion.npz

    # 批量目录推理
    python rvq_seamless_inference.py --model-path ./outputs/rvq_seamless/RVQVAE_Seamless_whole/net_best.pth --input-path ./datasets/seamless_interaction/improvised/session_0/ --output-path ./reconstructed_results/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob
import pynvml
from utils import rotation_conversions as rc

# 导入模型和数据配置
from models.vq.model import RVQVAE
from dataloaders.seamless_sep import CustomDataset
from omegaconf import OmegaConf

def setup_gpu(gpu_id):
    """设置GPU设备"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"使用GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device

def get_args_parser():
    """参数解析器"""
    parser = argparse.ArgumentParser(
        description='RVQ-VAE Seamless数据集推理脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必要参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--input-path', type=str, required=True,
                        help='输入NPZ文件或目录路径')
    parser.add_argument('--output-path', type=str, required=True,
                        help='输出NPZ文件或目录路径')

    # 可选参数
    parser.add_argument('--body-part', type=str, default='whole',
                        choices=['whole', 'upper', 'lower', 'hands'],
                        help='身体部位选择')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='推理批次大小')

    return parser.parse_args()

def get_body_mask(body_part):
    """获取身体部位关节掩码"""
    if body_part == "upper":
        # 上半身：13个关节点
        joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    elif body_part == "hands":
        # 手部：30个关节点 (22-51)
        joints = list(range(22, 52))
    elif body_part == "lower":
        # 下半身：9个关节点
        joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    elif body_part == "whole":
        # 全部52个关节点
        joints = list(range(52))
    else:
        raise ValueError(f"不支持的body_part: {body_part}")

    # 构建6D维度掩码
    body_mask = []
    for i in joints:
        body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

    return joints, body_mask

def load_model(model_path, device):
    """加载预训练的RVQ-VAE模型"""
    print(f"正在加载模型: {model_path}")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载checkpoint
    ckpt = torch.load(model_path, map_location='cpu')

    # 从checkpoint中获取模型配置
    # 如果保存时包含args，则使用；否则使用默认配置
    if 'args' in ckpt:
        args = ckpt['args']
    else:
        # 使用训练时的默认配置
        class DefaultArgs:
            def __init__(self):
                self.num_quantizers = 6
                self.shared_codebook = False
                self.quantize_dropout_prob = 0.0  # 推理时关闭dropout

                # 模型架构参数（根据错误信息修正）
                self.code_dim = 128          # 修正为128
                self.output_emb_width = 128  # 修正为128
                self.down_t = 2
                self.stride_t = 2
                self.width = 512
                self.depth = 3
                self.dilation_growth_rate = 3
                self.vq_act = 'relu'
                self.vq_norm = None

                # EMA参数
                self.mu = 0.99  # 默认EMA更新率

                # 其他量化器参数
                self.nb_code = 1024  # 代码本大小
                self.commit = 0.0     # 推理时不需要commitment loss

        args = DefaultArgs()

    # 根据body_part设置输入维度
    dim_pose = len(ckpt.get('body_mask', range(52))) * 6  # 默认使用全部52关节的6D表示

    # 创建模型
    model = RVQVAE(
        args,
        input_width=dim_pose,
        nb_code=args.nb_code if hasattr(args, 'nb_code') else 1024,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm
    )

    # 加载权重
    if 'net' in ckpt:
        model.load_state_dict(ckpt['net'], strict=True)
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        raise KeyError("未找到有效的权重键名，检查checkpoint格式")

    model = model.to(device)
    model.eval()

    print(f"✅ 模型加载成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    return model

def load_seamless_normalization():
    """加载Seamless数据集的归一化参数"""
    mean_pose_path = './mean_std_seamless/seamless_2_312_mean.npy'
    std_pose_path = './mean_std_seamless/seamless_2_312_std.npy'

    if not os.path.exists(mean_pose_path) or not os.path.exists(std_pose_path):
        raise FileNotFoundError("归一化文件不存在，请确保seamless数据集归一化文件在./mean_std_seamless/目录下")

    mean_pose = np.load(mean_pose_path)
    std_pose = np.load(std_pose_path)

    return mean_pose, std_pose

def process_npz_file(npz_path, joints, body_mask, device):
    """处理单个NPZ文件，转换为模型输入格式"""
    try:
        # 加载NPZ文件
        pose_data = np.load(npz_path, allow_pickle=True)

        # 检查必要的字段
        required_fields = ["smplh:global_orient", "smplh:body_pose",
                          "smplh:left_hand_pose", "smplh:right_hand_pose"]

        for field in required_fields:
            if field not in pose_data:
                raise KeyError(f"NPZ文件中缺少必要字段: {field}")

        # 提取姿态数据
        global_orient = pose_data["smplh:global_orient"]  # [N, 3]
        body_pose = pose_data["smplh:body_pose"]          # [N, 21, 3] -> [N, 63]
        left_hand_pose = pose_data["smplh:left_hand_pose"]  # [N, 15, 3] -> [N, 45]
        right_hand_pose = pose_data["smplh:right_hand_pose"] # [N, 15, 3] -> [N, 45]

        # 重塑为2D
        body_pose = body_pose.reshape(body_pose.shape[0], -1)
        left_hand_pose = left_hand_pose.reshape(left_hand_pose.shape[0], -1)
        right_hand_pose = right_hand_pose.reshape(right_hand_pose.shape[0], -1)

        # 组装姿态向量 [global_orient(3) + body(63) + left_hand(45) + right_hand(45)] = 156维
        poses = np.concatenate([
            global_orient,      # 3维
            body_pose,         # 63维
            left_hand_pose,    # 45维
            right_hand_pose    # 45维
        ], axis=1)            # 总计156维

        # 创建完整的52关节掩码映射
        # 52关节对应关系：
        # [0] global_orient
        # [1-21] body_pose (21个关节)
        # [22-36] left_hand_pose (15个关节)
        # [37-51] right_hand_pose (15个关节)

        # 为每个原始维度找到对应的输出维度
        output_indices = []
        for joint_idx in joints:
            if joint_idx == 0:  # global_orient
                output_indices.extend([0, 1, 2])  # 3维
            elif 1 <= joint_idx <= 21:  # body_pose
                body_joint_idx = joint_idx - 1
                for dim in range(3):
                    output_indices.append(3 + body_joint_idx * 3 + dim)  # 3 + (joint_idx-1)*3
            elif 22 <= joint_idx <= 36:  # left_hand_pose
                hand_joint_idx = joint_idx - 22
                for dim in range(3):
                    output_indices.append(66 + hand_joint_idx * 3 + dim)  # 66 + (joint_idx-22)*3
            elif 37 <= joint_idx <= 51:  # right_hand_pose
                hand_joint_idx = joint_idx - 37
                for dim in range(3):
                    output_indices.append(111 + hand_joint_idx * 3 + dim)  # 111 + (joint_idx-37)*3

        # 提取对应的维度
        masked_poses = poses[:, output_indices]  # [N, len(joints)*3]

        print(f"提取的关节数: {len(joints)}, 维度: {masked_poses.shape}")

        # 转换为6D旋转表示
        poses_tensor = torch.from_numpy(masked_poses).float()
        n_frames = poses_tensor.shape[0]
        n_joints = len(joints)

        # 重塑为 (N, J, 3) 格式
        poses_3d = poses_tensor.reshape(n_frames, n_joints, 3)

        # 转换为旋转矩阵
        poses_matrix = rc.axis_angle_to_matrix(poses_3d)

        # 转换为6D表示
        poses_6d = rc.matrix_to_rotation_6d(poses_matrix)

        # 重塑回 (N, J*6) 格式
        poses_6d = poses_6d.reshape(n_frames, -1)  # [N, len(joints)*6]

        return poses_6d.numpy(), pose_data

    except Exception as e:
        print(f"处理NPZ文件时出错 {npz_path}: {str(e)}")
        return None, None

def inference_motion(model, motion_data, body_mask, mean_pose, std_pose, device, batch_size=32):
    """执行运动推理"""
    print(f"开始推理，输入形状: {motion_data.shape}")

    # 应用归一化
    # motion_data的维度应该是 [N, len(joints)*6]，需要映射到312维的归一化参数
    mean_subset = mean_pose[body_mask]
    std_subset = std_pose[body_mask]
    motion_normalized = (motion_data - mean_subset) / std_subset

    # 转换为tensor
    motion_tensor = torch.from_numpy(motion_normalized).float().to(device)

    # 确保数据维度正确 [seq_len, dim]
    if len(motion_tensor.shape) == 2:
        motion_tensor = motion_tensor.unsqueeze(0)  # [1, seq_len, dim]

    # 分批处理长序列
    seq_len = motion_tensor.shape[1]
    dim = motion_tensor.shape[2]

    reconstructed_batches = []

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, seq_len, batch_size):
            end_idx = min(start_idx + batch_size, seq_len)
            batch_data = motion_tensor[:, start_idx:end_idx, :]

            # 模型推理
            output = model(batch_data)
            rec_motion = output['rec_pose']  # [1, batch_len, dim]

            reconstructed_batches.append(rec_motion.cpu().numpy())

    # 合并所有批次
    reconstructed_motion = np.concatenate(reconstructed_batches, axis=1)  # [1, seq_len, dim]
    reconstructed_motion = reconstructed_motion.squeeze(0)  # [seq_len, dim]

    # 反归一化
    mean_subset = mean_pose[body_mask]
    std_subset = std_pose[body_mask]
    reconstructed_motion = reconstructed_motion * std_subset + mean_subset

    print(f"✅ 推理完成，输出形状: {reconstructed_motion.shape}")

    return reconstructed_motion

def save_reconstructed_motion(rec_motion, original_data, output_path, joints, body_mask):
    """保存重建的运动数据为NPZ格式"""
    try:
        # 将6D表示转换回轴角表示
        rec_tensor = torch.from_numpy(rec_motion).float()
        n_frames = rec_tensor.shape[0]
        n_joints = len(joints)

        # 重塑为 (N, J, 6) 格式
        rec_6d = rec_tensor.reshape(n_frames, n_joints, 6)

        # 从6D转换回旋转矩阵
        rec_matrix = rc.rotation_6d_to_matrix(rec_6d)

        # 从旋转矩阵转换回轴角
        rec_axis_angle = rc.matrix_to_axis_angle(rec_matrix)

        # 重塑回 (N, J*3) 格式
        rec_axis_angle = rec_axis_angle.reshape(n_frames, -1).numpy()

        # 重建原始156维的完整姿态
        full_rec_poses = np.zeros((n_frames, 156))

        # 将重建的关节数据放回正确位置
        for i, joint_idx in enumerate(joints):
            if joint_idx == 0:  # global_orient
                full_rec_poses[:, i*3:(i+1)*3] = rec_axis_angle[:, i*3:(i+1)*3]
            elif 1 <= joint_idx <= 21:  # body_pose
                full_rec_poses[:, 3 + (joint_idx-1)*3:3 + joint_idx*3] = rec_axis_angle[:, i*3:(i+1)*3]
            elif 22 <= joint_idx <= 36:  # left_hand_pose
                full_rec_poses[:, 66 + (joint_idx-22)*3:66 + (joint_idx-21)*3] = rec_axis_angle[:, i*3:(i+1)*3]
            elif 37 <= joint_idx <= 51:  # right_hand_pose
                full_rec_poses[:, 111 + (joint_idx-37)*3:111 + (joint_idx-36)*3] = rec_axis_angle[:, i*3:(i+1)*3]

        # 分解回原始字段
        global_orient = full_rec_poses[:, :3]           # [N, 3]
        body_pose = full_rec_poses[:, 3:66]            # [N, 63] -> [N, 21, 3]
        left_hand_pose = full_rec_poses[:, 66:111]     # [N, 45] -> [N, 15, 3]
        right_hand_pose = full_rec_poses[:, 111:156]   # [N, 45] -> [N, 15, 3]

        # 重塑身体部位为3D格式
        body_pose = body_pose.reshape(n_frames, 21, 3)
        left_hand_pose = left_hand_pose.reshape(n_frames, 15, 3)
        right_hand_pose = right_hand_pose.reshape(n_frames, 15, 3)

        # 创建输出数据字典
        output_data = {}

        # 重建后的姿态数据
        output_data["smplh:global_orient"] = global_orient
        output_data["smplh:body_pose"] = body_pose
        output_data["smplh:left_hand_pose"] = left_hand_pose
        output_data["smplh:right_hand_pose"] = right_hand_pose

        # 保留其他原始字段（如果有）
        if original_data is not None:
            for key, value in original_data.items():
                if key not in ["smplh:global_orient", "smplh:body_pose",
                              "smplh:left_hand_pose", "smplh:right_hand_pose"]:
                    output_data[key] = value

        # 保存NPZ文件
        np.savez_compressed(output_path, **output_data)

        print(f"✅ 重建结果已保存到: {output_path}")
        return True

    except Exception as e:
        print(f"保存NPZ文件时出错: {str(e)}")
        return False

def main():
    """主函数"""
    args = get_args_parser()

    # 设置GPU设备
    device = setup_gpu(args.gpu_id)

    # 获取身体部位配置
    joints, body_mask = get_body_mask(args.body_part)
    print(f"身体部位: {args.body_part}, 关节数量: {len(joints)}, 维度: {len(body_mask)}")

    # 加载归一化参数
    print("加载归一化参数...")
    mean_pose, std_pose = load_seamless_normalization()

    # 加载模型
    model = load_model(args.model_path, device)

    # 确定输入文件列表
    input_path = Path(args.input_path)
    if input_path.is_file() and input_path.suffix == '.npz':
        npz_files = [input_path]
    elif input_path.is_dir():
        npz_files = list(input_path.glob("**/*.npz"))
        print(f"找到 {len(npz_files)} 个NPZ文件")
    else:
        raise FileNotFoundError(f"输入路径不存在或不是有效的NPZ文件: {args.input_path}")

    # 设置输出路径
    output_path = Path(args.output_path)
    if output_path.is_file() or (not output_path.exists() and input_path.is_file()):
        # 单文件输出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_files = [output_path]
    else:
        # 目录输出
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = []
        for npz_file in npz_files:
            rel_path = npz_file.relative_to(input_path)
            out_file = output_path / f"reconstructed_{rel_path.name}"
            output_files.append(out_file)

    # 批量处理文件
    success_count = 0
    for i, (npz_file, out_file) in enumerate(zip(npz_files, output_files)):
        print(f"\n处理文件 {i+1}/{len(npz_files)}: {npz_file}")

        # 处理NPZ文件
        motion_data, original_data = process_npz_file(npz_file, joints, body_mask, device)
        if motion_data is None:
            print(f"跳过文件 {npz_file}")
            continue

        # 执行推理
        reconstructed_motion = inference_motion(
            model, motion_data, body_mask, mean_pose, std_pose, device, args.batch_size
        )

        # 保存结果
        if save_reconstructed_motion(reconstructed_motion, original_data, out_file, joints, body_mask):
            success_count += 1

    print(f"\n🎉 推理完成！成功处理 {success_count}/{len(npz_files)} 个文件")

if __name__ == "__main__":
    main()