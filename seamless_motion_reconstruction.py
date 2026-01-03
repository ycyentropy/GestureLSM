#!/usr/bin/env python3
"""
Seamless数据集运动重建推理脚本

该脚本用于seamless数据集单个文件的运动重建，支持：
1. 读取原始运动数据NPZ文件
2. 对原始数据进行归一化处理
3. 分割数据为三种人体部位（upper、lower、hands）
4. 读取对应的三种预训练模型进行推理
5. 拼接所有部位的预测结果
6. 经过反归一化得到重建后的运动数据NPZ文件

"""

import os
import numpy as np
import torch
import argparse
import logging
import sys
from tqdm import tqdm

# 项目内部导入
from models.vq.model import RVQVAE
import utils.rotation_conversions as rc


def validate_input_file(npz_path):
    """验证输入NPZ文件的完整性"""
    required_keys = [
        "smplh:global_orient",
        "smplh:body_pose",
        "smplh:left_hand_pose",
        "smplh:right_hand_pose",
        "smplh:translation"
    ]

    try:
        data = np.load(npz_path)
        for key in required_keys:
            if key not in data:
                raise ValueError(f"缺少必需字段: {key}")

        # 检查数据维度
        seq_len = data["smplh:global_orient"].shape[0]
        for key in required_keys:
            if data[key].shape[0] != seq_len:
                raise ValueError(f"字段{key}的序列长度不一致")

        return True, data
    except Exception as e:
        return False, str(e)


def validate_models_exist(model_paths):
    """验证所有模型文件是否存在"""
    for body_part, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{body_part}模型文件不存在: {path}")


def setup_logger(output_dir):
    """设置日志记录器"""
    logger = logging.getLogger('SeamlessReconstruction')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(output_dir, 'reconstruction.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def axis_angle_to_6d(poses_axis_angle):
    """将轴角表示转换为6D旋转表示"""
    # poses_axis_angle: (seq_len, 156) 轴角表示
    # 首先重塑为 (seq_len, 52, 3)
    poses_reshaped = poses_axis_angle.reshape(-1, 52, 3)

    # 转换为torch tensor
    poses_tensor = torch.from_numpy(poses_reshaped)

    # 转换为旋转矩阵
    poses_matrix = rc.axis_angle_to_matrix(poses_tensor)

    # 转换为6D表示
    poses_6d = rc.matrix_to_rotation_6d(poses_matrix)

    return poses_6d.reshape(-1, 312).numpy()  # (seq_len, 312)


def d6_to_axis_angle(poses_6d):
    """将6D旋转表示转换回轴角表示"""
    # poses_6d: (seq_len, 312) 6D表示
    poses_6d_reshaped = poses_6d.reshape(-1, 52, 6)

    # 转换为torch tensor
    poses_6d_tensor = torch.from_numpy(poses_6d_reshaped)

    poses_matrix = rc.rotation_6d_to_matrix(poses_6d_tensor)
    poses_axis_angle = rc.matrix_to_axis_angle(poses_matrix)

    # 转换回numpy
    return poses_axis_angle.numpy().reshape(-1, 156)  # (seq_len, 156)


def assemble_seamless_pose(global_orient, body_pose, left_hand_pose, right_hand_pose):
    """组装seamless姿态向量为轴角表示"""
    return np.concatenate([
        global_orient,      # (N, 3)
        body_pose,         # (N, 63)
        left_hand_pose,     # (N, 45)
        right_hand_pose      # (N, 45)
    ], axis=1)  # (N, 156)


def split_to_body_parts(pose_6d, upper_mask, lower_mask, hand_mask):
    """将6D表示分割为不同身体部位"""
    return (
        pose_6d[:, upper_mask],  # 上半身 (N, 78)
        pose_6d[:, lower_mask],  # 下半身 (N, 54)
        pose_6d[:, hand_mask]   # 手部 (N, 180)
    )


def split_axis_angle_to_parts(poses_axis_angle, upper_joints, lower_joints, hand_joints):
    """将轴角表示分割为不同身体部位"""
    poses_reshaped = poses_axis_angle.reshape(-1, 52, 3)

    # 上半身：13个关节点
    upper_pose = poses_reshaped[:, upper_joints, :].reshape(-1, len(upper_joints)*3)

    # 下半身：9个关节点
    lower_pose = poses_reshaped[:, lower_joints, :].reshape(-1, len(lower_joints)*3)

    # 手部：30个关节点
    hand_pose = poses_reshaped[:, hand_joints, :].reshape(-1, len(hand_joints)*3)

    return upper_pose, lower_pose, hand_pose


def reconstruct_full_motion(upper_rec, lower_rec, hands_rec,
                         upper_mask, lower_mask, hand_mask,
                         mean_pose, std_pose):
    """将三种部位预测结果拼接并反归一化"""
    seq_len = upper_rec.shape[0]

    # 1. 创建完整姿态容器
    full_pose = np.zeros((seq_len, 312))

    # 2. 将各部位预测结果回填
    full_pose[:, upper_mask] = upper_rec
    full_pose[:, lower_mask] = lower_rec
    full_pose[:, hand_mask] = hands_rec

    # 3. 反归一化
    denormalized_pose = full_pose * std_pose + mean_pose

    return denormalized_pose


class RVQModelLoader:
    """RVQ-VAE模型加载器"""

    def __init__(self, device='cuda:0'):
        self.device = device
        self.models = {}

    def load_model(self, model_path, body_part):
        """加载指定身体部位的模型"""
        if body_part not in self.models:
            # 根据身体部位设置参数
            if body_part == 'upper':
                dim_pose = 13 * 6  # 78维
                joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            elif body_part == 'lower':
                dim_pose = 9 * 6   # 54维
                joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
            elif body_part == 'hands':
                dim_pose = 30 * 6  # 180维
                joints = list(range(22, 52))
            else:
                raise ValueError(f"未知的身体部位: {body_part}")

            # 创建模型实例
            args = self._create_model_args()
            model = RVQVAE(args, dim_pose, args.nb_code, args.code_dim,
                          args.output_emb_width, args.down_t, args.stride_t,
                          args.width, args.depth, args.dilation_growth_rate,
                          args.vq_act, args.vq_norm)

            # 加载权重
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location='cpu')

                # 检查checkpoint中的模型参数
                if 'net' in ckpt:
                    checkpoint_params = ckpt['net']
                else:
                    checkpoint_params = ckpt

                # 尝试使用strict=False加载，允许参数不匹配
                try:
                    model.load_state_dict(checkpoint_params, strict=False)
                except Exception as e:
                    print(f"⚠️ 模型加载警告: {e}")
                    print("🔧 尝试使用宽松模式加载...")
                    model.load_state_dict(checkpoint_params, strict=False)

                model.to(self.device)
                model.eval()
                self.models[body_part] = model
            else:
                raise FileNotFoundError(f"模型文件未找到: {model_path}")

    def _create_model_args(self):
        """创建模型参数对象"""
        class Args:
            # 必需的量化参数
            num_quantizers = 6
            shared_codebook = False
            quantize_dropout_prob = 0.2

            # 必需的架构参数
            mu = 0.99  # 指数移动平均，用于代码本更新
            nb_code = 2048  # 代码本大小
            code_dim = 256  # 代码维度 (与保存的模型匹配)
            output_emb_width = 256  # 输出嵌入宽度
            down_t = 2  # 下采样层数
            stride_t = 2  # 时间步长
            width = 512  # 网络宽度
            depth = 3  # 网络深度
            dilation_growth_rate = 3  # 膨胀增长率
            vq_act = 'relu'  # VQ激活函数
            vq_norm = None  # VQ归一化
        return Args()


def process_sequence(data, model, device):
    """处理任意长度序列数据，支持完整长度一次性推理"""
    seq_len = data.shape[0]

    # 一次性处理完整序列，不分块
    chunk_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(chunk_tensor)
        rec_data = result['rec_pose']  # 直接获取重构数据
        return rec_data.squeeze(0).cpu().numpy()


def get_args_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(description='Seamless数据集运动重建推理脚本',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 输入输出参数
    parser.add_argument('--input-npz', type=str, required=True,
                        help='输入运动数据NPZ文件路径')
    parser.add_argument('--output-npz', type=str, required=True,
                        help='输出重建后的运动数据NPZ文件路径')

    # 模型路径参数
    parser.add_argument('--upper-model', type=str,
                        default='outputs/rvq_seamless/seamless_144frame_1024batch_256dim_2048code_upper/net_best.pth',
                        help='上半身模型路径')
    parser.add_argument('--lower-model', type=str,
                        default='outputs/rvq_seamless/seamless_144frame_1024batch_256dim_2048code_lower/net_best.pth',
                        help='下半身模型路径')
    parser.add_argument('--hands-model', type=str,
                        default='outputs/rvq_seamless/seamless_144frame_1024batch_256dim_2048code_hands/net_best.pth',
                        help='手部模型路径')

    # 归一化参数
    parser.add_argument('--mean-pose', type=str,
                        default='./mean_std_seamless/seamless_2_312_mean.npy',
                        help='姿态归一化均值文件路径')
    parser.add_argument('--std-pose', type=str,
                        default='./mean_std_seamless/seamless_2_312_std.npy',
                        help='姿态归一化标准差文件路径')

    # 其他参数
    parser.add_argument('--gpu-id', type=int, default=1,
                        help='GPU设备ID')

    return parser.parse_args()


def main():
    """主函数"""
    args = get_args_parser()

    # 设置日志
    output_dir = os.path.dirname(args.output_npz)
    logger = setup_logger(output_dir)

    logger.info("="*60)
    logger.info("Seamless数据集运动重建推理脚本")
    logger.info("="*60)

    # 1. 验证输入文件
    logger.info(f"📂 验证输入文件: {args.input_npz}")
    is_valid, input_data = validate_input_file(args.input_npz)
    if not is_valid:
        logger.error(f"❌ 输入文件验证失败: {input_data}")
        return 1

    # 2. 验证模型文件
    model_paths = {
        'upper': args.upper_model,
        'lower': args.lower_model,
        'hands': args.hands_model
    }
    logger.info("🔍 验证模型文件...")
    validate_models_exist(model_paths)

    # 3. 加载归一化参数
    logger.info("📊 加载归一化参数...")
    mean_pose = np.load(args.mean_pose)
    std_pose = np.load(args.std_pose)
    logger.info(f"   均值形状: {mean_pose.shape}")
    logger.info(f"   标差形状: {std_pose.shape}")

    # 4. 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 使用设备: {device}")

    # 5. 创建身体部位掩码
    upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 13个关节点
    lower_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # 9个关节点
    hand_joints = list(range(22, 52))  # 30个关节点

    upper_mask = [i*6 + j for i in upper_joints for j in range(6)]  # 78维
    lower_mask = [i*6 + j for i in lower_joints for j in range(6)]  # 54维
    hand_mask = [i*6 + j for i in hand_joints for j in range(6)]   # 180维

    logger.info(f"👤 身体部位分割:")
    logger.info(f"   上半身: {len(upper_joints)}个关节, {len(upper_mask)}维")
    logger.info(f"   下半身: {len(lower_joints)}个关节, {len(lower_mask)}维")
    logger.info(f"   手部: {len(hand_joints)}个关节, {len(hand_mask)}维")

    # 6. 创建模型加载器
    model_loader = RVQModelLoader(device)

    # 7. 加载三个模型
    logger.info("🚀 加载预训练模型...")
    model_loader.load_model(args.upper_model, 'upper')
    model_loader.load_model(args.lower_model, 'lower')
    model_loader.load_model(args.hands_model, 'hands')

    logger.info("✅ 所有模型加载完成")

    # 8. 读取并处理输入数据
    logger.info("📥 读取并处理输入数据...")

    # 读取NPZ文件字段
    input_data = np.load(args.input_npz, allow_pickle=True)
    global_orient = input_data["smplh:global_orient"]  # (N, 3)
    body_pose = input_data["smplh:body_pose"].reshape(-1, 63)  # (N, 21, 3) -> (N, 63)
    left_hand_pose = input_data["smplh:left_hand_pose"].reshape(-1, 45)  # (N, 15, 3) -> (N, 45)
    right_hand_pose = input_data["smplh:right_hand_pose"].reshape(-1, 45)  # (N, 15, 3) -> (N, 45)
    translation = input_data["smplh:translation"]  # (N, 3)
    
    # 将平移数据从厘米转换为米
    translation = translation / 100.0

    # 组装为156维轴角表示
    poses_axis_angle = assemble_seamless_pose(global_orient, body_pose, left_hand_pose, right_hand_pose)
    logger.info(f"   原始数据形状: {poses_axis_angle.shape}")

    # 转换为6D表示并归一化
    poses_6d = axis_angle_to_6d(poses_axis_angle)
    normalized_poses = (poses_6d - mean_pose) / std_pose
    logger.info(f"   归一化后形状: {normalized_poses.shape}")

    # 分割为三个身体部位
    upper_data, lower_data, hands_data = split_to_body_parts(normalized_poses, upper_mask, lower_mask, hand_mask)

    logger.info(f"   上半身数据: {upper_data.shape}")
    logger.info(f"   下半身数据: {lower_data.shape}")
    logger.info(f"   手部数据: {hands_data.shape}")

    # 9. 分别推理三种身体部位
    logger.info("🤖 开始模型推理...")

    with torch.no_grad():
        # 上半身推理
        upper_rec = process_sequence(upper_data, model_loader.models['upper'], device)

        # 下半身推理
        lower_rec = process_sequence(lower_data, model_loader.models['lower'], device)

        # 手部推理
        hands_rec = process_sequence(hands_data, model_loader.models['hands'], device)

    logger.info("✅ 所有部位推理完成")

    # 10. 重建完整运动数据
    logger.info("🔗 重建完整运动数据...")

    reconstructed_6d = reconstruct_full_motion(
        upper_rec, lower_rec, hands_rec,
        upper_mask, lower_mask, hand_mask,
        mean_pose, std_pose
    )

    logger.info(f"   重建后形状: {reconstructed_6d.shape}")

    # 11. 转换回轴角表示
    reconstructed_axis_angle = d6_to_axis_angle(reconstructed_6d)
    logger.info(f"   轴角表示形状: {reconstructed_axis_angle.shape}")

    # 12. 分割回原始格式
    seq_len = reconstructed_axis_angle.shape[0]

    # 按照SMPL-X标准格式分割：global_orient(3) + body_pose(63) + left_hand_pose(45) + right_hand_pose(45)
    rec_global_orient = reconstructed_axis_angle[:, :3]  # (N, 3)
    rec_body_pose = reconstructed_axis_angle[:, 3:66].reshape(-1, 63)  # (N, 21, 3) -> (N, 63)
    rec_left_hand_pose = reconstructed_axis_angle[:, 66:111].reshape(-1, 45)  # (N, 15, 3) -> (N, 45)
    rec_right_hand_pose = reconstructed_axis_angle[:, 111:156].reshape(-1, 45)  # (N, 15, 3) -> (N, 45)
    rec_translation = translation  # 保持原始平移不变

    logger.info(f"   global_orient形状: {rec_global_orient.shape}")
    logger.info(f"   body_pose形状: {rec_body_pose.shape}")
    logger.info(f"   left_hand_pose形状: {rec_left_hand_pose.shape}")
    logger.info(f"   right_hand_pose形状: {rec_right_hand_pose.shape}")
    logger.info(f"   translation形状: {rec_translation.shape}")

    logger.info("💾 保存重建结果...")

    # 13. 保存结果
    output_dir = os.path.dirname(args.output_npz)
    if output_dir:  # 只有当目录不为空时才创建
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "smplh:global_orient": rec_global_orient,
        "smplh:body_pose": rec_body_pose,
        "smplh:left_hand_pose": rec_left_hand_pose,
        "smplh:right_hand_pose": rec_right_hand_pose,
        "smplh:translation": rec_translation
    }

    # 使用**kwargs方式保存，避免字典对象问题
    np.savez(args.output_npz, **output_data)
    logger.info(f"✅ 重建完成！结果保存至: {args.output_npz}")

    # 14. 输出统计信息
    logger.info("="*60)
    logger.info("📊 重建统计信息:")
    logger.info(f"   输入序列长度: {seq_len}")

    # 计算各部位重建误差
    upper_error = np.mean((upper_rec - upper_data)**2)
    lower_error = np.mean((lower_rec - lower_data)**2)
    hands_error = np.mean((hands_rec - hands_data)**2)

    logger.info(f"   上半身L2误差: {upper_error:.6f}")
    logger.info(f"   下半身L2误差: {lower_error:.6f}")
    logger.info(f"   手部L2误差: {hands_error:.6f}")
    logger.info(f"   整体L2误差: {(upper_error + lower_error + hands_error):.6f}")

    return 0

if __name__ == "__main__":
    main()
    
# python seamless_motion_reconstruction.py --input-npz V00_S0080_I00000377_P0115.npz --output-npz recon_144_V00_S0080_I00000377_P0115.npz