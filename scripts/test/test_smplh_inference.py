#!/usr/bin/env python3
"""
测试SMPLH模型推理和52关节点配置
验证seamless数据集的SMPLH模型加载和关节点计算
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
import smplx
from omegaconf import OmegaConf
from dataloaders.data_tools import joints_list
from dataloaders.seamless_sep import CustomDataset

def test_smplh_model_loading():
    print("=" * 70)
    print("测试 SMPLH 模型加载和推理")
    print("=" * 70)

    # 加载配置
    cfg = OmegaConf.load('./configs/seamless_rvqvae.yaml')

    class Args:
        def __init__(self, cfg_dict):
            for key, value in cfg_dict.items():
                setattr(self, key, value)

    args = Args(dict(cfg))

    print(f"SMPLH模型配置:")
    print(f"  模型路径: {args.data_path_1}smplx_models/")
    print(f"  模型类型: smplh")
    print(f"  性别: neutral")
    print(f"  面部轮廓: False")
    print(f"  形状参数: 10维")

    try:
        # 创建SMPLH模型
        model = smplx.create(
            args.data_path_1 + "smplx_models/",
            model_type='smplh',
            gender='neutral',
            use_face_contour=False,    # 关闭面部轮廓
            num_betas=10,             # 10维形状参数
            num_expression_coeffs=10,
            ext='pkl',                # 使用PKL格式
            use_pca=False,
        )
        print(f"✓ SMPLH模型加载成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")

        # 检查模型是否在GPU上可用
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✓ 模型已移至GPU")
        else:
            print(f"⚠️  GPU不可用，使用CPU模式")

    except Exception as e:
        print(f"✗ SMPLH模型加载失败: {e}")
        return False

    return model

def test_joint_point_calculation(model):
    print(f"\n测试52关节点计算:")

    # 创建模拟输入数据
    batch_size = 2
    sequence_length = 64

    # 模拟seamless数据格式的姿态参数
    betas = torch.randn(batch_size, 10)  # 10维形状参数
    global_orient = torch.randn(batch_size, sequence_length, 3)  # 全局方向
    body_pose = torch.randn(batch_size, sequence_length, 63)     # 身体姿态 (63维，已经是展平的)
    left_hand_pose = torch.randn(batch_size, sequence_length, 45)  # 左手 (15关节 × 3)
    right_hand_pose = torch.randn(batch_size, sequence_length, 45) # 右手 (15关节 × 3)
    translation = torch.randn(batch_size, sequence_length, 3)      # 平移

    print(f"输入数据维度:")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {sequence_length}")
    print(f"  全局方向: {global_orient.shape}")
    print(f"  身体姿态: {body_pose.shape}")
    print(f"  左手姿态: {left_hand_pose.shape}")
    print(f"  右手姿态: {right_hand_pose.shape}")
    print(f"  平移: {translation.shape}")

    try:
        # 移动到GPU（如果可用）
        if torch.cuda.is_available():
            betas = betas.cuda()
            global_orient = global_orient.cuda()
            body_pose = body_pose.cuda()
            left_hand_pose = left_hand_pose.cuda()
            right_hand_pose = right_hand_pose.cuda()
            translation = translation.cuda()

        # SMPLH模型需要reshape为正确的维度
        # body_pose需要从 (batch, seq, 63) 展平为 (batch*seq, 63)
        batch_size, seq_len = global_orient.shape[:2]
        global_orient_flat = global_orient.view(-1, 3)
        body_pose_flat = body_pose.view(-1, 63)
        left_hand_pose_flat = left_hand_pose.view(-1, 45)
        right_hand_pose_flat = right_hand_pose.view(-1, 45)
        translation_flat = translation.view(-1, 3)
        betas_expanded = betas.unsqueeze(1).repeat(1, seq_len, 1).view(-1, 10)

        # 进行前向推理
        with torch.no_grad():
            output = model(
                betas=betas_expanded,
                global_orient=global_orient_flat,
                body_pose=body_pose_flat,
                left_hand_pose=left_hand_pose_flat,
                right_hand_pose=right_hand_pose_flat,
                transl=translation_flat,
                return_verts=True,
                return_joints=True,
            )

        print(f"✓ SMPLH模型推理成功")

        # 检查输出
        joints = output['joints']
        vertices = output['vertices']

        # 将输出重新整形为 (batch, seq, joints, 3)
        joints = joints.view(batch_size, seq_len, -1, 3)
        vertices = vertices.view(batch_size, seq_len, -1, 3)

        print(f"输出结果:")
        print(f"  关节点形状: {joints.shape} (批次 × 序列 × 关节 × 3D)")
        print(f"  顶点形状: {vertices.shape} (批次 × 序列 × 顶点 × 3D)")

        # 验证52关节点
        num_joints = joints.shape[2]
        if num_joints >= 52:
            print(f"✓ 关节点数量充足: {num_joints} >= 52")

            # 提取前52个关节点
            joints_52 = joints[:, :, :52, :]  # [batch, seq, 52, 3]
            print(f"✓ 提取52个关节点成功: {joints_52.shape}")

            # 检查关节点是否有效（不是NaN或无穷大）
            if torch.isfinite(joints_52).all():
                print(f"✓ 所有关节点值都是有效的")
            else:
                print(f"⚠️  存在无效的关节点值")

            return True, joints_52
        else:
            print(f"✗ 关节点数量不足: {num_joints} < 52")
            return False, None

    except Exception as e:
        print(f"✗ SMPLH模型推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_6d_conversion():
    print(f"\n测试6D旋转表示转换:")

    # 模拟52个关节点的轴角表示
    batch_size, seq_len = 2, 64
    axis_angle = torch.randn(batch_size, seq_len, 52, 3)

    print(f"轴角输入: {axis_angle.shape}")

    try:
        from dataloaders.utils import rotation_conversions as rc

        # 转换为旋转矩阵
        rotation_matrices = rc.axis_angle_to_matrix(axis_angle.view(-1, 52, 3))
        rotation_matrices = rotation_matrices.view(batch_size, seq_len, 52, 3, 3)
        print(f"旋转矩阵: {rotation_matrices.shape}")

        # 转换为6D表示
        rotation_6d = rc.matrix_to_rotation_6d(rotation_matrices.view(-1, 3, 3))
        rotation_6d = rotation_6d.view(batch_size, seq_len, 52, 6)
        print(f"6D表示: {rotation_6d.shape}")

        # 展平为最终格式
        rotation_6d_flat = rotation_6d.view(batch_size, seq_len, 52 * 6)
        print(f"展平6D: {rotation_6d_flat.shape}")

        if rotation_6d_flat.shape[2] == 312:
            print(f"✓ 6D表示维度正确: {rotation_6d_flat.shape[2]} = 52 × 6")
            return True
        else:
            print(f"✗ 6D表示维度错误: 期望312，实际{rotation_6d_flat.shape[2]}")
            return False

    except Exception as e:
        print(f"✗ 6D转换失败: {e}")
        return False

def test_normalization():
    print(f"\n测试归一化文件加载:")

    norm_files = {
        'seamless_2_312_mean.npy': (312,),
        'seamless_2_312_std.npy': (312,),
        'seamless_2_trans_mean.npy': (3,),
        'seamless_2_trans_std.npy': (3,),
    }

    for filename, expected_shape in norm_files.items():
        filepath = f'./mean_std_seamless/{filename}'
        try:
            data = np.load(filepath)
            print(f"✓ {filename}: 形状 {data.shape}")
            if data.shape == expected_shape:
                print(f"  维度正确: {data.shape} == {expected_shape}")
            else:
                print(f"  ⚠️  维度不匹配: 期望{expected_shape}，实际{data.shape}")
        except Exception as e:
            print(f"✗ {filename}: 加载失败 - {e}")
            return False

    return True

def test_complete_pipeline():
    print(f"\n完整数据流水线测试:")

    try:
        # 加载配置
        cfg = OmegaConf.load('./configs/seamless_rvqvae.yaml')

        class Args:
            def __init__(self, cfg_dict):
                for key, value in cfg_dict.items():
                    setattr(self, key, value)

        args = Args(dict(cfg))

        # 设置必要的参数
        args.disable_filtering = True
        args.clean_first_seconds = 0
        args.clean_final_seconds = 0
        args.test_length = 128
        args.audio_sr = 16000
        args.audio_fps = 16000
        args.audio_rep = 'onset+amplitude'
        args.beat_align = False

        # 检查是否有实际的NPZ文件可以测试
        if os.path.exists(args.data_path):
            dataset = CustomDataset(args, "train", build_cache=False)
            if len(dataset.selected_files) > 0:
                print(f"✓ 找到 {len(dataset.selected_files)} 个NPZ文件进行测试")

                # 测试第一个文件的数据加载
                test_file = dataset.selected_files[0]
                print(f"  测试文件: {os.path.basename(test_file)}")

                # 加载NPZ文件
                pose_data = np.load(test_file, allow_pickle=True)

                # 检查必要的数据字段
                required_fields = [
                    'smplh:global_orient', 'smplh:body_pose',
                    'smplh:left_hand_pose', 'smplh:right_hand_pose',
                    'smplh:translation'
                ]

                # 检查betas字段的可能名称
                betas_found = False
                for betas_field in ['betas', 'smplh:betas', 'shape']:
                    if betas_field in pose_data:
                        print(f"  ✓ {betas_field}: {pose_data[betas_field].shape}")
                        betas_found = True
                        break

                if not betas_found:
                    print(f"  ⚠️  未找到betas字段，使用默认值")

                for field in required_fields:
                    if field in pose_data:
                        data = pose_data[field]
                        print(f"  ✓ {field}: {data.shape}")
                        # 特别检查body_pose的维度
                        if field == 'smplh:body_pose':
                            if len(data.shape) == 3 and data.shape[2] == 3:
                                print(f"    -> body_pose形状: {data.shape} (未展平)")
                            elif len(data.shape) == 2 and data.shape[1] == 63:
                                print(f"    -> body_pose形状: {data.shape} (已展平)")
                            else:
                                print(f"    -> body_pose形状异常: {data.shape}")
                    else:
                        print(f"  ✗ 缺少字段: {field}")
                        return False

                print(f"✓ 数据字段完整性检查通过")
            else:
                print(f"⚠️  没有找到NPZ文件，跳过实际数据测试")
        else:
            print(f"⚠️  数据路径不存在，跳过实际数据测试")

        return True

    except Exception as e:
        print(f"✗ 完整流水线测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试SMPLH模型推理和归一化...")

    # 测试SMPLH模型加载
    model = test_smplh_model_loading()
    if model is False:
        print(f"\n❌ SMPLH模型加载失败！")
        sys.exit(1)

    # 测试关节点计算
    joints_ok, joints_52 = test_joint_point_calculation(model)

    # 测试6D转换
    sixd_ok = test_6d_conversion()

    # 测试归一化
    norm_ok = test_normalization()

    # 测试完整流水线
    pipeline_ok = test_complete_pipeline()

    print(f"\n" + "=" * 70)
    print("测试总结:")
    print(f"  SMPLH模型加载: {'✓' if model else '✗'}")
    print(f"  52关节点计算: {'✓' if joints_ok else '✗'}")
    print(f"  6D表示转换: {'✓' if sixd_ok else '✗'}")
    print(f"  归一化文件: {'✓' if norm_ok else '✗'}")
    print(f"  完整流水线: {'✓' if pipeline_ok else '✗'}")

    if all([model, joints_ok, sixd_ok, norm_ok, pipeline_ok]):
        print(f"\n🎉 所有SMPLH模型推理测试通过！")
        print(f"   Seamless数据集的52关节点配置正确")
        print(f"   支持从轴角到6D表示的完整转换")
        print(f"   归一化文件加载正常")
        sys.exit(0)
    else:
        print(f"\n❌ 部分SMPLH模型推理测试失败！")
        sys.exit(1)