#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本用于可视化渲染seamless数据集的npz文件
基于render_one_sequence_no_gt函数实现
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# 导入必要的库
import utils.fast_render
import smplx

def visualize_seamless_npz(npz_path, output_dir, model_path, fps=30, device='cuda', 
                          start_frame=0, end_frame=None, resolution=(1024, 1024), add_audio=True):
    """
    可视化seamless数据集的npz文件
    
    Args:
        npz_path: npz文件路径
        output_dir: 输出目录
        model_path: SMPLX模型路径
        fps: 输出视频的帧率
        device: 运行设备
        start_frame: 起始帧
        end_frame: 结束帧
        resolution: 输出视频分辨率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载npz文件
    print(f"Loading npz file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 获取文件名作为基础名
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    
    # 检查SMPLH参数是否存在
    required_keys = ['smplh:body_pose', 'smplh:global_orient', 'smplh:left_hand_pose', 
                    'smplh:right_hand_pose', 'smplh:translation', 'smplh:is_valid']
    
    for key in required_keys:
        if key not in data.files:
            raise ValueError(f"Required key {key} not found in npz file")
    
    # 获取帧数
    n_frames = data['smplh:body_pose'].shape[0]
    print(f"Total frames: {n_frames}")
    
    # 确定帧范围
    if end_frame is None:
        end_frame = n_frames
    end_frame = min(end_frame, n_frames)
    frames_to_render = range(start_frame, end_frame)
    print(f"Rendering frames {start_frame} to {end_frame-1}")
    
    # 初始化SMPLH模型
    print(f"Loading SMPLH model from {model_path}")
    model = smplx.create(
        model_path,
        model_type='smplh',
        gender='neutral',
        num_betas=10,
        use_pca=False,
        ext='pkl'
    ).to(device)
    
    # 准备顶点数据列表
    all_vertices = []
    
    # 处理每一帧
    for i in tqdm(frames_to_render, desc="Processing frames"):
        # 检查帧是否有效
        if not data['smplh:is_valid'][i]:
            # 如果无效，使用前一帧的数据或默认姿势
            if i > 0 and all_vertices:
                vertices = all_vertices[-1].copy()
            else:
                # 默认姿势
                input_dict = {
                    'global_orient': torch.zeros(1, 3, device=device),
                    'body_pose': torch.zeros(1, 63, device=device),  # 21*3
                    'left_hand_pose': torch.zeros(1, 45, device=device),  # 15*3
                    'right_hand_pose': torch.zeros(1, 45, device=device),  # 15*3
                    'transl': torch.zeros(1, 3, device=device),
                    'betas': torch.zeros(1, 10, device=device),
                    'expression': torch.zeros(1, 10, device=device)
                }
                with torch.no_grad():
                    output = model(return_verts=True, **input_dict)
                vertices = output.vertices[0].cpu().numpy()
            all_vertices.append(vertices)
            continue
        
        # 准备模型输入
        input_dict = {
            'global_orient': torch.tensor(data['smplh:global_orient'][i], dtype=torch.float32, device=device).unsqueeze(0),
            'body_pose': torch.tensor(data['smplh:body_pose'][i].reshape(-1), dtype=torch.float32, device=device).unsqueeze(0),
            'left_hand_pose': torch.tensor(data['smplh:left_hand_pose'][i].reshape(-1), dtype=torch.float32, device=device).unsqueeze(0),
            'right_hand_pose': torch.tensor(data['smplh:right_hand_pose'][i].reshape(-1), dtype=torch.float32, device=device).unsqueeze(0),
            'transl': torch.tensor(data['smplh:translation'][i], dtype=torch.float32, device=device).unsqueeze(0),
            'betas': torch.zeros(1, 10, device=device),  # 使用默认体型
        }
        
        # 添加表情参数（如果存在）
        if 'movement:expression' in data.files:
            # 只使用前10个表情系数
            expr = data['movement:expression'][i][:10] if data['movement:expression'][i].shape[0] >= 10 else np.zeros(10)
            input_dict['expression'] = torch.tensor(expr, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            input_dict['expression'] = torch.zeros(1, 10, device=device)
        
        # 模型前向传播
        with torch.no_grad():
            output = model(return_verts=True, **input_dict)
        
        # 获取顶点数据
        vertices = output.vertices[0].cpu().numpy()
        all_vertices.append(vertices)
    
    # 转换为numpy数组
    all_vertices = np.array(all_vertices)
    
    # 渲染视频
    output_path = os.path.join(output_dir, f"{base_name}_rendered.mp4")
    print(f"Rendering video to {output_path}")
    
    # 使用fast_render.generate_silent_videos_no_gt函数渲染视频
    # 注意：我们需要创建临时参数对象来满足函数要求
    class Args:
        def __init__(self):
            self.render_video_fps = fps
            self.render_video_width = resolution[0]
            self.render_video_height = resolution[1]
            self.render_concurrent_num = 4
            self.render_tmp_img_filetype = 'png'
    
    args = Args()
    
    # 从模型中获取faces
    # 注意：faces已经是numpy数组，不需要.cpu()
    faces = model.faces
    
    # 直接构建无声视频的路径
    silent_video_file_path = os.path.join(output_dir, "silence_video.mp4")
    
    # 生成无声视频（不依赖返回值）
    try:
        utils.fast_render.generate_silent_videos_no_gt(
            args.render_video_fps,
            args.render_video_width,
            args.render_video_height,
            args.render_concurrent_num,
            args.render_tmp_img_filetype,
            len(all_vertices),
            all_vertices,
            faces,
            output_dir
        )
        
        # 验证视频文件是否生成
        if not os.path.exists(silent_video_file_path):
            raise FileNotFoundError(f"Silent video file was not generated: {silent_video_file_path}")
            
        print(f"Silent video generated successfully at {silent_video_file_path}")
        
        # 如果需要添加音频
        if add_audio:
            # 查找对应的wav文件（与npz文件同名）
            wav_path = npz_path.replace('.npz', '.wav')
            if os.path.exists(wav_path):
                print(f"Adding audio from {wav_path}")
                try:
                    # 使用utils.media.add_audio_to_video添加音频
                    utils.media.add_audio_to_video(silent_video_file_path, wav_path, output_path)
                    print(f"Audio added successfully, output saved to {output_path}")
                    # 安全删除无声视频
                    try:
                        os.remove(silent_video_file_path)
                    except Exception as e:
                        print(f"Warning: Failed to remove silent video: {e}")
                except Exception as e:
                    print(f"Error adding audio: {e}")
                    print("Using silent video as fallback")
                    # 如果添加音频失败，使用无声视频作为最终输出
                    import shutil
                    shutil.copy2(silent_video_file_path, output_path)
            else:
                print(f"Warning: Audio file {wav_path} not found, using silent video.")
                # 直接使用无声视频作为最终输出
                import shutil
                shutil.copy2(silent_video_file_path, output_path)
        else:
            # 直接使用无声视频作为最终输出
            import shutil
            shutil.copy2(silent_video_file_path, output_path)
    except Exception as e:
        print(f"Error during video rendering: {e}")
        # 尝试创建一个简单的替代方案或提供更友好的错误信息
        print("Please check if all required dependencies are installed and the model path is correct.")
        raise
    
    print(f"Visualization completed! Output saved to {output_path}")

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description='Visualize seamless dataset npz files')
    parser.add_argument('--npz_path', type=str, required=True, 
                      help='Path to the npz file')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Output directory for rendered videos')
    parser.add_argument('--model_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models/',
                      help='Path to SMPLH model')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for output video')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on (cuda or cpu)')
    parser.add_argument('--start_frame', type=int, default=0,
                      help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=None,
                      help='End frame index (exclusive)')
    parser.add_argument('--resolution', type=int, nargs=2, default=(1024, 1024),
                      help='Output video resolution (width height)')
    parser.add_argument('--no_audio', action='store_true',
                      help='Do not add audio to the output video')
    
    args = parser.parse_args()
    
    # 检查设备是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # 执行可视化
    visualize_seamless_npz(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        fps=args.fps,
        device=args.device,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        resolution=args.resolution,
        add_audio=not args.no_audio
    )

if __name__ == '__main__':
    main()