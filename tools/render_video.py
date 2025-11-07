#!/usr/bin/env python3
"""
Video Rendering Script for GestureLSM
This script renders MP4 videos from NPZ motion data files.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import trimesh
import pyrender
from loguru import logger
from utils import other_tools_hf
from utils.media import add_audio_to_video, convert_img_to_mp4

def setup_environment():
    """Setup environment"""
    import platform
    if platform.system() == "Linux":
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

class GestureRenderer:
    """Renderer for gesture motion sequences"""

    def __init__(self, smplx_model_path, output_dir):
        self.smplx_model_path = smplx_model_path
        self.output_dir = output_dir
        self.setup_renderer()

    def setup_renderer(self):
        """Setup PyRender scene"""
        # Create scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.5, 0.5, 0.5])

        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(camera, pose=camera_pose)

        # Add lights
        light = pyrender.DirectionalLight(intensity=3.0, color=[1.0, 1.0, 1.0])
        light_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(light, pose=light_pose)

        # Setup renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=1920,
            viewport_height=720,
            point_size=1.0
        )

    def load_smplx_model(self):
        """Load SMPL-X model"""
        try:
            import smplx
            self.smplx_model = smplx.create(
                self.smplx_model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100,
                ext='npz'
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load SMPL-X model: {e}")
            return False

    def render_frame(self, pose_params, trans, expression_params):
        """Render a single frame"""
        try:
            # Forward pass through SMPL-X
            with torch.no_grad():
                pose_params = torch.tensor(pose_params, dtype=torch.float32).unsqueeze(0)
                trans = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
                expression_params = torch.tensor(expression_params, dtype=torch.float32).unsqueeze(0)

                output = self.smplx_model(
                    body_pose=pose_params[:, 3:66],  # Remove global rotation
                    global_orient=pose_params[:, :3],
                    transl=trans,
                    expression=expression_params,
                    return_verts=True
                )

            vertices = output.vertices[0].cpu().numpy()
            faces = self.smplx_model.faces

            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Convert to PyRender mesh
            mesh = pyrender.Mesh.from_trimesh(mesh)

            # Clear previous meshes
            meshes_to_remove = []
            for node in self.scene.mesh_nodes:
                meshes_to_remove.append(node)
            for node in meshes_to_remove:
                self.scene.remove_node(node)

            # Add new mesh
            self.scene.add(mesh, pose=np.eye(4))

            # Render
            color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

            return color

        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
            return None

    def render_motion_sequence(self, npz_file, audio_file, output_video, fps=30):
        """Render complete motion sequence"""
        print(f"Loading motion data from: {npz_file}")

        # Load motion data
        data = np.load(npz_file, allow_pickle=True)
        poses = data['poses']
        trans = data['trans']
        expressions = data.get('expressions', np.zeros((len(poses), 100)))

        num_frames = len(poses)
        print(f"Rendering {num_frames} frames at {fps} FPS")

        # Load SMPL-X model
        if not self.load_smplx_model():
            raise RuntimeError("Failed to load SMPL-X model")

        # Create frames directory
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Render frames
        start_time = time.time()
        for i in range(num_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")

            if not os.path.exists(frame_path):
                print(f"Rendering frame {i+1}/{num_frames}")

                # Convert rotation matrices to axis-angle if needed
                pose_params = poses[i]
                if len(pose_params) == 165:  # SMPL-X format
                    pose_params = pose_params.reshape(55, 3)  # 55 joints * 3 parameters

                # Render frame
                frame_image = self.render_frame(pose_params, trans[i], expressions[i])

                if frame_image is not None:
                    # Save frame
                    from PIL import Image
                    img = Image.fromarray(frame_image)
                    img.save(frame_path)
                else:
                    print(f"Failed to render frame {i+1}")
                    # Create a black frame as fallback
                    black_frame = np.zeros((720, 1920, 4), dtype=np.uint8)
                    from PIL import Image
                    img = Image.fromarray(black_frame)
                    img.save(frame_path)

        render_time = time.time() - start_time
        print(f"Rendering completed in {render_time:.2f} seconds")

        # Create video from frames
        print("Creating video from frames...")
        silent_video_path = os.path.join(self.output_dir, "silent_video.mp4")

        # Convert frames to video using FFmpeg
        import subprocess
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        convert_img_to_mp4(frame_pattern, silent_video_path, framerate=fps)

        if not os.path.exists(silent_video_path):
            raise RuntimeError("Failed to create silent video")

        # Add audio
        print("Adding audio to video...")
        add_audio_to_video(silent_video_path, audio_file, output_video)

        # Clean up temporary files
        if os.path.exists(silent_video_path):
            os.remove(silent_video_path)

        # Clean up frames
        import shutil
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

        print(f"Video saved to: {output_video}")
        return output_video


def simple_render_video(npz_file, audio_file, output_video, args):
    """Simple video rendering using the existing pipeline"""
    print("Using simple rendering pipeline...")

    # Create output directory
    output_dir = os.path.dirname(output_video) if os.path.dirname(output_video) else "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Use the existing rendering function from other_tools_hf
    try:
        render_vid_path = other_tools_hf.render_one_sequence_no_gt(
            npz_file,
            output_dir,
            audio_file,
            args.data_path_1 + "smplx_models/",
            use_matplotlib=False,
            args=args,
        )

        if render_vid_path and os.path.exists(render_vid_path):
            # Move to desired output location if different
            if render_vid_path != output_video:
                import shutil
                shutil.move(render_vid_path, output_video)
            print(f"Video successfully rendered to: {output_video}")
            return output_video
        else:
            raise RuntimeError("Rendering failed - no output video generated")

    except Exception as e:
        print(f"Error in rendering: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Render video from motion data')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input NPZ motion file')
    parser.add_argument('--audio', '-a', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--output', '-o', type=str, default='output_video.mp4',
                       help='Output video file path')
    parser.add_argument('--config', '-c', type=str,
                       default='configs/shortcut_rvqvae_128_hf.yaml',
                       help='Path to config file')
    parser.add_argument('--method', '-m', type=str, default='simple',
                       choices=['simple', 'advanced'],
                       help='Rendering method: simple (existing pipeline) or advanced (custom)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input NPZ file not found: {args.input}")

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    # Setup environment
    setup_environment()

    print("Starting video rendering...")
    print(f"Input motion: {args.input}")
    print(f"Input audio: {args.audio}")
    print(f"Output video: {args.output}")
    print(f"Method: {args.method}")

    start_time = time.time()

    try:
        if args.method == 'simple':
            # Load config for simple method
            from utils import config
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['render_video.py', '--config', args.config]
            cfg, config_args = config.parse_args()
            sys.argv = original_argv

            # Set necessary arguments
            config_args.data_path_1 = "./datasets/hub/"
            config_args.render_video_width = 1920
            config_args.render_video_height = 720
            config_args.render_video_fps = 30
            config_args.render_tmp_img_filetype = "bmp"
            config_args.render_concurrent_num = 64
            config_args.debug = False
            config_args.gpus = [0]
            config_args.loader_workers = 0
            config_args.pose_fps = 30
            config_args.use_trans = True

            # Use simple rendering
            output_path = simple_render_video(args.input, args.audio, args.output, config_args)
        else:
            # Use advanced rendering
            output_dir = os.path.dirname(args.output) or "./output"
            renderer = GestureRenderer(
                smplx_model_path="./datasets/hub/smplx_models/",
                output_dir=output_dir
            )
            output_path = renderer.render_motion_sequence(args.input, args.audio, args.output)

        end_time = time.time()
        print("\n" + "="*50)
        print("VIDEO RENDERING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Rendering time: {int(end_time - start_time)} seconds")
        print(f"Output video: {output_path}")
        print(f"Video size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"Rendering failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()