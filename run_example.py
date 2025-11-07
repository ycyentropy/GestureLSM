#!/usr/bin/env python3
"""
Example script demonstrating how to use the inference and rendering scripts
"""

import os
import sys
import subprocess
import argparse

def run_inference(audio_path, output_dir):
    """Run gesture inference"""
    print("="*60)
    print("STEP 1: RUNNING GESTURE INFERENCE")
    print("="*60)

    cmd = [
        "python", "inference.py",
        "--config", "configs/shortcut_rvqvae_128_hf.yaml",
        "--audio", audio_path,
        "--output", output_dir,
        "--checkpoint", "ckpt/new_540_shortcut.bin"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Input audio: {audio_path}")
    print(f"Output directory: {output_dir}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Inference completed successfully!")
        print(result.stdout)

        # Find the generated NPZ file
        import glob
        npz_files = glob.glob(os.path.join(output_dir, "inference", "*", "result", "result_*.npz"))
        if npz_files:
            npz_file = npz_files[0]
            print(f"Generated motion file: {npz_file}")
            return npz_file
        else:
            print("❌ No NPZ file found in output directory")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed!")
        print(f"Error: {e.stderr}")
        return None

def run_rendering(npz_file, audio_file, output_video):
    """Run video rendering"""
    print("\n" + "="*60)
    print("STEP 2: RENDERING VIDEO")
    print("="*60)

    cmd = [
        "python", "render_video.py",
        "--input", npz_file,
        "--audio", audio_file,
        "--output", output_video,
        "--config", "configs/shortcut_rvqvae_128_hf.yaml",
        "--method", "simple"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Input motion: {npz_file}")
    print(f"Input audio: {audio_file}")
    print(f"Output video: {output_video}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Video rendering completed successfully!")
        print(result.stdout)

        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / 1024 / 1024
            print(f"Generated video: {output_video} ({size_mb:.2f} MB)")
            return output_video
        else:
            print("❌ No video file generated")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Video rendering failed!")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run complete gesture generation pipeline')
    parser.add_argument('--audio', '-a', type=str, required=True,
                       help='Path to input audio file')
    parser.add_argument('--output', '-o', type=str, default='./example_output/',
                       help='Output directory')
    parser.add_argument('--video', '-v', type=str, default='generated_gesture.mp4',
                       help='Output video filename')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.audio):
        print(f"❌ Audio file not found: {args.audio}")
        sys.exit(1)

    print("🚀 Starting complete gesture generation pipeline...")
    print(f"📁 Input audio: {args.audio}")
    print(f"📁 Output directory: {args.output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Run inference
    npz_file = run_inference(args.audio, args.output)

    if not npz_file:
        print("❌ Pipeline failed at inference step")
        sys.exit(1)

    # Step 2: Run rendering
    output_video_path = os.path.join(args.output, args.video)
    video_file = run_rendering(npz_file, args.audio, output_video_path)

    if not video_file:
        print("❌ Pipeline failed at rendering step")
        sys.exit(1)

    # Success!
    print("\n" + "🎉"*20)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("🎉"*20)
    print(f"📹 Generated video: {video_file}")
    print(f"📊 Motion data: {npz_file}")
    print(f"📁 Output directory: {args.output}")

    # Show file sizes
    video_size = os.path.getsize(video_file) / 1024 / 1024
    motion_size = os.path.getsize(npz_file) / 1024 / 1024
    print(f"💾 Video size: {video_size:.2f} MB")
    print(f"💾 Motion data size: {motion_size:.2f} MB")

    print("\nYou can now:")
    print(f"1. Play the video: {video_file}")
    print(f"2. Use motion data in 3D software: {npz_file}")
    print("3. Repeat with different audio files!")

if __name__ == "__main__":
    main()