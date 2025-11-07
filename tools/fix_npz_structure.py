#!/usr/bin/env python3
"""
Fix NPZ file structure for rendering
"""

import numpy as np
import os

def fix_npz_structure(input_file, output_file):
    # Load the data
    data = np.load(input_file, allow_pickle=True)
    
    # Extract and reshape data - remove batch dimension and adjust pose format
    poses = data['poses']  # Original shape: (1, 128, 330)
    trans = data['trans']  # Original shape: (1, 128, 3)
    expressions = data['expressions']  # Original shape: (1, 128, 100)
    
    # Remove batch dimension
    poses = poses[0]  # New shape: (128, 330)
    trans = trans[0]  # New shape: (128, 3)
    expressions = expressions[0]  # New shape: (128, 100)
    
    # Optional: Check if we need to reshape poses for SMPL-X format
    # For SMPL-X, we might need to reshape to (num_frames, 55, 3) format
    # Assuming 330 = 55 joints * 6 (rotation matrices) or 55 * 3 (axis-angle)
    # For this test, we'll keep as is since we just need to test rendering pipeline
    
    # Save fixed data
    np.savez(
        output_file,
        poses=poses,
        trans=trans,
        expressions=expressions,
        # Preserve other metadata
        betas=data.get('betas', np.zeros(10)),
        model=data.get('model', 'smplx'),
        gender=data.get('gender', 'neutral'),
        mocap_frame_rate=data.get('mocap_frame_rate', 30)
    )
    
    print(f"Fixed NPZ file saved to: {output_file}")
    print(f"New shapes:")
    print(f"  poses: {poses.shape}")
    print(f"  trans: {trans.shape}")
    print(f"  expressions: {expressions.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fix NPZ file structure for rendering')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input NPZ file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output NPZ file')
    args = parser.parse_args()
    
    fix_npz_structure(args.input, args.output)