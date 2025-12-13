import numpy as np
import os

# 检查被跳过的文件路径
file_path = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train/0005/0035/V00_S0235_I00000536_P0321.npz'

print(f"检查文件: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")

# 尝试加载文件并查看内容
try:
    data = np.load(file_path, allow_pickle=True)
    print("\n文件包含的键:")
    for key in data.files:
        print(f"- {key}")
    
    # 检查smplh相关数据
    print("\nSMPLH相关数据检查:")
    smplh_keys = [key for key in data.files if key.startswith('smplh:')]
    for key in smplh_keys:
        arr = data[key]
        print(f"{key}: 形状={arr.shape}, 类型={arr.dtype}")
        # 检查是否有空数据
        if isinstance(arr, np.ndarray):
            print(f"  是否包含NaN: {np.isnan(arr).any()}")
            print(f"  是否包含无穷大: {np.isinf(arr).any()}")
            print(f"  数据统计: 最小值={arr.min() if arr.size > 0 else 'N/A'}, 最大值={arr.max() if arr.size > 0 else 'N/A'}")
    
    # 检查body_pose和global_orient的维度，这可能是被跳过的原因
    if 'smplh:body_pose' in data:
        body_pose = data['smplh:body_pose']
        print(f"\nbody_pose 详细检查:")
        print(f"  维度: {body_pose.ndim}")
        print(f"  形状: {body_pose.shape}")
        print(f"  帧数: {body_pose.shape[0] if body_pose.ndim > 0 else 0}")
    
    if 'smplh:global_orient' in data:
        global_orient = data['smplh:global_orient']
        print(f"\nglobal_orient 详细检查:")
        print(f"  维度: {global_orient.ndim}")
        print(f"  形状: {global_orient.shape}")
        print(f"  帧数: {global_orient.shape[0] if global_orient.ndim > 0 else 0}")
        
    # 检查is_valid标志
    if 'smplh:is_valid' in data:
        is_valid = data['smplh:is_valid']
        print(f"\nis_valid 检查:")
        print(f"  形状: {is_valid.shape}")
        print(f"  有效值数量: {np.sum(is_valid) if isinstance(is_valid, np.ndarray) else 'N/A'}")
        print(f"  无效值数量: {len(is_valid) - np.sum(is_valid) if isinstance(is_valid, np.ndarray) and is_valid.size > 0 else 'N/A'}")
        
    # 检查是否有足够的有效帧
    if 'smplh:body_pose' in data and 'smplh:global_orient' in data:
        body_pose = data['smplh:body_pose']
        global_orient = data['smplh:global_orient']
        
        # 检查帧数量是否为0或不匹配
        if body_pose.ndim == 0 or body_pose.shape[0] == 0:
            print("\n问题: body_pose 帧数量为0")
        elif global_orient.ndim == 0 or global_orient.shape[0] == 0:
            print("\n问题: global_orient 帧数量为0")
        elif body_pose.shape[0] != global_orient.shape[0]:
            print(f"\n问题: 帧数量不匹配 - body_pose: {body_pose.shape[0]}, global_orient: {global_orient.shape[0]}")
        else:
            print(f"\n帧数量正常: {body_pose.shape[0]}")
            
        # 检查body_pose的维度是否符合要求 (应该是 (帧数, 21, 3))
        if body_pose.ndim >= 2 and body_pose.shape[1:] != (21, 3):
            print(f"\n问题: body_pose 维度不符合要求，期望 (帧数, 21, 3)，实际是 {body_pose.shape}")
            
        # 检查global_orient的维度是否符合要求 (应该是 (帧数, 3))
        if global_orient.ndim >= 2 and global_orient.shape[1:] != (3,):
            print(f"\n问题: global_orient 维度不符合要求，期望 (帧数, 3)，实际是 {global_orient.shape}")
    
    print("\n文件检查完成")
    
except Exception as e:
    print(f"\n加载文件时出错: {e}")