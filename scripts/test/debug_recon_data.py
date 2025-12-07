import numpy as np
import torch

# 检查重建数据的详细信息
recon_file = '/home/embodied/yangchenyu/GestureLSM/output_recon.npz'

print("=== 检查重建文件（使用allow_pickle=True）===")
try:
    with np.load(recon_file, allow_pickle=True) as data:
        print('文件中的键:')
        for key in data.keys():
            arr = data[key]
            print(f'- {key}:')
            print(f'  形状: {arr.shape}')
            print(f'  数据类型: {arr.dtype}')
            print(f'  维度: {arr.ndim}')
            if hasattr(arr, 'keys'):
                print(f'  包含子键: {list(arr.keys()) if hasattr(arr, "keys") else "无"}')
            print()

            # 如果是字典类型，检查其内容
            if hasattr(arr, 'keys'):
                try:
                    for sub_key in arr.keys():
                        sub_arr = arr[sub_key]
                        print(f'    {sub_key}: 形状 {getattr(sub_arr, "shape", "无")}, 类型 {getattr(sub_arr, "dtype", "无")}')
                except:
                    print('    无法访问子内容')

except Exception as e:
    print(f'❌ 无法加载文件: {e}')

print("\n=== 检查保存代码中的数据 ===")
# 模拟重建过程
rec_global_orient = np.random.randn(100, 3).astype(np.float32)
rec_body_pose = np.random.randn(100, 21, 3).astype(np.float32)
rec_left_hand_pose = np.random.randn(100, 15, 3).astype(np.float32)
rec_right_hand_pose = np.random.randn(100, 15, 3).astype(np.float32)
rec_translation = np.random.randn(100, 3).astype(np.float32)

output_data = {
    "smplh:global_orient": rec_global_orient,
    "smplh:body_pose": rec_body_pose,
    "smplh:left_hand_pose": rec_left_hand_pose,
    "smplh:right_hand_pose": rec_right_hand_pose,
    "smplh:translation": rec_translation
}

print("准备保存的数据:")
for key, value in output_data.items():
    print(f'- {key}: 形状 {value.shape}, 类型 {value.dtype}')