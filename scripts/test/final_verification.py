import numpy as np

# 验证重建结果
original_file = '/home/embodied/yangchenyu/GestureLSM/V00_S0080_I00000377_P0115.npz'
recon_file = '/home/embodied/yangchenyu/GestureLSM/output_recon_fixed.npz'

print("=== 原始数据 ===")
with np.load(original_file) as orig:
    orig_keys = [k for k in orig.keys() if k.startswith('smplh:')]
    for key in sorted(orig_keys):
        if key != 'smplh:is_valid':  # 跳过布尔键
            arr = orig[key]
            print(f'{key}: 形状 {arr.shape}, 类型 {arr.dtype}')

print("\n=== 重建数据 ===")
with np.load(recon_file) as recon:
    for key in sorted(recon.keys()):
        arr = recon[key]
        print(f'{key}: 形状 {arr.shape}, 类型 {arr.dtype}')

print("\n=== 数据完整性验证 ===")
with np.load(original_file) as orig, np.load(recon_file) as recon:
    orig_shape = orig['smplh:global_orient'].shape
    recon_shape = recon['smplh:global_orient'].shape

    print(f'序列长度 - 原始: {orig_shape[0]}, 重建: {recon_shape[0]}')
    print(f'形状匹配: {orig_shape == recon_shape}')

    # 计算简单差异统计
    orig_orient = orig['smplh:global_orient'][:10]  # 只取前10帧比较
    recon_orient = recon['smplh:global_orient'][:10]

    diff = np.mean(np.abs(orig_orient - recon_orient))
    print(f'前10帧平均差异: {diff:.6f}')

print("\n✅ 重建文件格式验证完成!")
print("注意: 重建文件使用allow_pickle=True时可以正确读取所有SMPL格式键名")