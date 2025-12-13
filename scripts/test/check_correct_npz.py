import numpy as np

# 检查原始文件和重建文件
original_file = '/home/embodied/yangchenyu/GestureLSM/V00_S0080_I00000377_P0115.npz'
recon_file = '/home/embodied/yangchenyu/GestureLSM/output_recon_fixed.npz'

print("=== 原始文件格式 ===")
try:
    with np.load(original_file) as data:
        print('文件中的键:')
        smpl_keys = [key for key in data.keys() if key.startswith('smplh:')]
        for key in smpl_keys:
            arr = data[key]
            print(f'- {key}: 形状 {arr.shape}, 类型 {arr.dtype}')
except Exception as e:
    print(f'❌ 无法加载原始文件: {e}')

print("\n=== 重建文件格式 ===")
try:
    with np.load(recon_file) as data:
        print('文件中的键:')
        for key in data.keys():
            try:
                arr = data[key]
                print(f'- {key}: 形状 {arr.shape}, 类型 {arr.dtype}')
            except Exception as e:
                print(f'- {key}: ❌ 无法读取 ({e})')
except Exception as e:
    print(f'❌ 无法加载重建文件: {e}')
    if 'Object arrays cannot be loaded' in str(e):
        print('这个文件包含object数据，需要allow_pickle=True检查内容')