import numpy as np

# 加载npz文件
npz_path = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train/0000/0000/V00_S0039_I00000581_P0061.npz'
data = np.load(npz_path, allow_pickle=True)

# 打印SMPLH相关参数
print('SMPLH parameters:')
for key in data.files:
    if 'smplh:' in key:
        print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')
        # 打印第一个样本的部分值，了解数据格式
        if len(data[key].shape) > 0:
            print(f'  First sample shape: {data[key][0].shape if len(data[key].shape) > 1 else data[key][0]}')

# 检查是否有需要的其他参数
print('\nOther potentially useful keys:')
for key in data.files:
    if 'movement:' in key and 'expression' in key:
        print(f'{key}: shape={data[key].shape}')