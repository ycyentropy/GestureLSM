import numpy as np

# 正确的文件路径（Linux系统使用/作为分隔符）
file_path = '/home/embodied/yangchenyu/GestureLSM/recon_144_V00_S0080_I00000377_P0115.npz'

# 加载npz文件
data = np.load(file_path)

# 打印文件中的所有键
print('文件中的键:')
for key in data.keys():
    print(f'- {key}')

# 打印每个键对应数据的详细信息
print('\n每个键的数据形状和类型:')
for key in data.keys():
    arr = data[key]
    print(f'键: {key}')
    print(f'  形状: {arr.shape}')
    print(f'  数据类型: {arr.dtype}')
    print(f'  维度: {arr.ndim}')
    print()