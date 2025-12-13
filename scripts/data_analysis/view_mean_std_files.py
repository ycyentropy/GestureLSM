import numpy as np
import os

# 文件路径
dir_path = '/home/embodied/yangchenyu/GestureLSM/mean_std_seamless/'
files = [
    'seamless_2_312_mean.npy',
    'seamless_2_312_std.npy', 
    'seamless_2_trans_mean.npy',
    'seamless_2_trans_std.npy'
]

# 查看每个文件的内容
for file_name in files:
    file_path = os.path.join(dir_path, file_name)
    print(f"\n=== {file_name} ===")
    
    # 加载数据
    data = np.load(file_path)
    
    # 显示基本信息
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: {np.min(data):.6f} 到 {np.max(data):.6f}")
    
    # 显示部分数据内容
    print("\n数据内容预览:")
    if data.size <= 20:
        print(data)
    else:
        # 对于大数组，只显示前10个和后10个元素
        print(f"前10个元素: {data.flatten()[:10]}")
        print(f"后10个元素: {data.flatten()[-10:]}")
    
    # 如果是姿态数据(312维度)，显示各部分的统计信息
    if '312' in file_name:
        # 假设312维度是由多个6D旋转组成的（52个关节，每个6D）
        print("\n各部分统计信息:")
        # 计算均值/标准差的均值
        print(f"平均值: {np.mean(data):.6f}")
        print(f"中位数: {np.median(data):.6f}")
        print(f"标准差: {np.std(data):.6f}")