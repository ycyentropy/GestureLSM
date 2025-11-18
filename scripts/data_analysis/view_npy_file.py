import numpy as np
import os

# 文件路径
file_path = '/home/embodied/yangchenyu/GestureLSM/mean_std/beatx_2_trans_std.npy'

# 检查文件是否存在
if os.path.exists(file_path):
    print(f"文件存在: {file_path}")
    
    # 读取npy文件
    data = np.load(file_path)
    
    # 显示基本信息
    print(f"数据类型: {data.dtype}")
    print(f"数据形状: {data.shape}")
    print(f"数据维度: {data.ndim}")
    print(f"数据大小: {data.size}")
    print(f"数据占用内存: {data.nbytes / (1024 * 1024):.6f} MB")
    
    # 显示数据内容（限制显示数量）
    print("\n数据内容:")
    print(data)
    
    # 如果是一维或二维数组，可以显示一些统计信息
    if data.ndim <= 2:
        print("\n统计信息:")
        print(f"最小值: {np.min(data)}")
        print(f"最大值: {np.max(data)}")
        print(f"平均值: {np.mean(data)}")
        print(f"标准差: {np.std(data)}")
else:
    print(f"文件不存在: {file_path}")