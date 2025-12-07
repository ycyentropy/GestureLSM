#!/usr/bin/env python3
"""
分析NPZ文件内容，不使用allow_pickle=True
"""

import numpy as np

# 文件路径
file_path = '/home/embodied/yangchenyu/GestureLSM/recon_144_V00_S0080_I00000377_P0115.npz'

try:
    # 不使用allow_pickle，看看能否获取基本信息
    with np.load(file_path, allow_pickle=False) as data:
        print('文件基本信息:')
        print(f'键的数量: {len(data.keys())}')
        print(f'所有键: {list(data.keys())}')

        # 尝试访问每个键
        for key in data.keys():
            try:
                arr = data[key]
                print(f'\n键: {key}')
                print(f'  形状: {arr.shape}')
                print(f'  数据类型: {arr.dtype}')

                # 如果是object类型，我们无法直接读取内容
                if arr.dtype == 'object':
                    print(f'  ⚠️  这是object类型，需要allow_pickle=True才能读取')
                else:
                    print(f'  可以安全读取此数据')

            except Exception as e:
                print(f'  ❌ 无法读取键 {key}: {e}')

except Exception as e:
    print(f'❌ 无法加载文件: {e}')
    print('可能需要使用allow_pickle=True')