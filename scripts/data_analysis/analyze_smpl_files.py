import numpy as np
import os

# 文件路径
file1_path = '/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models/smplh/SMPLH_NEUTRAL.npz'
file2_path = '/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'

def analyze_npz_file(file_path):
    """分析NPZ文件并返回其内容信息"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    print(f"\n分析文件: {file_path}")
    print("="*80)
    
    try:
        # 加载NPZ文件
        data = np.load(file_path, allow_pickle=True)
        
        # 获取所有键
        keys = list(data.keys())
        print(f"文件包含 {len(keys)} 个键:")
        print(keys)
        print()
        
        # 分析每个键的值
        file_info = {}
        for key in keys:
            value = data[key]
            print(f"\n键: {key}")
            print(f"  类型: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                print(f"  维度: {value.ndim}")
                
                # 对于数值型数组，显示一些统计信息
                if np.issubdtype(value.dtype, np.number):
                    print(f"  最小值: {value.min():.6f}")
                    print(f"  最大值: {value.max():.6f}")
                    print(f"  平均值: {value.mean():.6f}")
                    print(f"  标准差: {value.std():.6f}")
                
                # 保存信息用于后续比较
                file_info[key] = {
                    'type': str(type(value)),
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'ndim': value.ndim
                }
                if np.issubdtype(value.dtype, np.number):
                    file_info[key]['min'] = value.min()
                    file_info[key]['max'] = value.max()
                    file_info[key]['mean'] = value.mean()
            else:
                print(f"  值: {value}")
                file_info[key] = {
                    'type': str(type(value)),
                    'value': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                }
        
        return file_info
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def compare_files(file1_info, file2_info, file1_name, file2_name):
    """比较两个文件的信息"""
    if file1_info is None or file2_info is None:
        print("无法进行比较，因为一个或两个文件分析失败")
        return
    
    print(f"\n{'='*80}")
    print(f"比较 {file1_name} 和 {file2_name}")
    print(f"{'='*80}")
    
    # 获取所有键
    keys1 = set(file1_info.keys())
    keys2 = set(file2_info.keys())
    
    # 找出共同的键和各自独有的键
    common_keys = keys1 & keys2
    unique_keys1 = keys1 - keys2
    unique_keys2 = keys2 - keys1
    
    print(f"\n共同的键 ({len(common_keys)}):")
    for key in sorted(common_keys):
        info1 = file1_info[key]
        info2 = file2_info[key]
        
        # 检查形状是否相同
        shape_match = info1.get('shape') == info2.get('shape')
        dtype_match = info1.get('dtype') == info2.get('dtype')
        
        match_status = "✓ 完全匹配" if shape_match and dtype_match else "✗ 不完全匹配"
        print(f"  {key}: {match_status}")
        
        if not shape_match:
            print(f"    - {file1_name} 形状: {info1.get('shape')}")
            print(f"    - {file2_name} 形状: {info2.get('shape')}")
        if not dtype_match:
            print(f"    - {file1_name} 数据类型: {info1.get('dtype')}")
            print(f"    - {file2_name} 数据类型: {info2.get('dtype')}")
    
    print(f"\n仅在 {file1_name} 中存在的键 ({len(unique_keys1)}):")
    for key in sorted(unique_keys1):
        print(f"  {key}")
    
    print(f"\n仅在 {file2_name} 中存在的键 ({len(unique_keys2)}):")
    for key in sorted(unique_keys2):
        print(f"  {key}")
    
    # 深入比较共同键中的数值数组
    print(f"\n{'='*80}")
    print("共同键数值数组的详细比较:")
    print(f"{'='*80}")
    
    for key in sorted(common_keys):
        info1 = file1_info[key]
        info2 = file2_info[key]
        
        # 只有当两个都是数值数组时才进行详细比较
        if 'shape' in info1 and 'shape' in info2 and 'min' in info1 and 'min' in info2:
            print(f"\n键: {key}")
            print(f"  形状: {info1['shape']} vs {info2['shape']}")
            print(f"  数据类型: {info1['dtype']} vs {info2['dtype']}")
            print(f"  最小值: {info1['min']:.6f} vs {info2['min']:.6f}")
            print(f"  最大值: {info1['max']:.6f} vs {info2['max']:.6f}")
            print(f"  平均值: {info1['mean']:.6f} vs {info2['mean']:.6f}")

if __name__ == "__main__":
    print("开始分析SMPL模型文件...")
    
    # 分析第一个文件
    file1_info = analyze_npz_file(file1_path)
    
    # 分析第二个文件
    file2_info = analyze_npz_file(file2_path)
    
    # 比较两个文件
    compare_files(file1_info, file2_info, "SMPLH_NEUTRAL.npz", "SMPLX_NEUTRAL_2020.npz")
    
    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}")