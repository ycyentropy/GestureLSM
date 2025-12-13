import numpy as np
import os
import glob
from tqdm import tqdm

def check_file_integrity(data_dir):
    """
    检查所有NPZ文件的完整性，找出缺少必要键或帧数量为0的文件
    """
    # 收集所有npz文件路径
    npz_files = glob.glob(os.path.join(data_dir, '**/*.npz'), recursive=True)
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 统计信息
    missing_keys_files = []
    zero_frames_files = []
    dimension_error_files = []
    
    # 遍历所有文件
    for npz_file in tqdm(npz_files, desc="检查文件完整性"):
        try:
            data = np.load(npz_file)
            
            # 检查必要的键是否存在
            if 'smplh:body_pose' not in data or 'smplh:global_orient' not in data or 'smplh:translation' not in data:
                missing_keys_files.append(npz_file)
                continue
            
            # 获取姿态数据
            body_pose = data['smplh:body_pose']
            global_orient = data['smplh:global_orient']
            
            # 检查帧数量是否为0
            if body_pose.shape[0] == 0 or global_orient.shape[0] == 0:
                zero_frames_files.append(npz_file)
                continue
                
            # 检查维度是否正确
            dimension_error = False
            if len(body_pose.shape) != 3 or body_pose.shape[1:] != (21, 3):
                dimension_error = True
            if len(global_orient.shape) != 2 or global_orient.shape[1] != 3:
                dimension_error = True
            if dimension_error:
                dimension_error_files.append(npz_file)
                
        except Exception as e:
            print(f"处理文件 {npz_file} 时出错: {str(e)}")
    
    # 输出统计结果
    print(f"\n检查结果:")
    print(f"缺少必要键的文件数量: {len(missing_keys_files)}")
    for file in missing_keys_files[:5]:  # 只显示前5个
        print(f"  - {file}")
    if len(missing_keys_files) > 5:
        print(f"  ... 还有 {len(missing_keys_files) - 5} 个文件")
    
    print(f"\n帧数量为0的文件数量: {len(zero_frames_files)}")
    for file in zero_frames_files[:5]:  # 只显示前5个
        print(f"  - {file}")
    if len(zero_frames_files) > 5:
        print(f"  ... 还有 {len(zero_frames_files) - 5} 个文件")
    
    print(f"\n维度错误的文件数量: {len(dimension_error_files)}")
    for file in dimension_error_files[:5]:  # 只显示前5个
        print(f"  - {file}")
    if len(dimension_error_files) > 5:
        print(f"  ... 还有 {len(dimension_error_files) - 5} 个文件")

if __name__ == "__main__":
    # 数据集目录
    data_dir = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction'
    
    check_file_integrity(data_dir)