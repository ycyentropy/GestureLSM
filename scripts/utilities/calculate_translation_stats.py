import numpy as np
import os
import glob
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import time

# 检查GPU是否可用并获取可用GPU数量
def get_available_devices():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [torch.device(f'cuda:{i}') for i in range(device_count)]
        print(f"检测到 {device_count} 张可用GPU卡")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return devices
    else:
        print("未检测到GPU，使用CPU")
        return [torch.device('cpu')]

# 获取可用设备
devices = get_available_devices()
# 默认使用第一张卡作为主设备
device = devices[0]
print(f"主设备: {device}")

def detect_outliers_iqr(data, threshold=1.5, use_gpu=True):
    """使用IQR（四分位距）方法检测异常值"""
    # 对于IQR方法，即使有GPU也在CPU上计算分位数
    print("使用CPU计算IQR异常值检测，因为大量数据的分位数计算更适合CPU处理")
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    
    # 计算异常值的上下边界
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # 如果有GPU且数据量不是特别大，使用多GPU并行计算掩码
    if use_gpu and torch.cuda.is_available() and len(data) < 10000000:  # 数据量小于1000万时使用GPU加速掩码计算
        # 对于多张GPU卡，使用并行处理
        if len(devices) > 1:
            print(f"使用 {len(devices)} 张GPU卡并行计算掩码")
            # 计算每个GPU处理的数据量
            per_gpu_size = len(data) // len(devices)
            masks = []
            
            # 为每个GPU分配数据并并行计算
            for i, dev in enumerate(devices):
                start_idx = i * per_gpu_size
                end_idx = len(data) if i == len(devices) - 1 else (i + 1) * per_gpu_size
                
                # 将数据切片移到对应的GPU
                data_slice = torch.from_numpy(data[start_idx:end_idx]).to(dev)
                lower_bound_tensor = torch.from_numpy(lower_bound).to(dev)
                upper_bound_tensor = torch.from_numpy(upper_bound).to(dev)
                
                # 在当前GPU上计算掩码
                mask_slice = torch.all((data_slice >= lower_bound_tensor) & (data_slice <= upper_bound_tensor), dim=1)
                masks.append(mask_slice.cpu().numpy())
                
                # 释放内存
                del data_slice, lower_bound_tensor, upper_bound_tensor
                torch.cuda.empty_cache()
            
            # 合并所有掩码
            return np.concatenate(masks)
        else:
            # 单GPU情况
            data_tensor = torch.from_numpy(data).to(device)
            lower_bound_tensor = torch.from_numpy(lower_bound).to(device)
            upper_bound_tensor = torch.from_numpy(upper_bound).to(device)
            mask = torch.all((data_tensor >= lower_bound_tensor) & (data_tensor <= upper_bound_tensor), dim=1)
            return mask.cpu().numpy()
    else:
        # CPU情况
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        return mask

def filter_translation_outliers(translation_data, method='iqr', threshold=1.5, use_gpu=True):
    """过滤平移数据中的异常值"""
    original_shape = translation_data.shape
    print(f"过滤前平移数据形状: {original_shape}")
    
    # 使用GPU计算统计信息
    if use_gpu and torch.cuda.is_available():
        data_tensor = torch.from_numpy(translation_data).to(device)
        print(f"过滤前平移数据统计 - 最小值: {torch.min(data_tensor, dim=0)[0].cpu().numpy()}")
        print(f"过滤前平移数据统计 - 最大值: {torch.max(data_tensor, dim=0)[0].cpu().numpy()}")
        print(f"过滤前平移数据统计 - 均值: {torch.mean(data_tensor, dim=0).cpu().numpy()}")
        print(f"过滤前平移数据统计 - 标准差: {torch.std(data_tensor, dim=0).cpu().numpy()}")
    else:
        print(f"过滤前平移数据统计 - 最小值: {np.min(translation_data, axis=0)}")
        print(f"过滤前平移数据统计 - 最大值: {np.max(translation_data, axis=0)}")
        print(f"过滤前平移数据统计 - 均值: {np.mean(translation_data, axis=0)}")
        print(f"过滤前平移数据统计 - 标准差: {np.std(translation_data, axis=0)}")
    
    if method == 'iqr':
        mask = detect_outliers_iqr(translation_data, threshold, use_gpu)
    else:  # 备用方法
        mask = np.ones(len(translation_data), dtype=bool)
    
    filtered_data = translation_data[mask]
    filtered_shape = filtered_data.shape
    removed_count = original_shape[0] - filtered_shape[0]
    
    print(f"过滤方法: {method}, 阈值: {threshold}")
    print(f"过滤后平移数据形状: {filtered_shape}")
    print(f"过滤掉的异常值数量: {removed_count} ({removed_count/original_shape[0]*100:.2f}%)")
    
    # 使用GPU计算过滤后的统计信息
    if use_gpu and torch.cuda.is_available():
        filtered_tensor = torch.from_numpy(filtered_data).to(device)
        print(f"过滤后平移数据统计 - 最小值: {torch.min(filtered_tensor, dim=0)[0].cpu().numpy()}")
        print(f"过滤后平移数据统计 - 最大值: {torch.max(filtered_tensor, dim=0)[0].cpu().numpy()}")
        print(f"过滤后平移数据统计 - 均值: {torch.mean(filtered_tensor, dim=0).cpu().numpy()}")
        print(f"过滤后平移数据统计 - 标准差: {torch.std(filtered_tensor, dim=0).cpu().numpy()}")
    else:
        print(f"过滤后平移数据统计 - 最小值: {np.min(filtered_data, axis=0)}")
        print(f"过滤后平移数据统计 - 最大值: {np.max(filtered_data, axis=0)}")
        print(f"过滤后平移数据统计 - 均值: {np.mean(filtered_data, axis=0)}")
        print(f"过滤后平移数据统计 - 标准差: {np.std(filtered_data, axis=0)}")
    
    return filtered_data

def calculate_translation_stats(data_dir, output_dir):
    """
    计算平移数据的均值和方差
    
    Args:
        data_dir (str): 包含npz文件的目录
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有npz文件路径
    npz_files = glob.glob(os.path.join(data_dir, '**/*.npz'), recursive=True)
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 存储所有处理后的平移数据
    all_trans_data = []
    
    # 遍历所有文件
    for npz_file in tqdm(npz_files, desc="处理文件"):
        try:
            data = np.load(npz_file)
            
            # 检查必要的键是否存在
            if 'smplh:translation' not in data:
                print(f"跳过文件 {npz_file}，缺少必要的键 'smplh:translation'")
                continue
            
            # 获取平移数据
            translation = data['smplh:translation']  # 形状: (frames, 3)
            
            # 将平移数据从厘米转换为米
            translation = translation / 100.0
            
            # 检查帧数量是否为0
            if translation.shape[0] == 0:
                print(f"跳过文件 {npz_file}，帧数量为0")
                continue
                
            # 检查维度是否正确
            if len(translation.shape) != 2 or translation.shape[1] != 3:
                print(f"跳过文件 {npz_file}，translation维度不正确: {translation.shape}")
                continue
            
            # 按照seamless_sep.py的方式处理平移数据
            trans_each_file = translation.copy()
            # 处理x轴：减去初始x值
            trans_each_file[:, 0] = trans_each_file[:, 0] - trans_each_file[0, 0]
            # 处理z轴：减去初始z值
            trans_each_file[:, 2] = trans_each_file[:, 2] - trans_each_file[0, 2]
            # y轴保持不变
            
            # 添加到集合中
            all_trans_data.append(trans_each_file)
            
        except Exception as e:
            print(f"处理文件 {npz_file} 时出错: {str(e)}")
            continue
    
    if not all_trans_data:
        print("没有收集到有效的平移数据")
        return
    
    # 合并所有数据
    all_trans = np.concatenate(all_trans_data, axis=0)
    print(f"收集到的平移数据形状: {all_trans.shape}")
    
    # 异常值检测和过滤
    print("\n=== 开始异常值检测和过滤 ===")
    filtered_trans = filter_translation_outliers(all_trans, method='iqr', threshold=1.5, use_gpu=True)
    
    print("\n=== 开始计算统计量 ===")
    
    # 使用GPU加速计算均值和方差
    if torch.cuda.is_available():
        print("使用GPU计算均值和方差")
        
        # 分批处理大数据
        batch_size = min(5000000, len(filtered_trans))
        num_batches = (len(filtered_trans) + batch_size - 1) // batch_size
        
        total_sum = torch.zeros(filtered_trans.shape[1], device=device)
        total_sq_sum = torch.zeros(filtered_trans.shape[1], device=device)
        total_count = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(filtered_trans))
            batch_data = torch.from_numpy(filtered_trans[start_idx:end_idx]).to(device)
            
            total_sum += batch_data.sum(dim=0)
            total_sq_sum += (batch_data ** 2).sum(dim=0)
            total_count += batch_data.shape[0]
            
            del batch_data
            torch.cuda.empty_cache()
        
        # 计算均值和方差
        trans_mean = (total_sum / total_count).cpu().numpy()
        trans_var = (total_sq_sum / total_count).cpu().numpy() - (trans_mean ** 2)
        trans_var = np.maximum(trans_var, 1e-12)  # 防止负数方差
        trans_std = np.sqrt(trans_var)  # 计算标准差
    else:
        # 备用CPU计算
        print("使用CPU计算均值和标准差")
        trans_mean = np.mean(filtered_trans, axis=0)
        trans_std = np.std(filtered_trans, axis=0)  # 计算标准差
    
    print(f"平移均值: {trans_mean}")
    print(f"平移标准差: {trans_std}")
    
    # 确保结果是长度为3的向量
    assert trans_mean.shape == (3,), f"均值形状应为(3,)，实际为{trans_mean.shape}"
    assert trans_std.shape == (3,), f"标准差形状应为(3,)，实际为{trans_std.shape}"
    
    # 保存结果
    mean_path = os.path.join(output_dir, 'trans_mean.npy')
    std_path = os.path.join(output_dir, 'trans_std.npy')
    
    np.save(mean_path, trans_mean)
    np.save(std_path, trans_std)
    
    print(f"\n=== 计算完成 ===")
    print(f"平移均值已保存到: {mean_path}")
    print(f"平移标准差已保存到: {std_path}")
    print(f"均值形状: {trans_mean.shape}")
    print(f"标准差形状: {trans_std.shape}")
    print(f"最终平移数据点数: {filtered_trans.shape[0]}")

if __name__ == "__main__":
    # 数据集目录
    data_dir = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train'
    # 输出目录
    output_dir = '/home/embodied/yangchenyu/GestureLSM/trans_stats'
    
    calculate_translation_stats(data_dir, output_dir)