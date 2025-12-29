import numpy as np
import os
import glob
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
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
    # 对于IQR方法，即使有GPU也在CPU上计算，因为大量数据的分位数计算在GPU上可能导致内存不足
    # CPU的numpy在处理大数组分位数方面通常更稳定
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
        # 确定哪些数据点不是异常值
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        return mask

def detect_outliers_zscore(data, threshold=3.0, use_gpu=True):
    """使用Z-score方法检测异常值，支持分批处理大规模数据和多GPU并行计算"""
    if use_gpu and torch.cuda.is_available():
        # 对于大规模数据，分批处理以避免GPU内存不足
        print("使用多GPU并行计算Z-score异常值检测")
        
        # 先计算全局均值和标准差
        if len(devices) > 1:
            print(f"使用 {len(devices)} 张GPU卡并行计算均值和标准差")
            # 计算每个GPU处理的数据量
            per_gpu_size = len(data) // len(devices)
            
            # 初始化全局统计量
            total_sum = torch.zeros(data.shape[1], device=device)
            total_sq_sum = torch.zeros(data.shape[1], device=device)
            total_count = 0
            
            # 并行计算每个GPU上的数据统计
            for i, dev in enumerate(devices):
                start_idx = i * per_gpu_size
                end_idx = len(data) if i == len(devices) - 1 else (i + 1) * per_gpu_size
                
                # 将数据切片移到对应的GPU
                batch_data = torch.from_numpy(data[start_idx:end_idx]).to(dev)
                
                # 在当前GPU上计算统计量
                local_sum = batch_data.sum(dim=0)
                local_sq_sum = (batch_data ** 2).sum(dim=0)
                local_count = batch_data.shape[0]
                
                # 将结果移到主GPU并累加
                total_sum += local_sum.to(device)
                total_sq_sum += local_sq_sum.to(device)
                total_count += local_count
                
                # 释放内存
                del batch_data, local_sum, local_sq_sum
                torch.cuda.empty_cache()
        else:
            # 单GPU计算均值和标准差
            batch_size = min(5000000, len(data))  # 每批最多处理500万数据点
            num_batches = (len(data) + batch_size - 1) // batch_size
            
            print(f"使用单GPU分批处理 {num_batches} 批数据计算均值和标准差")
            
            # 分批计算均值和标准差（不存储所有数据）
            total_sum = torch.zeros(data.shape[1], device=device)
            total_sq_sum = torch.zeros(data.shape[1], device=device)
            total_count = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data))
                batch_data = torch.from_numpy(data[start_idx:end_idx]).to(device)
                
                # 累加和和平方和
                total_sum += batch_data.sum(dim=0)
                total_sq_sum += (batch_data ** 2).sum(dim=0)
                total_count += batch_data.shape[0]
                
                # 释放内存
                del batch_data
                torch.cuda.empty_cache()
        
        # 计算全局均值和标准差
        mean = total_sum / total_count
        var = (total_sq_sum / total_count) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-12))  # 防止负数方差
        
        # 处理标准差为0的情况
        std = torch.maximum(std, torch.tensor(1e-8, device=device))
        
        # 分批计算掩码，使用多GPU并行
        mask = np.ones(len(data), dtype=bool)
        batch_size = min(2000000, len(data))  # 每批处理200万数据点
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        print(f"使用多GPU并行计算掩码，共 {num_batches} 批")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            
            # 对于当前批次，再次划分到多个GPU
            if len(devices) > 1 and len(batch_data) > 10000:
                per_gpu_batch_size = len(batch_data) // len(devices)
                batch_masks = []
                
                for i, dev in enumerate(devices):
                    batch_start = i * per_gpu_batch_size
                    batch_end = len(batch_data) if i == len(devices) - 1 else (i + 1) * per_gpu_batch_size
                    
                    # 将批次数据切片移到对应的GPU
                    data_slice = torch.from_numpy(batch_data[batch_start:batch_end]).to(dev)
                    # 将均值和标准差移到对应的GPU
                    mean_dev = mean.to(dev)
                    std_dev = std.to(dev)
                    
                    # 计算Z-score
                    z_scores = torch.abs((data_slice - mean_dev) / std_dev)
                    
                    # 确定哪些数据点不是异常值
                    mask_slice = torch.all(z_scores < threshold, dim=1)
                    batch_masks.append(mask_slice.cpu().numpy())
                    
                    # 释放内存
                    del data_slice, mean_dev, std_dev, z_scores, mask_slice
                    torch.cuda.empty_cache()
                
                # 合并当前批次的掩码
                batch_mask = np.concatenate(batch_masks)
            else:
                # 单GPU处理小批次
                data_tensor = torch.from_numpy(batch_data).to(device)
                z_scores = torch.abs((data_tensor - mean) / std)
                batch_mask = torch.all(z_scores < threshold, dim=1).cpu().numpy()
                
                # 释放内存
                del data_tensor, z_scores
                torch.cuda.empty_cache()
            
            # 保存当前批次的掩码结果
            mask[start_idx:end_idx] = batch_mask
        
        return mask
    else:
        # CPU版本（保留原逻辑）
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # 处理标准差为0的情况
        std[std == 0] = 1e-8
        
        # 计算Z-score
        z_scores = np.abs((data - mean) / std)
        
        # 确定哪些数据点不是异常值（所有维度的Z-score都小于阈值）
        mask = np.all(z_scores < threshold, axis=1)
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
    else:  # 'zscore'
        mask = detect_outliers_zscore(translation_data, threshold, use_gpu)
    
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

def filter_pose_outliers(pose_data, method='zscore', threshold=3.0, use_gpu=True):
    """
    过滤姿态数据中的异常值，支持分批处理大规模数据
    
    Args:
        pose_data (np.ndarray): 姿态数据，形状为 (n_frames, 312)
        method (str): 异常值检测方法，'zscore' 或 'iqr'
        threshold (float): 异常值检测阈值
        use_gpu (bool): 是否使用GPU加速
    
    Returns:
        np.ndarray: 过滤异常值后的姿态数据
    """
    print(f"\n--- 处理姿态数据异常值 ---")
    print(f"过滤前姿态数据形状: {pose_data.shape}")
    
    # 根据选择的方法检测异常值，detect_outliers_zscore和detect_outliers_iqr已支持分批处理
    if method == 'zscore':
        print(f"过滤方法: zscore, 阈值: {threshold}")
        mask = detect_outliers_zscore(pose_data, threshold=threshold, use_gpu=use_gpu)
    else:
        print(f"过滤方法: iqr, 阈值: {threshold}")
        mask = detect_outliers_iqr(pose_data, threshold=threshold, use_gpu=use_gpu)
    
    # 应用过滤掩码
    filtered_data = pose_data[mask]
    outlier_count = len(pose_data) - len(filtered_data)
    outlier_percent = (outlier_count / len(pose_data)) * 100
    
    print(f"过滤后姿态数据形状: {filtered_data.shape}")
    print(f"过滤掉的异常值数量: {outlier_count} ({outlier_percent:.2f}%)")
    
    # 打印过滤后的统计信息（只计算和显示前3个维度，避免不必要的计算）
    first_3_dims = filtered_data[:, :3]
    print(f"过滤后姿态数据统计 - 最小值: {first_3_dims.min(axis=0)}")
    print(f"过滤后姿态数据统计 - 最大值: {first_3_dims.max(axis=0)}")
    print(f"过滤后姿态数据统计 - 均值: {first_3_dims.mean(axis=0)}")
    print(f"过滤后姿态数据统计 - 标准差: {first_3_dims.std(axis=0)}")
    
    return filtered_data

def calculate_mean_std(data_dir, output_dir):
    """
    计算seamless_interaction数据集的姿态和平移数据的均值和标准差
    根据用户要求，维度应该是55*6=330，其中55=1(global_orient)+21(body_pose)+15(left_hand)+15(right_hand)+3(fingers 22,23,24)
    原始数据是52*6=312维，需要扩展到330维，缺少的关节数据是22、23和24，也就是从22*6到24*6的维度数据要补0处理
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有npz文件路径
    npz_files = glob.glob(os.path.join(data_dir, '**/*.npz'), recursive=True)
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 存储所有姿态和平移数据
    all_pose_data = []
    all_trans_data = []
    
    # 遍历所有文件
    for npz_file in tqdm(npz_files, desc="处理文件"):
        try:
            data = np.load(npz_file)
            
            # 检查必要的键是否存在
            if 'smplh:body_pose' not in data or 'smplh:translation' not in data:
                print(f"跳过文件 {npz_file}，缺少必要的键")
                continue
            
            # 获取姿态数据
            body_pose = data['smplh:body_pose']  # 形状: (frames, 21, 3)
            global_orient = data['smplh:global_orient']  # 形状: (frames, 3)
            
            # 检查帧数量是否为0
            if body_pose.shape[0] == 0 or global_orient.shape[0] == 0:
                print(f"跳过文件 {npz_file}，帧数量为0")
                continue
                
            # 确保维度正确
            if len(body_pose.shape) != 3 or body_pose.shape[1:] != (21, 3):
                print(f"跳过文件 {npz_file}，body_pose维度不正确: {body_pose.shape}")
                continue
                
            if len(global_orient.shape) != 2 or global_orient.shape[1] != 3:
                print(f"跳过文件 {npz_file}，global_orient维度不正确: {global_orient.shape}")
                continue
            
            # 检查左右手数据是否存在
            left_hand_pose = data['smplh:left_hand_pose'] if 'smplh:left_hand_pose' in data else np.zeros((body_pose.shape[0], 15, 3), dtype=np.float32)
            right_hand_pose = data['smplh:right_hand_pose'] if 'smplh:right_hand_pose' in data else np.zeros((body_pose.shape[0], 15, 3), dtype=np.float32)
            
            # 获取平移数据 (3维)
            translation = data['smplh:translation']  # 形状: (frames, 3)
            
            # 初始化有效性标记
            N = body_pose.shape[0]
            is_valid = np.ones(N, dtype=bool)
            
            # 如果存在smplh:is_valid键，使用该键的标记进行过滤
            if "smplh:is_valid" in data:
                is_valid &= data["smplh:is_valid"]
            
            # 根据is_valid过滤数据
            if not np.all(is_valid):
                body_pose = body_pose[is_valid]
                global_orient = global_orient[is_valid]
                left_hand_pose = left_hand_pose[is_valid]
                right_hand_pose = right_hand_pose[is_valid]
                translation = translation[is_valid]
                N = body_pose.shape[0]
            
            # 将3D轴角表示正确转换为6D旋转表示
            # 1. 首先将numpy数组转换为PyTorch张量
            # 2. 使用axis_angle_to_matrix将3D轴角转换为旋转矩阵
            # 3. 使用matrix_to_rotation_6d将旋转矩阵转换为6D表示
            # 4. 最后将结果转换回numpy数组
            
            # 处理global_orient (1个关节，3维 -> 6维)
            # 需要添加一个维度使其形状为(frames, 1, 3)，然后处理完再reshape回(frames, 6)
            global_orient_tensor = torch.from_numpy(global_orient).unsqueeze(1).to(device)  # (frames, 1, 3)
            global_matrix = axis_angle_to_matrix(global_orient_tensor)  # (frames, 1, 3, 3)
            global_6d = matrix_to_rotation_6d(global_matrix).reshape(-1, 6).cpu().numpy()  # (frames, 6)
            
            # 处理body_pose (21个关节，每个3维 -> 6维)
            body_pose_tensor = torch.from_numpy(body_pose).to(device)  # (frames, 21, 3)
            body_matrix = axis_angle_to_matrix(body_pose_tensor)  # (frames, 21, 3, 3)
            body_6d = matrix_to_rotation_6d(body_matrix).reshape(body_pose.shape[0], -1).cpu().numpy()  # (frames, 21*6=126)
            
            # 处理left_hand_pose (15个关节，每个3维 -> 6维)
            left_hand_tensor = torch.from_numpy(left_hand_pose).to(device)  # (frames, 15, 3)
            left_hand_matrix = axis_angle_to_matrix(left_hand_tensor)  # (frames, 15, 3, 3)
            left_hand_6d = matrix_to_rotation_6d(left_hand_matrix).reshape(left_hand_pose.shape[0], -1).cpu().numpy()  # (frames, 15*6=90)
            
            # 处理right_hand_pose (15个关节，每个3维 -> 6维)
            right_hand_tensor = torch.from_numpy(right_hand_pose).to(device)  # (frames, 15, 3)
            right_hand_matrix = axis_angle_to_matrix(right_hand_tensor)  # (frames, 15, 3, 3)
            right_hand_6d = matrix_to_rotation_6d(right_hand_matrix).reshape(right_hand_pose.shape[0], -1).cpu().numpy()  # (frames, 15*6=90)
            
            # 组合所有6维表示的姿态数据（原始312维）
            # 总维度：1*6 + 21*6 + 15*6 + 15*6 = 6 + 126 + 90 + 90 = 312
            combined_pose_312 = np.concatenate([global_6d, body_6d, left_hand_6d, right_hand_6d], axis=1)
            
            # 扩展到330维，在关节22、23、24的位置补0
            # 关节22、23、24对应的是手指关节，每个关节6维，共18维
            # 在body_pose之后、left_hand_pose之前插入这18维的0值
            # 312维的结构：global_orient(6) + body_pose(126) + left_hand_pose(90) + right_hand_pose(90)
            # 330维的结构：global_orient(6) + body_pose(126) + fingers_22_23_24(18) + left_hand_pose(90) + right_hand_pose(90)
            
            # 创建18维的零值数据
            fingers_zero = np.zeros((N, 18), dtype=np.float32)
            
            # 在body_pose之后、left_hand_pose之前插入18维零值
            # body_pose结束位置：6 + 126 = 132
            combined_pose_330 = np.concatenate([
                combined_pose_312[:, :132],  # global_orient + body_pose
                fingers_zero,               # 关节22、23、24的零值
                combined_pose_312[:, 132:]   # left_hand_pose + right_hand_pose
            ], axis=1)
            
            # 添加到集合中
            all_pose_data.append(combined_pose_330)
            all_trans_data.append(translation)
            
        except Exception as e:
            print(f"处理文件 {npz_file} 时出错: {str(e)}")
            continue
    
    if not all_pose_data or not all_trans_data:
        print("没有收集到有效的数据")
        return
    
    # 合并所有数据
    all_pose = np.concatenate(all_pose_data, axis=0)
    all_trans = np.concatenate(all_trans_data, axis=0)
    
    print(f"收集到的姿态数据形状: {all_pose.shape}")
    print(f"收集到的平移数据形状: {all_trans.shape}")
    
    # 异常值检测和过滤
    print("\n=== 开始异常值检测和过滤 ===")
    
    # 过滤平移数据中的异常值
    print("\n--- 处理平移数据异常值 ---")
    # 对于平移数据，使用IQR方法更适合处理有极端值的情况
    filtered_trans = filter_translation_outliers(all_trans, method='iqr', threshold=1.5, use_gpu=True)
    
    # 过滤姿态数据中的异常值
    print("\n--- 处理姿态数据异常值 ---")
    # 对于姿态数据，使用Z-score方法
    filtered_pose = filter_pose_outliers(all_pose, method='zscore', threshold=3.0, use_gpu=True)
    
    print("\n=== 异常值检测和过滤完成 ===\n")
    
    # 使用多GPU并行计算均值和标准差
    if torch.cuda.is_available() and len(devices) > 1:
        print(f"使用 {len(devices)} 张GPU卡并行计算均值和标准差")
        
        # 定义一个函数用于在单个GPU上计算批次的统计量
        def compute_batch_stats(data, dev_idx):
            dev = devices[dev_idx]
            data_tensor = torch.from_numpy(data).to(dev)
            batch_sum = data_tensor.sum(dim=0)
            batch_sq_sum = (data_tensor ** 2).sum(dim=0)
            batch_count = data_tensor.shape[0]
            return (batch_sum.cpu(), batch_sq_sum.cpu(), batch_count)
        
        # 并行计算姿态数据的均值和标准差
        print("并行计算姿态数据统计量...")
        start_time = time.time()
        
        # 如果数据量很大，先分批，再对每批进行多GPU并行
        batch_size = min(10000000, len(filtered_pose))  # 每批最多处理1000万数据点
        num_batches = (len(filtered_pose) + batch_size - 1) // batch_size
        
        # 初始化全局统计量
        pose_total_sum = torch.zeros(filtered_pose.shape[1])
        pose_total_sq_sum = torch.zeros(filtered_pose.shape[1])
        pose_total_count = 0
        
        # 对每个批次进行并行处理
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(filtered_pose))
            batch_data = filtered_pose[start_idx:end_idx]
            
            print(f"处理姿态数据批次 {batch_idx+1}/{num_batches}, 数据大小: {batch_data.shape}")
            
            # 将批次数据划分到多个GPU
            per_gpu_size = len(batch_data) // len(devices)
            results = []
            
            # 并行计算每个GPU上的数据统计
            for i in range(len(devices)):
                start = i * per_gpu_size
                end = len(batch_data) if i == len(devices) - 1 else (i + 1) * per_gpu_size
                
                # 在当前GPU上计算统计量
                batch_sum, batch_sq_sum, batch_count = compute_batch_stats(batch_data[start:end], i)
                results.append((batch_sum, batch_sq_sum, batch_count))
            
            # 累加所有GPU的结果
            for batch_sum, batch_sq_sum, batch_count in results:
                pose_total_sum += batch_sum
                pose_total_sq_sum += batch_sq_sum
                pose_total_count += batch_count
            
            # 清理缓存
            torch.cuda.empty_cache()
        
        # 计算均值和方差
        pose_mean = (pose_total_sum / pose_total_count).numpy()
        pose_var = (pose_total_sq_sum / pose_total_count).numpy() - (pose_mean ** 2)
        pose_std = np.sqrt(np.maximum(pose_var, 1e-12))  # 防止负数方差
        
        print(f"姿态数据统计量计算完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 并行计算平移数据的均值和标准差
        print("并行计算平移数据统计量...")
        start_time = time.time()
        
        # 对平移数据也进行分批并行处理
        batch_size_trans = min(10000000, len(filtered_trans))
        num_batches_trans = (len(filtered_trans) + batch_size_trans - 1) // batch_size_trans
        
        # 初始化全局统计量
        trans_total_sum = torch.zeros(filtered_trans.shape[1])
        trans_total_sq_sum = torch.zeros(filtered_trans.shape[1])
        trans_total_count = 0
        
        # 对每个批次进行并行处理
        for batch_idx in range(num_batches_trans):
            start_idx = batch_idx * batch_size_trans
            end_idx = min((batch_idx + 1) * batch_size_trans, len(filtered_trans))
            batch_data = filtered_trans[start_idx:end_idx]
            
            print(f"处理平移数据批次 {batch_idx+1}/{num_batches_trans}, 数据大小: {batch_data.shape}")
            
            # 将批次数据划分到多个GPU
            per_gpu_size = len(batch_data) // len(devices)
            results = []
            
            # 并行计算每个GPU上的数据统计
            for i in range(len(devices)):
                start = i * per_gpu_size
                end = len(batch_data) if i == len(devices) - 1 else (i + 1) * per_gpu_size
                
                # 在当前GPU上计算统计量
                batch_sum, batch_sq_sum, batch_count = compute_batch_stats(batch_data[start:end], i)
                results.append((batch_sum, batch_sq_sum, batch_count))
            
            # 累加所有GPU的结果
            for batch_sum, batch_sq_sum, batch_count in results:
                trans_total_sum += batch_sum
                trans_total_sq_sum += batch_sq_sum
                trans_total_count += batch_count
            
            # 清理缓存
            torch.cuda.empty_cache()
        
        # 计算均值和方差
        trans_mean = (trans_total_sum / trans_total_count).numpy()
        trans_var = (trans_total_sq_sum / trans_total_count).numpy() - (trans_mean ** 2)
        trans_std = np.sqrt(np.maximum(trans_var, 1e-12))  # 防止负数方差
        
        print(f"平移数据统计量计算完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 处理标准差可能为0的情况
        pose_std = np.maximum(pose_std, 1e-8)
        trans_std = np.maximum(trans_std, 1e-8)
    
    # 使用GPU分批计算均值和标准差，避免内存不足
    elif torch.cuda.is_available():
        print("使用单GPU分批计算均值和标准差")
        
        # 计算姿态数据的均值和标准差
        batch_size = min(5000000, len(filtered_pose))
        num_batches = (len(filtered_pose) + batch_size - 1) // batch_size
        
        # 分批计算姿态数据均值和标准差
        pose_total_sum = torch.zeros(filtered_pose.shape[1], device=device)
        pose_total_sq_sum = torch.zeros(filtered_pose.shape[1], device=device)
        pose_total_count = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(filtered_pose))
            batch_data = torch.from_numpy(filtered_pose[start_idx:end_idx]).to(device)
            
            pose_total_sum += batch_data.sum(dim=0)
            pose_total_sq_sum += (batch_data ** 2).sum(dim=0)
            pose_total_count += batch_data.shape[0]
            
            del batch_data
            torch.cuda.empty_cache()
        
        # 计算均值和方差
        pose_mean = (pose_total_sum / pose_total_count).cpu().numpy()
        pose_var = (pose_total_sq_sum / pose_total_count).cpu().numpy() - (pose_mean ** 2)
        pose_std = np.sqrt(np.maximum(pose_var, 1e-12))  # 防止负数方差
        
        # 计算平移数据的均值和标准差
        batch_size_trans = min(5000000, len(filtered_trans))
        num_batches_trans = (len(filtered_trans) + batch_size_trans - 1) // batch_size_trans
        
        trans_total_sum = torch.zeros(filtered_trans.shape[1], device=device)
        trans_total_sq_sum = torch.zeros(filtered_trans.shape[1], device=device)
        trans_total_count = 0
        
        for i in range(num_batches_trans):
            start_idx = i * batch_size_trans
            end_idx = min((i + 1) * batch_size_trans, len(filtered_trans))
            batch_data = torch.from_numpy(filtered_trans[start_idx:end_idx]).to(device)
            
            trans_total_sum += batch_data.sum(dim=0)
            trans_total_sq_sum += (batch_data ** 2).sum(dim=0)
            trans_total_count += batch_data.shape[0]
            
            del batch_data
            torch.cuda.empty_cache()
        
        # 计算均值和方差
        trans_mean = (trans_total_sum / trans_total_count).cpu().numpy()
        trans_var = (trans_total_sq_sum / trans_total_count).cpu().numpy() - (trans_mean ** 2)
        trans_std = np.sqrt(np.maximum(trans_var, 1e-12))  # 防止负数方差
        
        # 处理标准差可能为0的情况
        pose_std = np.maximum(pose_std, 1e-8)
        trans_std = np.maximum(trans_std, 1e-8)
    else:
        # 备用CPU计算
        pose_mean = np.mean(filtered_pose, axis=0)
        pose_std = np.std(filtered_pose, axis=0)
        trans_mean = np.mean(filtered_trans, axis=0)
        trans_std = np.std(filtered_trans, axis=0)
        
        # 处理标准差可能为0的情况
        pose_std = np.maximum(pose_std, 1e-8)
        trans_std = np.maximum(trans_std, 1e-8)
    
    print(f"姿态均值形状: {pose_mean.shape}")
    print(f"姿态标准差形状: {pose_std.shape}")
    print(f"平移均值形状: {trans_mean.shape}")
    print(f"平移标准差形状: {trans_std.shape}")
    
    # 保存为npy文件
    np.save(os.path.join(output_dir, 'seamless_2_330_mean.npy'), pose_mean.astype(np.float32))
    np.save(os.path.join(output_dir, 'seamless_2_330_std.npy'), pose_std.astype(np.float32))
    np.save(os.path.join(output_dir, 'seamless_2_trans_mean.npy'), trans_mean)
    np.save(os.path.join(output_dir, 'seamless_2_trans_std.npy'), trans_std)
    
    print("\n=== 过滤后的统计信息摘要 ===")
    print(f"最终姿态数据点数: {filtered_pose.shape[0]}")
    print(f"最终平移数据点数: {filtered_trans.shape[0]}")
    
    # 使用GPU计算统计摘要
    if torch.cuda.is_available():
        pose_mean_tensor = torch.from_numpy(pose_mean).to(device)
        pose_std_tensor = torch.from_numpy(pose_std).to(device)
        print(f"姿态数据均值范围: [{torch.min(pose_mean_tensor).item():.6f}, {torch.max(pose_mean_tensor).item():.6f}]")
        print(f"姿态数据标准差范围: [{torch.min(pose_std_tensor).item():.6f}, {torch.max(pose_std_tensor).item():.6f}]")
        print(f"平移数据均值: {trans_mean}")
        print(f"平移数据标准差: {trans_std}")
    else:
        print(f"姿态数据均值范围: [{np.min(pose_mean):.6f}, {np.max(pose_mean):.6f}]")
        print(f"姿态数据标准差范围: [{np.min(pose_std):.6f}, {np.max(pose_std):.6f}]")
        print(f"平移数据均值: {trans_mean}")
        print(f"平移数据标准差: {trans_std}")
    
    print(f"已保存均值和标准差文件到 {output_dir}")
    print(f"姿态数据维度: 330 (55个关节 × 6维)")
    print(f"平移数据维度: 3")

if __name__ == "__main__":
    # 数据集目录
    data_dir = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction'
    # 输出目录
    output_dir = '/home/embodied/yangchenyu/GestureLSM/mean_std_seamless'
    
    calculate_mean_std(data_dir, output_dir)