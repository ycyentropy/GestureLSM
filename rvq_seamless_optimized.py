import argparse
import json
import os
import time
import gc
import warnings
import psutil
import platform
import numpy as np

# 抑制pynvml的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="The pynvml package is deprecated")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lazy_window_dataset import LazySeamlessInteractionWindowDataset
from models.vq.model import RVQVAE
try:
    import nvidia_ml_py3 as pynvml
except ImportError:
    import pynvml


def fast_collate_fn(batch):
    """
    快速批处理函数，用于处理可能大小不一致的数据
    
    Args:
        batch: 批次数据，每个元素是一个字典
        
    Returns:
        result: 处理后的批次数据
    """
    if len(batch) == 0:
        return {}
    
    # 获取所有键
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        # 对于每个键，收集所有样本的值
        values = [item[key] for item in batch]
        
        # 处理不同类型的值
        if isinstance(values[0], np.ndarray):
            # 如果是numpy数组，转换为tensor
            # 对于pose数据，可能需要填充到相同长度
            if key == 'pose' and len(set(v.shape for v in values)) > 1:
                # 找到最大长度
                max_len = max(v.shape[0] for v in values)
                # 填充到相同长度
                padded_values = []
                for v in values:
                    if v.shape[0] < max_len:
                        padding = np.zeros((max_len - v.shape[0],) + v.shape[1:], dtype=v.dtype)
                        padded_v = np.concatenate([v, padding], axis=0)
                        padded_values.append(padded_v)
                    else:
                        padded_values.append(v)
                result[key] = torch.stack([torch.from_numpy(v) for v in padded_values])
            else:
                # 直接堆叠
                result[key] = torch.stack([torch.from_numpy(v) for v in values])
        elif isinstance(values[0], torch.Tensor):
            # 如果是tensor，可能需要填充到相同长度
            if key == 'pose' and len(set(v.shape for v in values)) > 1:
                # 找到最大长度
                max_len = max(v.shape[0] for v in values)
                # 填充到相同长度
                padded_values = []
                for v in values:
                    if v.shape[0] < max_len:
                        padding = torch.zeros((max_len - v.shape[0],) + v.shape[1:], dtype=v.dtype, device=v.device)
                        padded_v = torch.cat([v, padding], axis=0)
                        padded_values.append(padded_v)
                    else:
                        padded_values.append(v)
                result[key] = torch.stack(padded_values)
            else:
                # 直接堆叠
                result[key] = torch.stack(values)
        else:
            # 其他类型，直接堆叠
            result[key] = torch.tensor(values)
    
    return result


def get_system_info():
    """获取系统信息"""
    info = {
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }
    return info


# 获取系统信息
system_info = get_system_info()


def dynamic_memory_management(model, device, args, logger=None):
    """
    动态内存管理
    
    Args:
        model: 模型
        device: 设备
        args: 训练参数
        logger: 日志记录器
    """
    if logger:
        logger.info("执行动态内存管理...")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 设置内存分配策略
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        memory_fraction = getattr(args, 'memory_fraction', 0.9)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        if logger:
            logger.info(f"设置GPU内存分配比例为: {memory_fraction}")
    
    # 启用内存映射
    if hasattr(torch.cuda, 'set_memory_strategy'):
        torch.cuda.set_memory_strategy('memory_mapping')
        if logger:
            logger.info("启用GPU内存映射策略")
    
    # 监控初始内存使用
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        if logger:
            logger.info(f"初始GPU内存使用: {initial_memory:.2f} GB")
    
    if logger:
        logger.info("动态内存管理完成")


def profile_memory_usage(model, data_loader, device, logger=None):
    """
    分析内存使用情况
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        logger: 日志记录器
    """
    if logger:
        logger.info("开始分析内存使用情况...")
    
    model.eval()
    
    # 记录初始内存
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
    
    # 运行一个批次并记录内存使用
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 3:  # 只测试3个批次
                break
                
            gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                pred_motion, loss_commit, perplexity = model(gt_motion).values()
            
            # 记录峰值内存
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            current_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            
            if logger:
                logger.info(f"批次 {i+1}: 当前内存 {current_memory:.2f} GB, 峰值内存 {peak_memory:.2f} GB")
    
    # 计算模型参数大小
    model_params = sum(p.numel() for p in model.parameters())
    model_size = model_params * 4 / (1024**2)  # 假设float32，转换为MB
    
    if logger:
        logger.info(f"模型参数数量: {model_params:,}")
        logger.info(f"模型大小: {model_size:.2f} MB")
        logger.info(f"初始内存: {initial_memory:.2f} GB")
        logger.info(f"峰值内存: {peak_memory:.2f} GB")
        logger.info("内存使用分析完成")


def monitor_gpu_memory(gpu_id, logger=None):
    """
    监控GPU内存使用情况
    
    Args:
        gpu_id: GPU ID
        logger: 日志记录器
        
    Returns:
        memory_info: 内存使用信息
    """
    try:
        # 初始化pynvml
        pynvml.nvmlInit()
        
        # 获取GPU句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        # 获取内存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # 获取GPU利用率
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        memory_info = {
            'total': mem_info.total / (1024**3),  # GB
            'used': mem_info.used / (1024**3),    # GB
            'free': mem_info.free / (1024**3),    # GB
            'usage_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_util': util_rates.gpu
        }
        
        if logger:
            logger.info(f"GPU {gpu_id} 内存使用: {memory_info['used']:.2f}/{memory_info['total']:.2f} GB "
                       f"({memory_info['usage_percent']:.1f}%), GPU利用率: {memory_info['gpu_util']}%")
        
        return memory_info
    except Exception as e:
        if logger:
            logger.warning(f"无法获取GPU {gpu_id} 内存信息: {e}")
        
        # 回退到PyTorch的内存监控
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)   # GB
            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
            
            memory_info = {
                'total': memory_total,
                'used': memory_allocated,
                'reserved': memory_reserved,
                'free': memory_total - memory_reserved,
                'usage_percent': (memory_allocated / memory_total) * 100,
                'gpu_util': None
            }
            
            if logger:
                logger.info(f"GPU {gpu_id} 内存使用(回退): {memory_info['used']:.2f}/{memory_info['total']:.2f} GB "
                           f"({memory_info['usage_percent']:.1f}%)")
            
            return memory_info
        else:
            return None


def save_performance_report(perf_monitor, output_dir, logger=None):
    """
    保存性能报告
    
    Args:
        perf_monitor: 性能监控器
        output_dir: 输出目录
        logger: 日志记录器
    """
    if logger:
        logger.info("保存性能报告...")
    
    # 计算性能统计
    if len(perf_monitor['iter_times']) > 1:
        total_time = perf_monitor['iter_times'][-1] - perf_monitor['iter_times'][0]
        avg_iter_time = total_time / len(perf_monitor['iter_times'])
        
        if len(perf_monitor['throughput_history']) > 0:
            avg_throughput = sum(perf_monitor['throughput_history']) / len(perf_monitor['throughput_history'])
        else:
            avg_throughput = 0
    else:
        avg_iter_time = 0
        avg_throughput = 0
    
    # 创建性能报告
    report = {
        'total_time': total_time,
        'avg_iter_time': avg_iter_time,
        'avg_throughput': avg_throughput,
        'best_loss': perf_monitor['best_loss'],
        'total_iterations': len(perf_monitor['iter_times']),
        'auto_tune_history': perf_monitor['tune_history']
    }
    
    # 保存报告
    with open(f"{output_dir}/performance_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    if logger:
        logger.info(f"性能报告已保存到 {output_dir}/performance_report.json")


def plot_performance_curves(perf_monitor, output_dir, logger=None):
    """
    绘制性能曲线
    
    Args:
        perf_monitor: 性能监控器
        output_dir: 输出目录
        logger: 日志记录器
    """
    try:
        import matplotlib.pyplot as plt
        
        if logger:
            logger.info("绘制性能曲线...")
        
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制损失曲线
        if len(perf_monitor['loss_history']) > 0:
            axs[0, 0].plot(perf_monitor['loss_history'])
            axs[0, 0].set_title('Training Loss')
            axs[0, 0].set_xlabel('Iteration')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].grid(True)
        
        # 绘制吞吐量曲线
        if len(perf_monitor['throughput_history']) > 0:
            axs[0, 1].plot(perf_monitor['throughput_history'])
            axs[0, 1].set_title('Throughput')
            axs[0, 1].set_xlabel('Iteration')
            axs[0, 1].set_ylabel('Samples/sec')
            axs[0, 1].grid(True)
        
        # 绘制GPU内存使用曲线
        if len(perf_monitor['gpu_memory_history']) > 0:
            axs[1, 0].plot(perf_monitor['gpu_memory_history'])
            axs[1, 0].set_title('GPU Memory Usage')
            axs[1, 0].set_xlabel('Iteration')
            axs[1, 0].set_ylabel('Memory (GB)')
            axs[1, 0].grid(True)
        
        # 绘制GPU利用率曲线
        if len(perf_monitor['gpu_util_history']) > 0:
            axs[1, 1].plot(perf_monitor['gpu_util_history'])
            axs[1, 1].set_title('GPU Utilization')
            axs[1, 1].set_xlabel('Iteration')
            axs[1, 1].set_ylabel('Utilization (%)')
            axs[1, 1].grid(True)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_curves.png", dpi=300)
        plt.close()
        
        if logger:
            logger.info(f"性能曲线已保存到 {output_dir}/performance_curves.png")
            
    except ImportError:
        if logger:
            logger.warning("matplotlib未安装，跳过性能曲线绘制")
    except Exception as e:
        if logger:
            logger.error(f"绘制性能曲线时出错: {e}")


def get_logger(out_dir, rank):
    """
    获取日志记录器
    
    Args:
        out_dir: 输出目录
        rank: 当前进程排名
    
    Returns:
        logger: 日志记录器
    """
    import logging
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(f"{out_dir}/log.txt")
    logger.addHandler(handler)
    
    return logger


def auto_tune_batch_size(model, device, initial_batch_size, max_batch_size, train_loader, world_size, rank, args, logger=None):
    """
    自动调整批次大小
    
    Args:
        model: 模型
        device: 设备
        initial_batch_size: 初始批次大小
        max_batch_size: 最大批次大小
        train_loader: 训练数据加载器
        world_size: 分布式训练中的进程总数
        rank: 当前进程排名
        args: 训练参数
        logger: 日志记录器
    
    Returns:
        optimal_batch_size: 最佳批次大小
    """
    import time
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 预热
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 预热3个批次
            break
        gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)
        pred_motion, _, _ = model(gt_motion).values()
        loss = pred_motion.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 基准测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 测试10个批次
            break
        gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)
        pred_motion, _, _ = model(gt_motion).values()
        loss = pred_motion.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算性能指标
    elapsed_time = end_time - start_time
    batch_time = elapsed_time / 10
    samples_per_sec = args.batch_size * world_size / batch_time
    
    metrics = {
        'batch_time': batch_time,
        'samples_per_sec': samples_per_sec,
        'throughput': samples_per_sec * world_size
    }
    
    if rank == 0:
        print(f"分布式训练基准测试结果:")
        print(f"  批次时间: {batch_time:.4f}s")
        print(f"  每秒处理样本数: {samples_per_sec:.2f}")
        print(f"  总吞吐量: {metrics['throughput']:.2f} 样本/秒")
    
    return initial_batch_size  # 返回初始批次大小，暂时不进行自动调整


def setup_performance_monitoring(args, logger=None):
    """
    设置性能监控
    
    参数:
        args: 训练参数
        logger: 日志记录器
        
    返回:
        perf_monitor: 性能监控器
    """
    if logger:
        logger.info("设置性能监控...")
    
    # 创建性能监控器
    perf_monitor = {
        'start_time': time.time(),
        'iter_times': [],
        'gpu_memory_history': [],
        'gpu_util_history': [],
        'throughput_history': [],
        'loss_history': [],
        'best_loss': float('inf'),
        'last_save_time': time.time(),
        'auto_save_interval': args.auto_save_interval if hasattr(args, 'auto_save_interval') else 3600,  # 默认1小时
        'performance_log_interval': args.performance_log_interval if hasattr(args, 'performance_log_interval') else 100,  # 默认100次迭代
        'auto_tune_enabled': args.auto_tune_performance if hasattr(args, 'auto_tune_performance') else False,
        'tune_history': []
    }
    
    if logger:
        logger.info("性能监控设置完成")
    
    return perf_monitor


def update_performance_monitor(perf_monitor, nb_iter, loss, batch_size, device, logger=None):
    """
    更新性能监控
    
    参数:
        perf_monitor: 性能监控器
        nb_iter: 当前迭代次数
        loss: 当前损失
        batch_size: 当前批次大小
        device: 设备
        logger: 日志记录器
        
    返回:
        perf_monitor: 更新后的性能监控器
    """
    current_time = time.time()
    
    # 计算迭代时间
    if len(perf_monitor['iter_times']) > 0:
        iter_time = current_time - perf_monitor['iter_times'][-1]
    else:
        iter_time = current_time - perf_monitor['start_time']
    
    perf_monitor['iter_times'].append(current_time)
    
    # 计算吞吐量
    throughput = batch_size / iter_time
    perf_monitor['throughput_history'].append(throughput)
    
    # 记录损失
    perf_monitor['loss_history'].append(loss)
    
    # 更新最佳损失
    if loss < perf_monitor['best_loss']:
        perf_monitor['best_loss'] = loss
    
    # 监控GPU内存
    gpu_memory = monitor_gpu_memory(device.index)
    if gpu_memory:
        perf_monitor['gpu_memory_history'].append({
            'time': current_time,
            'utilization': gpu_memory['utilization'],
            'used': gpu_memory['used'],
            'total': gpu_memory['total']
        })
    
    # 监控GPU利用率
    gpu_util = monitor_gpu_utilization(device)
    if gpu_util:
        perf_monitor['gpu_util_history'].append({
            'time': current_time,
            'utilization': gpu_util['utilization'],
            'memory_util': gpu_util['memory_util']
        })
    
    # 定期记录性能日志
    if nb_iter % perf_monitor['performance_log_interval'] == 0 and logger:
        avg_throughput = sum(perf_monitor['throughput_history'][-perf_monitor['performance_log_interval']:]) / min(len(perf_monitor['throughput_history']), perf_monitor['performance_log_interval'])
        avg_loss = sum(perf_monitor['loss_history'][-perf_monitor['performance_log_interval']:]) / min(len(perf_monitor['loss_history']), perf_monitor['performance_log_interval'])
        
        if perf_monitor['gpu_memory_history']:
            avg_gpu_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history'][-perf_monitor['performance_log_interval']:]]) / min(len(perf_monitor['gpu_memory_history']), perf_monitor['performance_log_interval'])
        else:
            avg_gpu_util = 0
        
        logger.info(f"性能监控 - 迭代 {nb_iter}: 吞吐量 {avg_throughput:.2f} samples/s, 平均损失 {avg_loss:.5f}, GPU利用率 {avg_gpu_util:.1f}%")
    
    # 自动保存检查点
    if current_time - perf_monitor['last_save_time'] > perf_monitor['auto_save_interval']:
        perf_monitor['last_save_time'] = current_time
        if logger:
            logger.info(f"定期自动保存检查点 (每 {perf_monitor['auto_save_interval']} 秒)")
    
    return perf_monitor


def auto_tune_performance(perf_monitor, args, model, optimizer, device, logger=None):
    """
    自动调优性能参数
    
    参数:
        perf_monitor: 性能监控器
        args: 训练参数
        model: 模型
        optimizer: 优化器
        device: 设备
        logger: 日志记录器
        
    返回:
        args: 更新后的训练参数
        tune_actions: 执行的调优操作列表
    """
    tune_actions = []
    
    if not perf_monitor['auto_tune_enabled']:
        return args, tune_actions
    
    # 只在足够的历史数据后进行调优
    if len(perf_monitor['throughput_history']) < 10:
        return args, tune_actions
    
    # 计算最近的平均吞吐量和GPU利用率
    recent_throughput = sum(perf_monitor['throughput_history'][-10:]) / 10
    recent_loss = sum(perf_monitor['loss_history'][-10:]) / 10
    
    if perf_monitor['gpu_util_history']:
        recent_gpu_util = sum([u['utilization'] for u in perf_monitor['gpu_util_history'][-10:]]) / 10
    else:
        recent_gpu_util = 0
    
    if perf_monitor['gpu_memory_history']:
        recent_mem_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history'][-10:]]) / 10
    else:
        recent_mem_util = 0
    
    # 调优规则
    # 1. 如果GPU利用率低且内存使用率低，尝试增加批次大小
    if recent_gpu_util < 70 and recent_mem_util < 70 and hasattr(args, 'batch_size'):
        if args.batch_size < args.max_batch_size if hasattr(args, 'max_batch_size') else args.batch_size * 2:
            old_batch_size = args.batch_size
            args.batch_size = min(args.batch_size * 2, args.max_batch_size if hasattr(args, 'max_batch_size') else args.batch_size * 2)
            tune_actions.append(f"增加批次大小: {old_batch_size} -> {args.batch_size}")
            
            if logger:
                logger.info(f"自动调优: 增加批次大小 {old_batch_size} -> {args.batch_size} (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
    
    # 2. 如果GPU利用率高但内存使用率高，尝试启用梯度检查点
    elif recent_gpu_util > 80 and recent_mem_util > 85 and not args.gradient_checkpointing:
        args.gradient_checkpointing = True
        tune_actions.append("启用梯度检查点")
        
        if logger:
            logger.info(f"自动调优: 启用梯度检查点 (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
        
        # 如果模型支持梯度检查点，启用它
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
    
    # 3. 如果GPU利用率低但内存使用率高，尝试减少批次大小
    elif recent_gpu_util < 50 and recent_mem_util > 85 and hasattr(args, 'batch_size') and args.batch_size > 8:
        old_batch_size = args.batch_size
        args.batch_size = max(args.batch_size // 2, 8)
        tune_actions.append(f"减少批次大小: {old_batch_size} -> {args.batch_size}")
        
        if logger:
            logger.info(f"自动调优: 减少批次大小 {old_batch_size} -> {args.batch_size} (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
    
    # 4. 如果损失增加，尝试降低学习率
    elif len(perf_monitor['loss_history']) >= 20:
        recent_loss = sum(perf_monitor['loss_history'][-10:]) / 10
        older_loss = sum(perf_monitor['loss_history'][-20:-10]) / 10
        
        if recent_loss > older_loss * 1.1:  # 损失增加了10%以上
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.9  # 降低10%
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            tune_actions.append(f"降低学习率: {old_lr:.6f} -> {new_lr:.6f}")
            
            if logger:
                logger.info(f"自动调优: 降低学习率 {old_lr:.6f} -> {new_lr:.6f} (损失增加: {older_loss:.5f} -> {recent_loss:.5f})")
    
    # 记录调优历史
    if tune_actions:
        perf_monitor['tune_history'].append({
            'time': time.time(),
            'actions': tune_actions,
            'throughput': recent_throughput,
            'gpu_util': recent_gpu_util,
            'mem_util': recent_mem_util,
            'loss': recent_loss
        })
    
    return args, tune_actions


def save_performance_report(perf_monitor, args, save_path, logger=None):
    """
    保存性能报告
    
    参数:
        perf_monitor: 性能监控器
        args: 训练参数
        save_path: 保存路径
        logger: 日志记录器
    """
    if logger:
        logger.info(f"保存性能报告到 {save_path}")
    
    # 计算总体统计
    total_time = time.time() - perf_monitor['start_time']
    total_iters = len(perf_monitor['iter_times'])
    
    if total_iters > 0:
        avg_iter_time = total_time / total_iters
        avg_throughput = sum(perf_monitor['throughput_history']) / len(perf_monitor['throughput_history']) if perf_monitor['throughput_history'] else 0
        avg_loss = sum(perf_monitor['loss_history']) / len(perf_monitor['loss_history']) if perf_monitor['loss_history'] else 0
    else:
        avg_iter_time = 0
        avg_throughput = 0
        avg_loss = 0
    
    if perf_monitor['gpu_memory_history']:
        avg_gpu_mem_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history']]) / len(perf_monitor['gpu_memory_history'])
    else:
        avg_gpu_mem_util = 0
    
    if perf_monitor['gpu_util_history']:
        avg_gpu_util = sum([u['utilization'] for u in perf_monitor['gpu_util_history']]) / len(perf_monitor['gpu_util_history'])
    else:
        avg_gpu_util = 0
    
    # 创建报告
    report = {
        'training_args': vars(args),
        'performance_summary': {
            'total_time': total_time,
            'total_iters': total_iters,
            'avg_iter_time': avg_iter_time,
            'avg_throughput': avg_throughput,
            'avg_loss': avg_loss,
            'best_loss': perf_monitor['best_loss'],
            'avg_gpu_memory_util': avg_gpu_mem_util,
            'avg_gpu_util': avg_gpu_util,
        },
        'tune_history': perf_monitor['tune_history'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存报告
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4, sort_keys=True)
    
    if logger:
        logger.info(f"性能报告已保存: 平均吞吐量 {avg_throughput:.2f} samples/s, 平均GPU利用率 {avg_gpu_util:.1f}%, 最佳损失 {perf_monitor['best_loss']:.5f}")


def plot_performance_curves(perf_monitor, save_path, logger=None):
    """
    绘制性能曲线
    
    参数:
        perf_monitor: 性能监控器
        save_path: 保存路径
        logger: 日志记录器
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        if logger:
            logger.info(f"绘制性能曲线到 {save_path}")
        
        # 创建子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 吞吐量曲线
        if perf_monitor['throughput_history']:
            axs[0, 0].plot(perf_monitor['throughput_history'])
            axs[0, 0].set_title('Throughput (samples/s)')
            axs[0, 0].set_xlabel('Iteration')
            axs[0, 0].set_ylabel('Throughput')
            axs[0, 0].grid(True)
        
        # 2. 损失曲线
        if perf_monitor['loss_history']:
            axs[0, 1].plot(perf_monitor['loss_history'])
            axs[0, 1].set_title('Training Loss')
            axs[0, 1].set_xlabel('Iteration')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].grid(True)
        
        # 3. GPU内存利用率曲线
        if perf_monitor['gpu_memory_history']:
            times = [m['time'] - perf_monitor['start_time'] for m in perf_monitor['gpu_memory_history']]
            utils = [m['utilization'] for m in perf_monitor['gpu_memory_history']]
            axs[1, 0].plot(times, utils)
            axs[1, 0].set_title('GPU Memory Utilization')
            axs[1, 0].set_xlabel('Time (s)')
            axs[1, 0].set_ylabel('Utilization (%)')
            axs[1, 0].grid(True)
        
        # 4. GPU利用率曲线
        if perf_monitor['gpu_util_history']:
            times = [u['time'] - perf_monitor['start_time'] for u in perf_monitor['gpu_util_history']]
            utils = [u['utilization'] for u in perf_monitor['gpu_util_history']]
            axs[1, 1].plot(times, utils)
            axs[1, 1].set_title('GPU Utilization')
            axs[1, 1].set_xlabel('Time (s)')
            axs[1, 1].set_ylabel('Utilization (%)')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        if logger:
            logger.info(f"性能曲线已保存")
    except ImportError:
        if logger:
            logger.warning("matplotlib未安装，无法绘制性能曲线")
    except Exception as e:
        if logger:
            logger.error(f"绘制性能曲线时出错: {e}")


def main(args):
    """主函数"""
    
    # 检查是否在分布式环境中运行
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    # 初始化is_main_process
    is_main_process = True  # 默认值，稍后会被更新
    
    # 初始化local_rank
    local_rank = 0
    
    if is_distributed:
        # 从环境变量获取分布式参数
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        
        # 初始化分布式训练环境
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 确保每个进程使用不同的GPU
        torch.cuda.set_device(local_rank)
        
        # 更新is_main_process
        is_main_process = rank == 0
        
        # 打印GPU分配信息
        if is_main_process:
            print(f"分布式训练模式: 使用 {world_size} 个GPU")
            print(f"当前进程 (rank {rank}) 使用 GPU {local_rank}")
            print(f"系统配置: CPU核心数={system_info['cpu_count']}, GPU数量={system_info['gpu_count']}, 内存={system_info['memory_total']:.1f}GB")
    else:
        # 非分布式模式
        rank = 0
        world_size = 1
        is_main_process = True
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print("警告: 未检测到分布式环境，运行在单GPU模式")
        print(f"系统配置: CPU核心数={system_info['cpu_count']}, GPU数量={system_info['gpu_count']}, 内存={system_info['memory_total']:.1f}GB")
    
    # 设置local_rank到args
    args.local_rank = local_rank
    
    # 只有主进程打印信息
    is_main_process = rank == 0
    
    torch.manual_seed(args.seed)
    
    # 启用cudnn优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建输出目录（只有主进程创建）
    if is_main_process:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = get_logger(args.out_dir, rank)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
        logger.info(f"系统配置: {system_info}")
    else:
        logger = None
        writer = None
    
    # 设置性能监控
    perf_monitor = setup_performance_monitoring(args, logger) if is_main_process else None
    
    # 创建数据集
    if is_main_process:
        print("Building dataset...")
    
    # 处理多长度训练参数
    split = 'train'  # 定义split变量
    if args.multi_length_training is not None:
        if split == 'train':
            multi_length_ratios = args.multi_length_training
            if is_main_process:
                print(f"使用多长度训练比例: {multi_length_ratios}")
        else:
            # 验证集和测试集只使用1.0比例，以匹配缓存文件
            multi_length_ratios = [1.0]
    else:
        multi_length_ratios = [1.0]
        if is_main_process:
            print("使用单长度训练比例: [1.0]")
    
    # 处理缓存路径别名
    if args.cache_train is not None:
        args.cache_path = args.cache_train
    if args.cache_val is not None:
        args.val_cache_path = args.cache_val

    # 如果提供了缓存路径，自动启用缓存
    if args.cache_path is not None or args.val_cache_path is not None or args.cache_train is not None or args.cache_val is not None:
        args.use_cache = True
        if is_main_process:
            print(f"检测到缓存路径，自动启用缓存模式")
            print(f"缓存路径 - 训练: {args.cache_path or args.cache_train}, 验证: {args.val_cache_path or args.cache_val}")

    # 确保缓存路径正确设置
    if args.use_cache:
        if args.cache_path is None and args.cache_train is not None:
            args.cache_path = args.cache_train
        if args.val_cache_path is None and args.cache_val is not None:
            args.val_cache_path = args.cache_val

    # 确定缓存路径
    if args.use_cache:
        train_cache_path = args.cache_path
        val_cache_path = args.val_cache_path if args.val_cache_path is not None else args.cache_path

        if train_cache_path is None or val_cache_path is None:
            # 自动生成缓存路径
            cache_dir = os.path.join(os.path.dirname(args.data_path), 'window_params')
            os.makedirs(cache_dir, exist_ok=True)
            train_cache_path = os.path.join(cache_dir, f"window_params_train_ws{args.window_size}_ws{args.window_stride}.pkl")
            val_cache_path = os.path.join(cache_dir, f"window_params_val_ws{args.window_size}_ws{args.window_stride}.pkl")
    else:
        train_cache_path = None
        val_cache_path = None
    
    # 创建训练数据集
    if args.use_cache and train_cache_path is not None and os.path.exists(train_cache_path):
        # 使用缓存加载训练数据集
        if is_main_process:
            print(f"使用训练集缓存文件: {train_cache_path}")
        
        try:
            from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
            train_dataset = CachedLazySeamlessInteractionWindowDataset(
                data_path=args.data_path,
                split='train',
                window_size=args.window_size,
                window_stride=args.window_stride,
                multi_length_training=multi_length_ratios,
                load_video=False,
                load_audio=False,
                max_samples=args.max_samples,
                cache_path=train_cache_path
            )
            if len(train_dataset) == 0:
                raise ValueError("训练数据集为空，缓存可能损坏")
        except Exception as e:
            if is_main_process:
                print(f"加载训练集缓存失败: {e}")
                print("回退到常规数据加载方式...")
            args.use_cache = False
    
    if not args.use_cache or train_cache_path is None or not os.path.exists(train_cache_path):
        # 使用常规方式加载训练数据集
        if is_main_process:
            print("使用常规方式加载训练数据集...")
        
        train_dataset = LazySeamlessInteractionWindowDataset(
            data_path=args.data_path,
            split='train',
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=multi_length_ratios,
            load_video=False,
            load_audio=False,
            max_samples=args.max_samples
        )
    
    # 创建验证数据集
    if args.use_cache and val_cache_path is not None and os.path.exists(val_cache_path):
        # 使用缓存加载验证数据集
        if is_main_process:
            print(f"使用验证集缓存文件: {val_cache_path}")
        
        try:
            from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset
            val_dataset = CachedLazySeamlessInteractionWindowDataset(
                data_path=args.data_path,
                split='val',
                window_size=args.window_size,
                window_stride=args.window_stride,
                multi_length_training=[1.0],  # 验证集只使用1.0比例
                load_video=False,
                load_audio=False,
                max_samples=args.max_samples,
                cache_path=val_cache_path
            )
            if len(val_dataset) == 0:
                raise ValueError("验证数据集为空，缓存可能损坏")
        except Exception as e:
            if is_main_process:
                print(f"加载验证集缓存失败: {e}")
                print("回退到常规数据加载方式...")
            args.use_cache = False
    
    if not args.use_cache or val_cache_path is None or not os.path.exists(val_cache_path):
        # 使用常规方式加载验证数据集
        if is_main_process:
            print("使用常规方式加载验证数据集...")
        
        val_dataset = LazySeamlessInteractionWindowDataset(
            data_path=args.data_path,
            split='val',  # 使用val作为验证集
            window_size=args.window_size,
            window_stride=args.window_stride,
            multi_length_training=[1.0],  # 验证集只使用1.0比例
            load_video=False,
            load_audio=False,
            max_samples=args.max_samples
        )
    
    if is_main_process:
        print(f"使用 {len(train_dataset)} 个训练窗口和 {len(val_dataset)} 个验证窗口进行训练")
    
    # 选择批处理函数
    # 使用自定义的fast_collate_fn来处理可能大小不一致的数据
    collate_fn = fast_collate_fn
    
    # 使用分布式采样器（如果是分布式模式）
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # 确保num_workers有有效值
    num_workers = args.num_workers if args.num_workers is not None else 4
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else 2
    
    # 优化数据加载器参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 使用DistributedSampler时不能同时使用shuffle
        num_workers=num_workers,  # 根据系统自动优化
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
        persistent_workers=True if num_workers > 0 else False,  # 保持工作进程活跃
        pin_memory=args.pin_memory,  # 使用固定内存，加速GPU传输
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,  # 根据系统自动优化
    )
    
    # 评估批次大小
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),  # 验证时可以使用较少的工作进程
        drop_last=True,
        collate_fn=collate_fn,
        sampler=val_sampler,
        persistent_workers=True if num_workers > 0 else False,  # 保持工作进程活跃
        pin_memory=args.pin_memory,  # 使用固定内存，加速GPU传输
        prefetch_factor=prefetch_factor if num_workers > 0 else None,  # 预取数据
    )
    
    # 加载均值和标准差
    if args.use_cache_mean_std:
        # 缓存均值和标准差到GPU
        mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
        std_pose = np.load('mean_std/seamless_smplh_std.npy')
        
        # 预处理均值和标准差
        if mean_pose.shape[0] == 330:
            mean_pose = mean_pose[:156]
            std_pose = std_pose[:156]
        
        # 转换为tensor并移到GPU
        mean_pose_tensor = torch.from_numpy(mean_pose).float().cuda()
        std_pose_tensor = torch.from_numpy(std_pose).float().cuda()
    else:
        mean_pose = np.load('mean_std/seamless_smplh_mean.npy')
        std_pose = np.load('mean_std/seamless_smplh_std.npy')
    
    # 创建模型
    # 使用传入的参数，如果没有传入则使用默认值
    if not hasattr(args, 'code_dim') or args.code_dim == 128:
        args.code_dim = 128  # 保持与参数解析器一致
    if not hasattr(args, 'width') or args.width == 512:
        args.width = 512  # 默认值
    if not hasattr(args, 'depth') or args.depth == 3:
        args.depth = 3  # 默认值
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99
    
    # 输入维度是156（完整姿态）
    input_dim = 156
    
    model = RVQVAE(
        args,
        input_dim,
        args.nb_code,
        args.code_dim,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    )
    
    # 设置设备
    device = torch.device(f"cuda:{args.local_rank}" if is_distributed else "cuda")
    model.cuda(device)
    
    # 动态内存管理
    dynamic_memory_management(model, device, args, logger)
    
    # 自动调整批次大小
    if args.auto_tune_batch_size and is_main_process:
        optimal_batch_size = auto_tune_batch_size(
            model, device, args.batch_size, args.max_batch_size, train_loader, world_size, rank, args, logger
        )
        args.batch_size = optimal_batch_size
        
        # 更新数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),  # 如果没有采样器，则shuffle
            num_workers=args.num_workers if args.num_workers is not None else 0,  # 根据系统自动优化
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            persistent_workers=True if args.num_workers is not None and args.num_workers > 0 else False,  # 保持工作进程活跃
            pin_memory=args.pin_memory,  # 使用固定内存，加速GPU传输
            prefetch_factor=args.prefetch_factor if args.num_workers is not None and args.num_workers > 0 else None,  # 根据系统自动优化
        )
        
        if is_main_process:
            print(f"自动调整批次大小为: {args.batch_size}")
    
    # 分析内存使用情况
    if hasattr(args, 'profile_memory') and args.profile_memory and is_main_process:
        profile_memory_usage(model, train_loader, device, logger)
    
    # 编译模型（如果支持）
    if args.compile_model and hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        if is_main_process:
            print("编译模型以优化性能...")
        try:
            model = torch.compile(model)
            if is_main_process:
                print("模型编译成功")
        except Exception as e:
            if is_main_process:
                print(f"模型编译失败: {e}")
    
    # 使用分布式数据并行（如果是分布式模式）
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank], 
                   find_unused_parameters=args.find_unused_parameters, 
                   output_device=args.local_rank)
    
    model.train()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    
    # 创建损失函数
    Loss = ReConsLoss(args.recons_loss)
    
    # 设置混合精度训练
    use_amp = args.mixed_precision
    scaler = GradScaler() if use_amp else None
    
    # 设置梯度检查点
    if args.gradient_checkpointing:
        # 如果模型支持梯度检查点，启用它
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
        else:
            if is_main_process:
                print("警告: 模型不支持梯度检查点")
    
    # 训练循环
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    
    if is_main_process:
        print("开始训练循环...")
        print(f"开始训练，总共 {args.total_iter} 次迭代")
        print(f"使用 {world_size} 个GPU，每个GPU的batch_size为 {args.batch_size}")
        print(f"总有效batch_size为 {args.batch_size * world_size}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"混合精度训练: {use_amp}")
        print(f"数据加载器工作进程数: {args.num_workers}")
        print(f"数据预取因子: {args.prefetch_factor}")
        
        # 创建进度条
        pbar = tqdm(range(1, args.total_iter + 1), desc="训练进度", 
                    bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}]',
                    mininterval=1.0,  # 最小更新间隔为1秒
                    miniters=1)  # 最小迭代次数为1
    else:
        pbar = range(1, args.total_iter + 1)
    
    # 记录最佳损失
    best_recons_loss = float('inf')
    
    if is_main_process:
        print("即将进入训练循环...")
    
    # 创建数据迭代器
    train_iter = iter(train_loader)
    
    # 预热GPU
    if is_main_process:
        print("预热GPU...")
    with torch.no_grad():
        for _ in range(3):  # 预热3个批次
            try:
                batch = next(train_iter)
                gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)
                with autocast(enabled=use_amp):
                    _ = model(gt_motion)
            except StopIteration:
                train_iter = iter(train_loader)
                break
            except Exception:
                break
    
    # 重置数据迭代器
    train_iter = iter(train_loader)
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 记录GPU内存使用情况
    if is_main_process:
        initial_gpu_memory = monitor_gpu_memory(device.index, logger)
        logger.info(f"初始GPU内存使用: {initial_gpu_memory}")
    
    # 自适应批次大小变量
    current_batch_size = args.batch_size
    batch_adjust_counter = 0
    
    for nb_iter in pbar:
        # 设置分布式采样器的epoch（如果是分布式模式）
        if is_distributed:
            train_sampler.set_epoch(nb_iter)
        
        # 记录迭代开始时间（用于性能监控）
        iter_start_time = time.time()

        try:
            # 获取数据批次
            try:
                batch = next(train_iter)
            except StopIteration:
                # 数据迭代器耗尽，重新创建
                train_iter = iter(train_loader)
                batch = next(train_iter)
                
            gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)  # 使用non_blocking加速传输
            batch_mask = batch['mask'].to(device, non_blocking=args.non_blocking_transfer)  # 使用non_blocking加速传输

            # 标准化 - 使用缓存的均值和标准差
            if args.use_cache_mean_std:
                # 扩展维度以匹配(batch_size, seq_len, dim)
                mean_pose_expanded = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                std_pose_expanded = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
                
                gt_motion = (gt_motion - mean_pose_expanded) / std_pose_expanded
            else:
                # 均值和标准差是针对完整姿态的(330,)，但我们的数据是(batch_size, seq_len, 156)
                # 我们需要只使用对应部分的均值和标准差
                if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                    # 如果均值是完整姿态(330)，我们只使用前156个值
                    mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
                    std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
                else:
                    mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
                    std_pose_tensor = torch.from_numpy(std_pose).to(device)

                # 扩展维度以匹配(batch_size, seq_len, dim)
                mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)

                gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor

            # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
            with autocast(enabled=use_amp):
                pred_motion, loss_commit, perplexity = model(gt_motion).values()
                loss_motion = Loss.my_forward(pred_motion, gt_motion, list(range(gt_motion.shape[2])))  # 使用实际维度
                loss_vel = 0  # 暂时不计算速度损失
                loss = (loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel) / args.gradient_accumulation_steps

            # 反向传播和优化
            if use_amp:
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (nb_iter % args.gradient_accumulation_steps == 0) or (nb_iter == args.total_iter):
                    # 梯度裁剪
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                
                # 梯度累积
                if (nb_iter % args.gradient_accumulation_steps == 0) or (nb_iter == args.total_iter):
                    # 梯度裁剪
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            avg_recons += loss_motion.item()
            avg_perplexity += perplexity.item()
            avg_commit += loss_commit.item()

            # 更新最佳损失
            if loss_motion.item() < best_recons_loss:
                best_recons_loss = loss_motion.item()

        except Exception as e:
            if is_main_process:
                print(f"获取数据批次时出错: {e}")
                import traceback
                traceback.print_exc()
            break
        
        # 记录迭代结束时间并更新性能监控
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        
        # 更新性能监控
        if is_main_process and perf_monitor:
            perf_monitor = update_performance_monitor(
                perf_monitor, nb_iter, iter_time, device, loss_motion.item(), 
                current_batch_size, world_size, logger
            )
            
            # 自动性能调优
            if args.auto_tune_performance and nb_iter % 1000 == 0:
                perf_monitor, args = auto_tune_performance(
                    perf_monitor, model, optimizer, args, logger
                )
        
        # 自适应批次大小调整
        if args.adaptive_batching and is_main_process and nb_iter % args.batch_adjust_interval == 0:
            # 监控GPU利用率
            gpu_util_info = monitor_gpu_utilization(device, logger)
            if gpu_util_info:
                gpu_util = gpu_util_info['memory_util']  # 使用内存利用率作为代理
                
                # 调整批次大小
                new_batch_size, action = adaptive_batch_sizing(
                    current_batch_size, gpu_util, args.target_gpu_util
                )
                
                if action != "maintain" and new_batch_size != current_batch_size:
                    if logger:
                        logger.info(f"自适应调整批次大小: {current_batch_size} -> {new_batch_size} ({action})")
                    
                    # 更新数据加载器
                    current_batch_size = new_batch_size
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=current_batch_size,
                        shuffle=(train_sampler is None),  # 如果没有采样器，则shuffle
                        num_workers=args.num_workers,  # 根据系统自动优化
                        drop_last=True,
                        collate_fn=collate_fn,
                        sampler=train_sampler,
                        persistent_workers=True if args.num_workers > 0 else False,  # 保持工作进程活跃
                        pin_memory=args.pin_memory,  # 使用固定内存，加速GPU传输
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,  # 根据系统自动优化
                    )
                    
                    # 重置数据迭代器
                    train_iter = iter(train_loader)
        
        # 监控分布式训练状态
        if nb_iter % 1000 == 0 and is_main_process:
            dist_status = monitor_distributed_training(is_distributed, rank, world_size, logger)
            if writer:
                writer.add_scalar('./System/WorldSize', dist_status['world_size'], nb_iter)
        
        # 更新进度条显示
        if is_main_process:
            pbar.set_postfix({
                'Recons': f'{loss_motion.item():.5f}',
                'PPL': f'{perplexity.item():.2f}',
                'Commit': f'{loss_commit.item():.5f}',
                'Best': f'{best_recons_loss:.5f}',
                'Batch': f'{current_batch_size}'
            })
            # 不需要手动调用pbar.update(1)，因为tqdm会自动更新
        
        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            
            if is_main_process:
                writer.add_scalar('./Train/L1', avg_recons, nb_iter)
                writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
                writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
                
                # 计算并记录训练速度
                elapsed_time = time.time() - training_start_time
                iters_per_sec = nb_iter / elapsed_time
                writer.add_scalar('./Train/ItersPerSec', iters_per_sec, nb_iter)
                
                # 记录GPU内存使用情况
                current_gpu_memory = monitor_gpu_memory(device.index, logger)
                if current_gpu_memory:
                    writer.add_scalar('./System/GPUMemoryUtil', current_gpu_memory['utilization'], nb_iter)
                
                # 记录批次大小
                writer.add_scalar('./Train/BatchSize', current_batch_size, nb_iter)
                
                logger.info(f"Train. Iter {nb_iter}: \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons. {avg_recons:.5f} \t Speed: {iters_per_sec:.2f} iters/s \t Batch: {current_batch_size}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        
        if nb_iter % args.eval_iter == 0:
            if is_main_process:
                torch.save({'net': model.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
            
            # 使用验证集进行评估
            model.eval()
            total_l2 = 0
            num_batches = 0
            
            # 创建评估进度条
            if is_main_process:
                eval_pbar = tqdm(val_loader, desc=f"验证集评估 (迭代 {nb_iter})", leave=False)
            else:
                eval_pbar = val_loader
            
            eval_start_time = time.time()
            
            with torch.no_grad():
                for batch in eval_pbar:
                    gt_motion = batch['pose'].to(device, non_blocking=args.non_blocking_transfer)  # 使用non_blocking加速传输
                    batch_mask = batch['mask'].to(device, non_blocking=args.non_blocking_transfer)  # 使用non_blocking加速传输
                    
                    # 标准化 - 使用缓存的均值和标准差
                    if args.use_cache_mean_std:
                        # 扩展维度以匹配(batch_size, seq_len, dim)
                        mean_pose_expanded = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                        std_pose_expanded = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
                        
                        gt_motion = (gt_motion - mean_pose_expanded) / std_pose_expanded
                    else:
                        # 标准化
                        if mean_pose.shape[0] == 330 and gt_motion.shape[2] == 156:
                            # 如果均值是完整姿态(330)，我们只使用前156个值
                            mean_pose_tensor = torch.from_numpy(mean_pose[:156]).to(device)
                            std_pose_tensor = torch.from_numpy(std_pose[:156]).to(device)
                        else:
                            mean_pose_tensor = torch.from_numpy(mean_pose).to(device)
                            std_pose_tensor = torch.from_numpy(std_pose).to(device)
                        
                        # 扩展维度以匹配(batch_size, seq_len, dim)
                        mean_pose_tensor = mean_pose_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 156)
                        std_pose_tensor = std_pose_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, 156)
                        
                        gt_motion = (gt_motion - mean_pose_tensor) / std_pose_tensor
                    
                    # 模型期望输入为(bs, seq_len, dim_pose)，会自己进行转置
                    with autocast(enabled=use_amp):
                        pred_motion, loss_commit, perplexity = model(gt_motion).values()
                    
                    # 模型输出已经是(bs, seq_len, dim_pose)格式，不需要转置
                    
                    # 计算L2距离
                    diff = pred_motion - gt_motion
                    l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
                    total_l2 += l2_batch
                    num_batches += 1
                    
                    # 更新评估进度条
                    if is_main_process:
                        eval_pbar.set_postfix({'L2': f'{l2_batch:.5f}'})
            
            avg_l2 = total_l2 / num_batches
            eval_time = time.time() - eval_start_time
            
            if is_main_process:
                logger.info(f"Validation. Iter {nb_iter}: \t L2 Distance: {avg_l2:.5f} \t Eval Time: {eval_time:.2f}s")
                writer.add_scalar('./Val/L2', avg_l2, nb_iter)
                writer.add_scalar('./Val/EvalTime', eval_time, nb_iter)
            
            model.train()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
    
    if is_main_process:
        total_training_time = time.time() - training_start_time
        print(f"训练完成! 总训练时间: {total_training_time:.2f}s")
        print(f"平均训练速度: {args.total_iter / total_training_time:.2f} iters/s")
        
        # 记录最终GPU内存使用情况
        final_gpu_memory = monitor_gpu_memory(device.index, logger)
        logger.info(f"最终GPU内存使用: {final_gpu_memory}")
        
        # 保存性能报告
        if perf_monitor:
            save_performance_report(perf_monitor, args.out_dir, logger)
            plot_performance_curves(perf_monitor, args.out_dir, logger)
    
    # 清理分布式训练环境（如果是分布式模式）
    if is_distributed:
        dist.destroy_process_group()


def get_args_parser():
    parser = argparse.ArgumentParser(description='RVQ-VAE training for Seamless Interaction - Optimized Version',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据集参数
    data_group = parser.add_argument_group('数据集参数')
    data_group.add_argument('--dataname', type=str, default='seamless_interaction', help='数据集名称')
    data_group.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction', help='数据集路径')
    data_group.add_argument('--window-size', type=int, default=64, help='窗口大小(帧数)')
    data_group.add_argument('--window-stride', type=int, default=20, help='窗口步长(帧数)')
    data_group.add_argument('--multi_length_training', type=float, nargs='+', default=None, help='多长度训练比例列表')
    data_group.add_argument('--body-part', type=str, default='upper', choices=['whole', 'upper', 'lower', 'hands', 'face'], help='训练的身体部位')
    data_group.add_argument('--max_samples', type=int, default=None, help='最大样本数，设置为None使用整个数据集')
    
    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument("--code-dim", type=int, default=512, help="嵌入维度")
    model_group.add_argument("--nb-code", type=int, default=1024, help="嵌入数量")
    model_group.add_argument("--down-t", type=int, default=2, help="下采样率")
    model_group.add_argument("--stride-t", type=int, default=2, help="步长大小")
    model_group.add_argument("--width", type=int, default=1024, help="网络宽度")
    model_group.add_argument("--depth", type=int, default=6, help="网络深度")
    model_group.add_argument("--dilation-growth-rate", type=int, default=3, help="膨胀增长率")
    model_group.add_argument("--output-emb-width", type=int, default=1024, help="输出嵌入宽度")
    model_group.add_argument('--vq-act', type=str, default='gelu', choices=['relu', 'silu', 'gelu'], help='激活函数')
    model_group.add_argument('--vq-norm', type=str, default='layer_norm', choices=['batch_norm', 'layer_norm', 'none'], help='归一化方法')
    model_group.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'], help="量化器类型")
    model_group.add_argument("--num_quantizers", type=int, default=12, help="量化器数量")
    model_group.add_argument("--shared_codebook", type=bool, default=False, help="是否使用共享码本")
    model_group.add_argument("--quantize_dropout_prob", type=float, default=0.2, help="量化丢弃概率")
    model_group.add_argument('--beta', type=float, default=1.0, help='标准VQ中的承诺损失')
    model_group.add_argument("--mu", type=float, default=0.99, help="更新码本的指数移动平均")
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--total-iter', default=300000, type=int, help='总迭代次数')
    train_group.add_argument('--warm-up-iter', default=1000, type=int, help='预热迭代次数')
    train_group.add_argument('--lr', default=1e-4, type=float, help='最大学习率')
    train_group.add_argument('--lr-scheduler', default=[200000, 250000], nargs="+", type=int, help="学习率调度(迭代)")
    train_group.add_argument('--gamma', default=0.1, type=float, help="学习率衰减")
    train_group.add_argument('--weight-decay', default=0.01, type=float, help='权重衰减')
    train_group.add_argument("--commit", type=float, default=0.02, help="承诺损失的超参数")
    train_group.add_argument('--loss-vel', type=float, default=0.1, help='速度损失的超参数')
    train_group.add_argument('--recons-loss', type=str, default='l1_smooth', help='重建损失')
    train_group.add_argument('--batch-size', default=1024, type=int, help='批次大小')
    train_group.add_argument('--eval-batch-size', type=int, default=1024, help='评估批次大小')
    train_group.add_argument('--print-iter', default=500, type=int, help='打印频率')
    train_group.add_argument('--eval-iter', default=5000, type=int, help='评估频率')
    train_group.add_argument('--seed', default=123, type=int, help='初始化训练的种子')
    train_group.add_argument('--mode', type=str, default='train', help='训练或评估')
    
    # 数据加载优化参数
    data_opt_group = parser.add_argument_group('数据加载优化参数')
    data_opt_group.add_argument('--num-workers', type=int, default=None, help='数据加载器工作进程数(默认自动优化)')
    data_opt_group.add_argument('--prefetch-factor', type=int, default=None, help='预取因子(默认自动优化)')
    data_opt_group.add_argument('--use-fast-collate', action='store_true', default=True, help='使用快速批处理函数')
    data_opt_group.add_argument('--pin-memory', action='store_true', default=True, help='使用固定内存')
    data_opt_group.add_argument('--non-blocking-transfer', action='store_true', default=True, help='使用非阻塞传输')
    data_opt_group.add_argument('--persistent-workers', action='store_true', default=True, help='保持工作进程活跃')
    data_opt_group.add_argument('--cache-path', type=str, default=None, help='缓存文件路径')
    data_opt_group.add_argument('--val-cache-path', type=str, default=None, help='验证集缓存文件路径')
    data_opt_group.add_argument('--cache-train', type=str, default=None, help='训练集缓存文件路径(别名)')
    data_opt_group.add_argument('--cache-val', type=str, default=None, help='验证集缓存文件路径(别名)')
    data_opt_group.add_argument('--use-cache', action='store_true', default=False, help='是否使用缓存')
    data_opt_group.add_argument('--use-cache-mean-std', action='store_true', default=False, help='是否使用缓存的均值和标准差')
    
    # GPU内存和计算优化参数
    gpu_opt_group = parser.add_argument_group('GPU内存和计算优化参数')
    gpu_opt_group.add_argument('--max-batch-size', type=int, default=2048, help='最大批次大小')
    gpu_opt_group.add_argument('--auto-tune-batch-size', action='store_true', default=True, help='自动调整批次大小')
    gpu_opt_group.add_argument('--adaptive-batching', action='store_true', default=True, help='自适应批次大小')
    gpu_opt_group.add_argument('--batch-adjust-interval', type=int, default=500, help='批次大小调整间隔(迭代)')
    gpu_opt_group.add_argument('--target-gpu-util', type=float, default=85.0, help='目标GPU利用率(%)')
    gpu_opt_group.add_argument('--gradient-accumulation-steps', type=int, default=1, help='梯度累积步数')
    gpu_opt_group.add_argument('--gradient-checkpointing', action='store_true', default=False, help='启用梯度检查点')
    gpu_opt_group.add_argument('--use-amp', action='store_true', default=True, help='使用自动混合精度')
    gpu_opt_group.add_argument('--memory-fraction', type=float, default=0.9, help='GPU内存分配比例')
    
    # 分布式训练参数
    dist_group = parser.add_argument_group('分布式训练参数')
    dist_group.add_argument('--local-rank', type=int, default=0, help='本地进程排名')
    dist_group.add_argument('--find-unused-parameters', action='store_true', default=False, help='查找未使用参数')
    dist_group.add_argument('--benchmark-distributed', action='store_true', default=False, help='运行分布式基准测试')
    
    # 性能监控参数
    perf_group = parser.add_argument_group('性能监控参数')
    perf_group.add_argument('--auto-save-interval', type=int, default=3600, help='自动保存间隔(秒)')
    perf_group.add_argument('--performance-log-interval', type=int, default=100, help='性能日志记录间隔(迭代)')
    perf_group.add_argument('--auto-tune-performance', action='store_true', default=False, help='启用自动性能调优')
    perf_group.add_argument('--profile-memory', action='store_true', default=False, help='启用内存分析')
    perf_group.add_argument('--compile-model', action='store_true', default=False, help='编译模型以优化性能')
    
    # 恢复训练参数
    resume_group = parser.add_argument_group('恢复训练参数')
    resume_group.add_argument("--resume-pth", type=str, default=None, help='VQ恢复路径')
    
    # 输出目录参数
    output_group = parser.add_argument_group('输出目录参数')
    output_group.add_argument('--out-dir', type=str, default='outputs/rvqvae_seamless_optimized', help='输出目录')
    output_group.add_argument('--exp-name', type=str, default='RVQVAE_Seamless_Optimized', help='实验名称')
    
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)