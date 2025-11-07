import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

from models.vq.model import RVQVAE

def test_memory_usage_single_gpu(batch_size, gpu_id=1):
    """测试在指定GPU上的显存使用情况"""
    
    # 设置使用的GPU
    torch.cuda.set_device(gpu_id)
    
    # 创建与训练脚本相同的参数
    args = argparse.Namespace()
    args.num_quantizers = 6
    args.shared_codebook = False
    args.quantize_dropout_prob = 0.2
    args.nb_code = 512
    args.code_dim = 256
    args.down_t = 2
    args.stride_t = 2
    args.width = 512
    args.depth = 3
    args.dilation_growth_rate = 3
    args.vq_act = 'relu'
    args.vq_norm = None
    args.mu = 0.99
    
    # 创建模型
    model = RVQVAE(
        args,
        156,  # pose_dim
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
    
    model.cuda()
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    
    # 创建测试数据 (与训练脚本相同)
    seq_len = 64  # 窗口大小
    pose_dim = 156  # 全身姿态维度
    
    # 模拟数据加载器
    test_motion = torch.randn(batch_size, seq_len, pose_dim).cuda()
    test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
    
    # 记录初始显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    
    # 前向传播
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    output = model(test_motion)
    pred_motion = output['rec_pose']
    loss_commit = output['commit_loss']
    
    # 计算损失 (与训练脚本相同)
    loss_motion = nn.L1Loss()(pred_motion, test_motion)
    loss = loss_motion + 0.02 * loss_commit
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 记录峰值显存
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return {
        'batch_size': batch_size,
        'gpu_id': gpu_id,
        'peak_memory_mb': peak_memory,
        'used_memory_mb': peak_memory - initial_memory,
        'used_memory_gb': (peak_memory - initial_memory) / 1024
    }

def main():
    print("测试单个GPU环境下的显存使用情况...\n")
    
    # 获取GPU信息
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_info = torch.cuda.get_device_properties(i)
        total_gpu_memory = gpu_info.total_memory / (1024 ** 3)  # GB
        print(f"GPU {i}: {gpu_info.name}, 总显存: {total_gpu_memory:.2f} GB")
    
    # 选择一个空闲的GPU进行测试 (GPU 1-7都是空闲的)
    test_gpu_id = 1
    gpu_info = torch.cuda.get_device_properties(test_gpu_id)
    total_gpu_memory = gpu_info.total_memory / (1024 ** 3)  # GB
    print(f"\n使用GPU {test_gpu_id}进行测试，总显存: {total_gpu_memory:.2f} GB")
    
    # 测试不同batch_size下的显存使用情况
    batch_sizes = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
    results = []
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            result = test_memory_usage_single_gpu(batch_size, test_gpu_id)
            results.append(result)
            
            print(f"Batch Size: {result['batch_size']}")
            print(f"  峰值显存占用: {result['peak_memory_mb']:.2f} MB")
            print(f"  实际使用显存: {result['used_memory_mb']:.2f} MB ({result['used_memory_gb']:.2f} GB)")
            print(f"  显存利用率: {result['used_memory_gb'] / total_gpu_memory * 100:.1f}%")
            print()
            
            # 如果显存占用超过GPU总显存的90%，停止增加batch_size
            if result['used_memory_gb'] > total_gpu_memory * 0.9:
                print(f"警告: Batch Size {batch_size} 的显存占用接近GPU上限!")
                break
                
        except torch.cuda.OutOfMemoryError:
            print(f"Batch Size {batch_size} 导致显存不足!")
            break
    
    # 找出最接近GPU显存上限的batch_size
    if results:
        best_result = max(results, key=lambda x: x['used_memory_gb'])
        print("\n推荐配置:")
        print(f"最佳Batch Size: {best_result['batch_size']}")
        print(f"显存占用: {best_result['used_memory_gb']:.2f} GB ({best_result['used_memory_gb'] / total_gpu_memory * 100:.1f}%)")
        
        # 计算数据量建议
        print("\n数据量建议:")
        print(f"当前训练使用10个样本，分割为2697个时间窗口")
        print(f"建议增加max_samples参数以使用更多数据")
        print(f"每个样本约产生270个时间窗口")
        
        # 计算可以使用的数据量
        max_samples_options = [100, 500, 1000, 5000, 9009]
        
        print(f"\n不同数据量下的时间窗口数:")
        for max_samples in max_samples_options:
            print(f"  使用{max_samples}个样本: ~{max_samples * 270}个时间窗口")
        
        # 计算训练命令
        print("\n推荐的训练命令:")
        print(f"python rvq_seamless_train.py --mode train --batch-size {best_result['batch_size']} --max-samples 1000")
        print(f"python rvq_seamless_train.py --mode train --batch-size {best_result['batch_size']} --max-samples 5000")
        print(f"python rvq_seamless_train.py --mode train --batch-size {best_result['batch_size']} --max-samples 9009")
        
        # 计算多GPU训练的潜在收益
        print("\n多GPU训练建议:")
        print(f"当前有{num_gpus}个GPU可用，可以使用DataParallel或DistributedDataParallel进行多GPU训练")
        print(f"使用{num_gpus}个GPU，每个GPU使用batch_size={best_result['batch_size']}")
        print(f"总有效batch_size: {best_result['batch_size'] * num_gpus}")
        print(f"多GPU训练命令示例:")
        print(f"python -m torch.distributed.launch --nproc_per_node={num_gpus} rvq_seamless_train.py --mode train --batch-size {best_result['batch_size']} --max-samples 5000")
        
        # 计算训练时间估算
        print("\n训练时间估算:")
        for max_samples in [1000, 5000, 9009]:
            num_windows = max_samples * 270
            num_batches = num_windows // best_result['batch_size']
            # 假设每个批次需要0.5秒
            estimated_time_hours = num_batches * 0.5 / 3600
            print(f"  使用{max_samples}个样本: ~{num_windows}个时间窗口, {num_batches}个批次, 预计训练时间: {estimated_time_hours:.2f}小时")

if __name__ == "__main__":
    main()