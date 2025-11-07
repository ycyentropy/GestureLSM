import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

from models.vq.model import RVQVAE

def calculate_memory_usage(batch_size, seq_len=64, pose_dim=156):
    """计算不同batch_size下的显存使用情况"""
    
    # 创建模型参数
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
        pose_dim,
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
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型参数显存占用 (假设float32，每个参数4字节)
    model_memory = total_params * 4 / (1024 ** 2)  # MB
    
    # 创建测试数据 (batch_size, seq_len, pose_dim)
    test_motion = torch.randn(batch_size, seq_len, pose_dim).cuda()  # 恢复原始维度顺序
    test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
    
    # 计算输入数据显存占用
    input_memory = batch_size * seq_len * pose_dim * 4 / (1024 ** 2)  # MB
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        # 模型期望的输入是 (bs, seq_len, dim_pose) 格式
        input_motion = test_motion  # 已经是正确的维度顺序
        
        # 前向传播
        pred_motion, loss_commit, perplexity = model(input_motion).values()
        
        # 计算中间激活的显存占用 (估算)
        # 假设中间激活的大小与输入数据相当
        activation_memory = input_memory * 2  # 估算为输入数据的2倍
    
    # 计算梯度显存占用 (训练时)
    gradient_memory = model_memory  # 梯度与参数大小相同
    
    # 计算优化器状态显存占用 (Adam优化器，需要存储动量和方差)
    optimizer_memory = model_memory * 2  # Adam需要存储动量和方差
    
    # 总显存占用
    total_memory = model_memory + input_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'batch_size': batch_size,
        'model_params': total_params,
        'trainable_params': trainable_params,
        'model_memory_mb': model_memory,
        'input_memory_mb': input_memory,
        'activation_memory_mb': activation_memory,
        'gradient_memory_mb': gradient_memory,
        'optimizer_memory_mb': optimizer_memory,
        'total_memory_mb': total_memory,
        'total_memory_gb': total_memory / 1024
    }

def main():
    print("计算不同batch_size下的显存使用情况...\n")
    
    # 获取GPU信息
    gpu_info = torch.cuda.get_device_properties(0)
    total_gpu_memory = gpu_info.total_memory / (1024 ** 3)  # GB
    print(f"GPU总显存: {total_gpu_memory:.2f} GB")
    
    # 计算不同batch_size下的显存使用情况
    batch_sizes = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = calculate_memory_usage(batch_size)
            results.append(result)
            
            print(f"Batch Size: {result['batch_size']}")
            print(f"  模型参数量: {result['model_params']:,}")
            print(f"  可训练参数量: {result['trainable_params']:,}")
            print(f"  模型参数显存: {result['model_memory_mb']:.2f} MB")
            print(f"  输入数据显存: {result['input_memory_mb']:.2f} MB")
            print(f"  中间激活显存: {result['activation_memory_mb']:.2f} MB")
            print(f"  梯度显存: {result['gradient_memory_mb']:.2f} MB")
            print(f"  优化器显存: {result['optimizer_memory_mb']:.2f} MB")
            print(f"  总显存占用: {result['total_memory_gb']:.2f} GB")
            print(f"  显存利用率: {result['total_memory_gb'] / total_gpu_memory * 100:.1f}%")
            print()
            
            # 如果显存占用超过GPU总显存的90%，停止增加batch_size
            if result['total_memory_gb'] > total_gpu_memory * 0.9:
                print(f"警告: Batch Size {batch_size} 的显存占用接近GPU上限!")
                break
                
        except torch.cuda.OutOfMemoryError:
            print(f"Batch Size {batch_size} 导致显存不足!")
            break
    
    # 找出最接近GPU显存上限的batch_size
    if results:
        best_result = max(results, key=lambda x: x['total_memory_gb'])
        print("\n推荐配置:")
        print(f"最佳Batch Size: {best_result['batch_size']}")
        print(f"显存占用: {best_result['total_memory_gb']:.2f} GB ({best_result['total_memory_gb'] / total_gpu_memory * 100:.1f}%)")
        
        # 计算可以使用的数据量
        print("\n数据量建议:")
        print(f"当前训练使用10个样本，分割为2697个时间窗口")
        print(f"建议增加max_samples参数以使用更多数据")
        print(f"每个样本约产生270个时间窗口")
        print(f"使用100个样本将产生约27000个时间窗口")
        print(f"使用500个样本将产生约135000个时间窗口")
        print(f"使用全部9009个样本将产生约2432430个时间窗口")

if __name__ == "__main__":
    main()