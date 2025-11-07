import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入ResidualVQ相关模块
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')
from models.vq.residual_vq import ResidualVQ
from models.vq.quantizer import QuantizeEMAReset

class SimpleArgs:
    """简化参数类，用于创建ResidualVQ实例"""
    def __init__(self):
        self.mu = 0.99  # EMA衰减率
        self.num_quantizers = 4  # 量化层数
        self.shared_codebook = False  # 不共享码本
        self.quantize_dropout_prob = 0.5  # 量化丢弃概率
        self.quantize_dropout_cutoff_index = 0

def create_residual_vq():
    """创建ResidualVQ实例"""
    args = SimpleArgs()
    residual_vq = ResidualVQ(
        num_quantizers=args.num_quantizers,
        shared_codebook=args.shared_codebook,
        quantize_dropout_prob=args.quantize_dropout_prob,
        quantize_dropout_cutoff_index=args.quantize_dropout_cutoff_index,
        nb_code=16,  # 码本大小（减小以便可视化）
        code_dim=8,  # 码本维度（减小以便可视化）
        args=args
    )
    return residual_vq

def visualize_residual_vq_process():
    """可视化ResidualVQ的量化过程"""
    print("\n" + "=" * 80)
    print("ResidualVQ量化过程可视化")
    print("=" * 80)
    
    # 创建ResidualVQ实例
    residual_vq = create_residual_vq()
    
    # 创建示例输入
    batch_size, seq_len, code_dim = 1, 10, 8
    x = torch.randn(batch_size, code_dim, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 逐步量化过程
    residual = x.clone()
    quantized_out = torch.zeros_like(x)
    
    plt.figure(figsize=(15, 10))
    
    # 绘制原始输入
    plt.subplot(5, 1, 1)
    plt.plot(x[0, 0].detach().numpy(), 'b-', linewidth=2, label='原始输入')
    plt.title('原始输入 (第1个通道)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    all_quantized = []
    all_residuals = []
    
    # 逐层量化
    for i, layer in enumerate(residual_vq.layers):
        # 量化当前残差
        quantized, indices, loss, perplexity = layer(residual, return_idx=True, temperature=1.0)
        
        # 更新残差
        residual = residual - quantized.detach()
        
        # 累积量化结果
        quantized_out = quantized_out + quantized
        
        # 保存结果
        all_quantized.append(quantized[0, 0].detach().numpy())
        all_residuals.append(residual[0, 0].detach().numpy())
        
        # 绘制当前层的量化结果
        plt.subplot(5, 1, i+2)
        plt.plot(quantized[0, 0].detach().numpy(), 'r-', linewidth=2, label=f'第{i+1}层量化')
        plt.plot(residual[0, 0].detach().numpy(), 'g--', linewidth=1, alpha=0.7, label=f'第{i+1}层残差')
        plt.title(f'第{i+1}层量化结果和残差')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/residual_vq_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制累积量化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x[0, 0].detach().numpy(), 'b-', linewidth=2, label='原始输入')
    for i, quantized in enumerate(all_quantized):
        plt.plot(quantized, '--', alpha=0.7, label=f'第{i+1}层量化')
    plt.title('各层量化结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(x[0, 0].detach().numpy(), 'b-', linewidth=2, label='原始输入')
    cumulative_quantized = np.cumsum(all_quantized, axis=0)
    for i in range(len(cumulative_quantized)):
        plt.plot(cumulative_quantized[i], '--', alpha=0.7, label=f'累积到第{i+1}层')
    plt.title('累积量化结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/residual_vq_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("量化过程可视化已保存到:")
    print("- /home/embodied/yangchenyu/GestureLSM/visualizations/residual_vq_process.png")
    print("- /home/embodied/yangchenyu/GestureLSM/visualizations/residual_vq_cumulative.png")

def visualize_codebook_usage():
    """可视化码本使用情况"""
    print("\n" + "=" * 80)
    print("码本使用情况可视化")
    print("=" * 80)
    
    # 创建ResidualVQ实例
    residual_vq = create_residual_vq()
    
    # 创建示例输入并多次量化以更新码本
    for _ in range(20):
        x = torch.randn(4, 8, 20)  # 批次大小=4, 特征维度=8, 序列长度=20
        residual_vq(x, sample_codebook_temp=1.0)  # 前向传播，更新码本
    
    # 获取各层码本使用情况
    codebook_usage = []
    for i, layer in enumerate(residual_vq.layers):
        if hasattr(layer, 'code_count'):
            usage = (layer.code_count > 0).float().cpu().numpy()
            codebook_usage.append(usage)
            print(f"第{i+1}层码本利用率: {usage.mean():.2%}")
    
    # 可视化码本使用情况
    plt.figure(figsize=(12, 8))
    plt.imshow(codebook_usage, cmap='viridis', aspect='auto')
    plt.xlabel('码本索引')
    plt.ylabel('量化层')
    plt.title('各层码本使用情况')
    plt.colorbar(label='使用状态')
    
    # 添加数值标签
    for i in range(len(codebook_usage)):
        for j in range(len(codebook_usage[i])):
            text = "1" if codebook_usage[i][j] > 0 else "0"
            plt.text(j, i, text, ha='center', va='center', color='white' if codebook_usage[i][j] < 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/codebook_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("码本使用情况可视化已保存到:")
    print("- /home/embodied/yangchenyu/GestureLSM/visualizations/codebook_usage.png")

def visualize_quantization_error():
    """可视化量化误差"""
    print("\n" + "=" * 80)
    print("量化误差分析")
    print("=" * 80)
    
    # 创建ResidualVQ实例
    residual_vq = create_residual_vq()
    
    # 创建示例输入
    x = torch.randn(2, 8, 30)
    
    # 量化
    quantized, indices, commit_loss, perplexity = residual_vq(x, sample_codebook_temp=1.0)
    
    # 计算重建误差
    reconstruction_error = F.mse_loss(x, quantized)
    
    print(f"输入形状: {x.shape}")
    print(f"量化结果形状: {quantized.shape}")
    print(f"索引形状: {indices.shape}")
    print(f"重建误差: {reconstruction_error.item():.6f}")
    print(f"承诺损失: {commit_loss.item():.6f}")
    print(f"困惑度: {perplexity.item():.6f}")
    
    # 可视化原始信号和量化信号
    plt.figure(figsize=(15, 10))
    
    for i in range(min(4, x.shape[1])):  # 只可视化前4个通道
        plt.subplot(4, 1, i+1)
        plt.plot(x[0, i].detach().numpy(), 'b-', linewidth=2, label='原始信号')
        plt.plot(quantized[0, i].detach().numpy(), 'r--', linewidth=2, label='量化信号')
        error = np.abs(x[0, i].detach().numpy() - quantized[0, i].detach().numpy())
        plt.fill_between(range(len(error)), error, alpha=0.3, color='red', label='量化误差')
        plt.title(f'通道{i+1}量化结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/quantization_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("量化误差可视化已保存到:")
    print("- /home/embodied/yangchenyu/GestureLSM/visualizations/quantization_error.png")

def visualize_quantizer_dropout():
    """可视化量化丢弃效果"""
    print("\n" + "=" * 80)
    print("量化丢弃效果可视化")
    print("=" * 80)
    
    # 创建ResidualVQ实例
    residual_vq = create_residual_vq()
    
    # 创建示例输入
    x = torch.randn(2, 8, 30)
    
    # 完整量化
    quantized_full, indices_full, _, _ = residual_vq(x, force_dropout_index=-1, sample_codebook_temp=1.0)
    
    # 部分量化（只使用前2层）
    quantized_partial, indices_partial, _, _ = residual_vq(x, force_dropout_index=2, sample_codebook_temp=1.0)
    
    print(f"完整量化索引形状: {indices_full.shape}")
    print(f"部分量化索引形状: {indices_partial.shape}")
    print(f"完整量化使用的唯一索引值: {torch.unique(indices_full).tolist()}")
    print(f"部分量化使用的唯一索引值: {torch.unique(indices_partial).tolist()}")
    
    # 可视化完整量化和部分量化的差异
    plt.figure(figsize=(15, 10))
    
    channel_idx = 0  # 选择第一个通道进行可视化
    
    plt.subplot(3, 1, 1)
    plt.plot(x[0, channel_idx].detach().numpy(), 'b-', linewidth=2, label='原始信号')
    plt.title('原始信号')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(quantized_full[0, channel_idx].detach().numpy(), 'r-', linewidth=2, label='完整量化')
    plt.title('完整量化结果（使用所有4层）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(quantized_partial[0, channel_idx].detach().numpy(), 'g-', linewidth=2, label='部分量化')
    plt.title('部分量化结果（只使用前2层）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/quantizer_dropout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("量化丢弃效果可视化已保存到:")
    print("- /home/embodied/yangchenyu/GestureLSM/visualizations/quantizer_dropout.png")

def explain_residual_vq_concepts():
    """解释ResidualVQ的核心概念"""
    print("\n" + "=" * 80)
    print("ResidualVQ核心概念解释")
    print("=" * 80)
    
    print("\n1. 基本概念")
    print("-" * 40)
    print("ResidualVQ（残差向量量化）是一种多层向量量化方法，通过逐步量化输入的残差来实现更精细的表示。")
    print("每个量化层量化前一层留下的残差，最终输出是所有层量化结果的总和。")
    
    print("\n2. 量化过程")
    print("-" * 40)
    print("第1层: 量化原始输入 → 量化结果1 + 残差1")
    print("第2层: 量化残差1 → 量化结果2 + 残差2")
    print("第3层: 量化残差2 → 量化结果3 + 残差3")
    print("...")
    print("最终输出: 量化结果1 + 量化结果2 + 量化结果3 + ...")
    
    print("\n3. 码本结构")
    print("-" * 40)
    print("- 每个量化层有独立的码本")
    print("- 码本大小: nb_code (如1024)")
    print("- 码本维度: code_dim (如128)")
    print("- 可选: 所有层共享同一码本")
    
    print("\n4. 量化索引")
    print("-" * 40)
    print("- 每个量化层产生一组索引")
    print("- 最终索引形状: (batch_size, sequence_length, num_quantizers)")
    print("- 每个位置有num_quantizers个索引，对应不同量化层")
    
    print("\n5. EMA更新机制")
    print("-" * 40)
    print("- 使用指数移动平均更新码本")
    print("- 公式: 新码本 = μ * 旧码本 + (1-μ) * 当前统计量")
    print("- μ通常设置为0.99，确保码本稳定更新")
    
    print("\n6. Reset机制")
    print("-" * 40)
    print("- 当码本使用不均匀时，重置未使用的码本")
    print("- 使用当前输入数据复制并添加噪声作为新码本")
    print("- 防止码本退化，提高利用率")
    
    print("\n7. 量化丢弃机制")
    print("-" * 40)
    print("- 训练时随机丢弃部分量化层")
    print("- 提高训练稳定性和模型鲁棒性")
    print("- 使模型对不同量化层数具有适应性")
    
    print("\n8. 承诺损失")
    print("-" * 40)
    print("- 确保编码器输出与码本向量保持一致")
    print("- 公式: MSE(x, quantized) + MSE(x, quantized.detach())")
    print("- 引导编码器输出接近码本向量")
    
    print("\n9. 直通估计器")
    print("-" * 40)
    print("- 实现梯度反向传播")
    print("- 公式: output = input + (quantized - input).detach()")
    print("- 前向传播使用量化结果，反向传播使用原始输入")

def main():
    """主函数"""
    print("ResidualVQ详细解释与可视化")
    print("=" * 80)
    
    # 解释核心概念
    explain_residual_vq_concepts()
    
    # 可视化量化过程
    visualize_residual_vq_process()
    
    # 可视化码本使用情况
    visualize_codebook_usage()
    
    # 可视化量化误差
    visualize_quantization_error()
    
    # 可视化量化丢弃效果
    visualize_quantizer_dropout()
    
    print("\n" + "=" * 80)
    print("ResidualVQ解释与可视化完成！")
    print("=" * 80)
    print("\n生成的可视化图片:")
    print("1. residual_vq_process.png - 量化过程可视化")
    print("2. residual_vq_cumulative.png - 累积量化结果")
    print("3. codebook_usage.png - 码本使用情况")
    print("4. quantization_error.png - 量化误差分析")
    print("5. quantizer_dropout.png - 量化丢弃效果")
    
    print("\n详细解释文档:")
    print("- ResidualVQ_详解.md")

if __name__ == "__main__":
    main()