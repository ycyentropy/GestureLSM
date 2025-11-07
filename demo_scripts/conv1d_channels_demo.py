import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def visualize_conv1d_channels():
    """
    演示Conv1D中多通道数据如何进行卷积
    """
    # 创建一个简单的Conv1d层
    # 输入通道: 3, 输出通道: 2, 卷积核大小: 3
    conv1d = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    
    # 创建输入数据 (batch_size=1, channels=3, length=5)
    # 模拟3个通道的时间序列数据
    input_data = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # 通道1
            [2.0, 3.0, 4.0, 5.0, 6.0],  # 通道2
            [3.0, 4.0, 5.0, 6.0, 7.0],  # 通道3
        ]
    ], dtype=torch.float32)
    
    print("输入数据形状:", input_data.shape)
    print("输入数据:")
    print(input_data)
    
    # 获取卷积核权重
    # 权重形状: (out_channels, in_channels, kernel_size)
    weights = conv1d.weight.data
    print("\n卷积核权重形状:", weights.shape)
    print("每个输出通道的卷积核:")
    for i in range(weights.shape[0]):
        print(f"输出通道 {i+1}:")
        print(weights[i])
    
    # 执行卷积操作
    output = conv1d(input_data)
    print("\n输出数据形状:", output.shape)
    print("输出数据:")
    print(output)
    
    # 手动计算第一个输出通道的第一个位置，验证卷积过程
    print("\n手动计算第一个输出通道的第一个位置:")
    print("输出[0,0,0] = 偏置 + Σ(输入通道 × 对应卷积核)")
    
    # 第一个输出通道的偏置
    bias = conv1d.bias.data[0]
    print(f"偏置: {bias}")
    
    # 手动计算卷积
    manual_output = bias
    for in_ch in range(3):
        for k in range(3):
            # 由于padding=1，第一个位置对应输入的0,1,2位置
            manual_output += input_data[0, in_ch, k] * weights[0, in_ch, k]
    
    print(f"手动计算结果: {manual_output}")
    print(f"PyTorch计算结果: {output[0, 0, 0]}")
    print(f"差异: {abs(manual_output - output[0, 0, 0])}")

def explain_conv1d_channels():
    """
    解释Conv1D中多通道卷积的工作原理
    """
    print("=" * 60)
    print("Conv1D中多通道卷积的工作原理")
    print("=" * 60)
    
    print("\n1. 输入数据格式:")
    print("   - 形状: (batch_size, in_channels, sequence_length)")
    print("   - 例如: (1, 330, 128) 表示1个样本，330个通道，128个时间步")
    
    print("\n2. 卷积核格式:")
    print("   - 形状: (out_channels, in_channels, kernel_size)")
    print("   - 例如: (512, 330, 3) 表示512个输出通道，每个输出通道有330个输入通道的卷积核")
    
    print("\n3. 卷积过程:")
    print("   - 每个输出通道都有一组完整的输入通道卷积核")
    print("   - 对于每个输出通道:")
    print("     a. 对每个输入通道应用对应的卷积核")
    print("     b. 将所有输入通道的卷积结果相加")
    print("     c. 加上该输出通道的偏置项")
    
    print("\n4. 数学表达式:")
    print("   output[b, out_c, t] = bias[out_c] + Σ(input[b, in_c, t+k] * weight[out_c, in_c, k])")
    print("   其中:")
    print("   - b: batch索引")
    print("   - out_c: 输出通道索引")
    print("   - in_c: 输入通道索引")
    print("   - t: 时间位置")
    print("   - k: 卷积核位置")
    
    print("\n5. 在RVQ-VAE中的应用:")
    print("   - 输入: (512, 330, 128) - 512个样本，330个姿态特征，128个时间步")
    print("   - 卷积: Conv1d(330, 512, 3, 1, 1)")
    print("   - 输出: (512, 512, 128) - 512个样本，512个特征，128个时间步")
    print("   - 作用: 将330维的姿态特征扩展到512维，同时保持时间分辨率")

if __name__ == "__main__":
    explain_conv1d_channels()
    print("\n" + "=" * 60)
    print("具体示例演示")
    print("=" * 60)
    visualize_conv1d_channels()