import torch
import torch.nn as nn
import numpy as np

def correct_conv1d_demo():
    """
    正确演示Conv1D中多通道数据如何进行卷积
    """
    # 创建一个简单的Conv1d层
    # 输入通道: 3, 输出通道: 2, 卷积核大小: 3
    conv1d = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    
    # 为了便于验证，我们手动设置权重和偏置
    with torch.no_grad():
        conv1d.weight.data = torch.tensor([
            # 输出通道1的权重 (3个输入通道，每个通道3个卷积核元素)
            [
                [0.1, 0.2, 0.3],  # 输入通道1的卷积核
                [0.4, 0.5, 0.6],  # 输入通道2的卷积核
                [0.7, 0.8, 0.9],  # 输入通道3的卷积核
            ],
            # 输出通道2的权重
            [
                [0.2, 0.3, 0.4],  # 输入通道1的卷积核
                [0.5, 0.6, 0.7],  # 输入通道2的卷积核
                [0.8, 0.9, 1.0],  # 输入通道3的卷积核
            ]
        ])
        
        conv1d.bias.data = torch.tensor([0.1, 0.2])  # 两个输出通道的偏置
    
    # 创建输入数据 (batch_size=1, channels=3, length=5)
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
    # 由于padding=1，第一个位置对应输入的0,0, 0,1, 0,2位置
    manual_output = bias
    
    # 对每个输入通道进行卷积
    for in_ch in range(3):
        # 对于padding=1，第一个位置的卷积计算
        # 左边padding一个0，右边padding一个0
        padded_input = torch.cat([torch.zeros(1), input_data[0, in_ch, :2], torch.zeros(1)])
        print(f"输入通道{in_ch+1}的填充数据: {padded_input}")
        
        # 卷积核
        kernel = weights[0, in_ch]
        print(f"输入通道{in_ch+1}的卷积核: {kernel}")
        
        # 计算卷积
        conv_result = torch.sum(padded_input * kernel)
        print(f"输入通道{in_ch+1}的卷积结果: {conv_result}")
        
        manual_output += conv_result
    
    print(f"手动计算结果: {manual_output}")
    print(f"PyTorch计算结果: {output[0, 0, 0]}")
    print(f"差异: {abs(manual_output - output[0, 0, 0])}")
    
    # 验证第二个位置
    print("\n手动计算第一个输出通道的第二个位置:")
    manual_output_pos1 = bias
    
    for in_ch in range(3):
        # 第二个位置对应输入的0,0, 0,1, 0,2位置
        padded_input = torch.cat([torch.zeros(1), input_data[0, in_ch, :3], torch.zeros(1)])
        kernel = weights[0, in_ch]
        conv_result = torch.sum(padded_input[1:4] * kernel)
        manual_output_pos1 += conv_result
    
    print(f"手动计算结果: {manual_output_pos1}")
    print(f"PyTorch计算结果: {output[0, 0, 1]}")
    print(f"差异: {abs(manual_output_pos1 - output[0, 0, 1])}")

def explain_padding():
    """
    解释padding在Conv1D中的作用
    """
    print("=" * 60)
    print("Conv1D中Padding的作用")
    print("=" * 60)
    
    print("\n1. Padding=1的含义:")
    print("   - 在序列的两侧各填充1个零")
    print("   - 原始序列: [x1, x2, x3, x4, x5]")
    print("   - 填充后序列: [0, x1, x2, x3, x4, x5, 0]")
    
    print("\n2. 卷积核大小为3时:")
    print("   - 第一个输出位置: 0*x1 + x1*x2 + x2*x3")
    print("   - 第二个输出位置: x1*x1 + x2*x2 + x3*x3")
    print("   - 最后一个输出位置: x4*x1 + x5*x2 + 0*x3")
    
    print("\n3. Padding的作用:")
    print("   - 保持输出序列长度与输入序列长度相同")
    print("   - 使边界位置的卷积计算更加合理")
    print("   - 减少边界效应")

if __name__ == "__main__":
    explain_padding()
    print("\n" + "=" * 60)
    print("修正版Conv1D多通道卷积演示")
    print("=" * 60)
    correct_conv1d_demo()