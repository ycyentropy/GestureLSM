import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def correct_final_conv1d_demo():
    """
    完全正确的Conv1D多通道卷积演示
    """
    # 创建一个简单的Conv1d层
    # 输入通道: 3, 输出通道: 2, 卷积核大小: 3, padding=1
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
    
    # 手动计算第一个输出通道的第一个位置
    print("\n手动计算第一个输出通道的第一个位置:")
    print("输出[0,0,0] = 偏置 + Σ(输入通道 × 对应卷积核)")
    
    # 第一个输出通道的偏置
    bias = conv1d.bias.data[0]
    print(f"偏置: {bias}")
    
    # 手动计算卷积
    # 使用F.pad进行padding
    padded_input = F.pad(input_data, (1, 1), mode='constant', value=0)
    print(f"\n填充后的输入数据形状: {padded_input.shape}")
    print("填充后的输入数据:")
    print(padded_input)
    
    manual_output = bias
    
    # 对每个输入通道进行卷积
    for in_ch in range(3):
        # 获取第一个输入通道的填充数据
        channel_data = padded_input[0, in_ch]
        print(f"\n输入通道{in_ch+1}的填充数据: {channel_data}")
        
        # 卷积核
        kernel = weights[0, in_ch]
        print(f"输入通道{in_ch+1}的卷积核: {kernel}")
        
        # 计算卷积 - 第一个输出位置对应填充数据的0,1,2位置
        conv_result = torch.sum(channel_data[0:3] * kernel)
        print(f"输入通道{in_ch+1}的卷积结果: {conv_result}")
        print(f"详细计算: {channel_data[0]}*{kernel[0]} + {channel_data[1]}*{kernel[1]} + {channel_data[2]}*{kernel[2]} = {channel_data[0]*kernel[0]} + {channel_data[1]*kernel[1]} + {channel_data[2]*kernel[2]} = {conv_result}")
        
        manual_output += conv_result
    
    print(f"\n手动计算结果: {manual_output}")
    print(f"PyTorch计算结果: {output[0, 0, 0]}")
    print(f"差异: {abs(manual_output - output[0, 0, 0])}")
    
    # 验证第二个位置
    print("\n手动计算第一个输出通道的第二个位置:")
    manual_output_pos1 = bias
    
    for in_ch in range(3):
        # 获取第一个输入通道的填充数据
        channel_data = padded_input[0, in_ch]
        kernel = weights[0, in_ch]
        
        # 第二个输出位置对应填充数据的1,2,3位置
        conv_result = torch.sum(channel_data[1:4] * kernel)
        print(f"输入通道{in_ch+1}: {channel_data[1:4]} * {kernel} = {channel_data[1]*kernel[0]} + {channel_data[2]*kernel[1]} + {channel_data[3]*kernel[2]} = {conv_result}")
        manual_output_pos1 += conv_result
    
    print(f"手动计算结果: {manual_output_pos1}")
    print(f"PyTorch计算结果: {output[0, 0, 1]}")
    print(f"差异: {abs(manual_output_pos1 - output[0, 0, 1])}")

def explain_conv1d_math():
    """
    详细解释Conv1D的数学原理
    """
    print("=" * 60)
    print("Conv1D的数学原理")
    print("=" * 60)
    
    print("\n1. 输入数据格式:")
    print("   - 形状: (batch_size, in_channels, sequence_length)")
    print("   - 例如: (1, 3, 5) 表示1个样本，3个通道，每个通道5个时间步")
    
    print("\n2. 卷积核格式:")
    print("   - 形状: (out_channels, in_channels, kernel_size)")
    print("   - 例如: (2, 3, 3) 表示2个输出通道，每个通道有3个输入通道的卷积核，每个卷积核大小为3")
    
    print("\n3. Padding的作用:")
    print("   - padding=1 表示在序列的两侧各填充1个零")
    print("   - 原始序列: [x1, x2, x3, x4, x5]")
    print("   - 填充后序列: [0, x1, x2, x3, x4, x5, 0]")
    
    print("\n4. 卷积过程:")
    print("   - 每个输出通道对应一个卷积核组")
    print("   - 每个卷积核组包含in_channels个卷积核")
    print("   - 每个输入通道与对应的卷积核进行卷积")
    print("   - 所有输入通道的卷积结果相加，再加上偏置")
    
    print("\n5. 数学表达式:")
    print("   output[b, out_ch, t] = bias[out_ch] + Σ(input[b, in_ch, t:t+kernel_size] * weight[out_ch, in_ch, :])")
    
    print("\n6. 具体示例:")
    print("   - 输入: [1, 2, 3, 4, 5]")
    print("   - 卷积核: [0.1, 0.2, 0.3]")
    print("   - padding=1, 填充后: [0, 1, 2, 3, 4, 5, 0]")
    print("   - 第一个输出位置: 0*0.1 + 1*0.2 + 2*0.3 = 0 + 0.2 + 0.6 = 0.8")
    print("   - 第二个输出位置: 1*0.1 + 2*0.2 + 3*0.3 = 0.1 + 0.4 + 0.9 = 1.4")
    print("   - 第三个输出位置: 2*0.1 + 3*0.2 + 4*0.3 = 0.2 + 0.6 + 1.2 = 2.0")
    
    print("\n7. 多通道卷积:")
    print("   - 每个输入通道有对应的卷积核")
    print("   - 各通道的卷积结果相加")
    print("   - 最后加上偏置")

if __name__ == "__main__":
    explain_conv1d_math()
    print("\n" + "=" * 60)
    print("完全正确的Conv1D多通道卷积演示")
    print("=" * 60)
    correct_final_conv1d_demo()