import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def final_correct_conv1d_demo():
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
    
    # 验证第三个位置
    print("\n手动计算第一个输出通道的第三个位置:")
    manual_output_pos2 = bias
    
    for in_ch in range(3):
        # 获取第一个输入通道的填充数据
        channel_data = padded_input[0, in_ch]
        kernel = weights[0, in_ch]
        
        # 第三个输出位置对应填充数据的2,3,4位置
        conv_result = torch.sum(channel_data[2:5] * kernel)
        print(f"输入通道{in_ch+1}: {channel_data[2:5]} * {kernel} = {channel_data[2]*kernel[0]} + {channel_data[3]*kernel[1]} + {channel_data[4]*kernel[2]} = {conv_result}")
        manual_output_pos2 += conv_result
    
    print(f"手动计算结果: {manual_output_pos2}")
    print(f"PyTorch计算结果: {output[0, 0, 2]}")
    print(f"差异: {abs(manual_output_pos2 - output[0, 0, 2])}")

def verify_with_step_by_step():
    """
    使用分步计算验证Conv1D的工作原理
    """
    print("\n" + "=" * 60)
    print("使用分步计算验证Conv1D的工作原理")
    print("=" * 60)
    
    # 创建输入数据
    input_data = np.array([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # 通道1
            [2.0, 3.0, 4.0, 5.0, 6.0],  # 通道2
            [3.0, 4.0, 5.0, 6.0, 7.0],  # 通道3
        ]
    ], dtype=np.float32)
    
    # 创建卷积核权重
    weights = np.array([
        # 输出通道1的权重
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
    ], dtype=np.float32)
    
    # 偏置
    bias = np.array([0.1, 0.2], dtype=np.float32)
    
    print("输入数据:")
    print(input_data)
    print("\n卷积核权重:")
    print(weights)
    print("\n偏置:", bias)
    
    # 填充输入数据
    padded_input = np.pad(input_data, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
    print("\n填充后的输入数据:")
    print(padded_input)
    
    # 计算第一个输出通道的第一个位置
    print("\n计算第一个输出通道的第一个位置:")
    output_pos0 = bias[0]
    print(f"初始值（偏置）: {output_pos0}")
    
    for in_ch in range(3):
        # 获取输入数据
        window = padded_input[0, in_ch, 0:3]
        kernel = weights[0, in_ch]
        conv_result = np.sum(window * kernel)
        output_pos0 += conv_result
        print(f"输入通道{in_ch+1}: {window} * {kernel} = {conv_result}, 累加结果: {output_pos0}")
    
    print(f"最终结果: {output_pos0}")
    
    # 使用PyTorch验证
    torch_input = torch.tensor(input_data)
    torch_weights = torch.tensor(weights)
    torch_bias = torch.tensor(bias)
    torch_conv1d = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        torch_conv1d.weight.data = torch_weights
        torch_conv1d.bias.data = torch_bias
    torch_output = torch_conv1d(torch_input)
    
    print(f"PyTorch结果: {torch_output[0, 0, 0].item()}")
    print(f"差异: {abs(output_pos0 - torch_output[0, 0, 0].item())}")
    
    # 计算第一个输出通道的第二个位置
    print("\n计算第一个输出通道的第二个位置:")
    output_pos1 = bias[0]
    print(f"初始值（偏置）: {output_pos1}")
    
    for in_ch in range(3):
        # 获取输入数据
        window = padded_input[0, in_ch, 1:4]
        kernel = weights[0, in_ch]
        conv_result = np.sum(window * kernel)
        output_pos1 += conv_result
        print(f"输入通道{in_ch+1}: {window} * {kernel} = {conv_result}, 累加结果: {output_pos1}")
    
    print(f"最终结果: {output_pos1}")
    print(f"PyTorch结果: {torch_output[0, 0, 1].item()}")
    print(f"差异: {abs(output_pos1 - torch_output[0, 0, 1].item())}")

def explain_rvq_vae_conv1d():
    """
    解释RVQ-VAE中Conv1D的应用
    """
    print("\n" + "=" * 60)
    print("RVQ-VAE中Conv1D的应用")
    print("=" * 60)
    
    print("\n1. RVQ-VAE编码器中的Conv1D:")
    print("   - 代码: blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))")
    print("   - input_emb_width = 330 (输入特征维度)")
    print("   - width = 512 (输出特征维度)")
    print("   - kernel_size = 3 (卷积核大小)")
    print("   - stride = 1 (步长)")
    print("   - padding = 1 (填充)")
    
    print("\n2. 输入输出维度变化:")
    print("   - 输入: (batch_size, 330, sequence_length)")
    print("   - 输出: (batch_size, 512, sequence_length)")
    print("   - 作用: 将特征维度从330扩展到512，同时保持时间维度不变")
    
    print("\n3. 在RVQ-VAE中的位置:")
    print("   - 这是编码器的第一层")
    print("   - 用于初始特征提取和维度扩展")
    print("   - 后续会经过下采样和ResNet1D块")
    
    print("\n4. 为什么使用Conv1D而不是Linear:")
    print("   - Conv1D能够捕捉局部时序特征")
    print("   - 卷积操作具有平移不变性")
    print("   - 参数共享，减少模型参数数量")
    print("   - 适合处理序列数据")

if __name__ == "__main__":
    explain_rvq_vae_conv1d()
    print("\n" + "=" * 60)
    print("完全正确的Conv1D多通道卷积演示")
    print("=" * 60)
    final_correct_conv1d_demo()
    verify_with_step_by_step()