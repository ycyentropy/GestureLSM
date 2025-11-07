import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 从项目中导入ResNet1D相关类
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')
from models.vq.resnet import ResConv1DBlock, Resnet1D

def visualize_resnet1d():
    """
    详细解释ResNet1D的结构和工作原理
    """
    print("=" * 80)
    print("ResNet1D 详细解释")
    print("=" * 80)
    
    print("\n1. ResNet1D 基本概念")
    print("-" * 40)
    print("ResNet1D 是一维残差网络，专门用于处理序列数据（如时间序列、音频、运动数据等）")
    print("它通过引入残差连接解决了深度网络中的梯度消失问题")
    print("在RVQ-VAE中，ResNet1D用于提取运动序列的时序特征")
    
    print("\n2. ResConv1DBlock 结构")
    print("-" * 40)
    print("ResConv1DBlock是ResNet1D的基本构建块，包含以下组件：")
    print("  - 归一化层 (可选: LN/GN/BN)")
    print("  - 激活函数 (可选: ReLU/SiLU/GELU)")
    print("  - 1D卷积层 (带膨胀率)")
    print("  - 残差连接 (跳跃连接)")
    
    print("\n3. ResConv1DBlock 详细结构")
    print("-" * 40)
    print("输入数据形状: (batch_size, channels, sequence_length)")
    print("\n前向传播过程:")
    print("1. 保存原始输入 x_orig = x")
    print("2. 第一个归一化层: norm1(x)")
    print("3. 第一个激活函数: activation1(norm1(x))")
    print("4. 第一个卷积层: conv1(activation1(norm1(x)))")
    print("5. 第二个归一化层: norm2(conv1(...))")
    print("6. 第二个激活函数: activation2(norm2(...))")
    print("7. 第二个卷积层: conv2(activation2(...))")
    print("8. Dropout层: dropout(conv2(...))")
    print("9. 残差连接: x + x_orig")
    
    print("\n4. 膨胀卷积 (Dilated Convolution)")
    print("-" * 40)
    print("膨胀卷积通过在卷积核元素之间插入空白来扩大感受野")
    print("膨胀率(dilation)控制插入空白的数量")
    print("  - dilation=1: 标准卷积，无空白")
    print("  - dilation=2: 每个元素间插入1个空白")
    print("  - dilation=3: 每个元素间插入2个空白")
    print("\n膨胀卷积的公式:")
    print("  有效感受野 = (kernel_size - 1) * dilation + 1")
    print("  例如: kernel_size=3, dilation=3 → 有效感受野 = (3-1)*3+1 = 7")
    
    print("\n5. ResNet1D 结构")
    print("-" * 40)
    print("ResNet1D由多个ResConv1DBlock堆叠而成")
    print("关键参数:")
    print("  - n_in: 输入通道数")
    print("  - n_depth: ResConv1DBlock的数量")
    print("  - dilation_growth_rate: 膨胀率增长率")
    print("  - reverse_dilation: 是否反转膨胀率顺序")
    print("\n膨胀率计算:")
    print("  如果dilation_growth_rate=3, n_depth=3:")
    print("    - Block 0: dilation = 3^0 = 1")
    print("    - Block 1: dilation = 3^1 = 3")
    print("    - Block 2: dilation = 3^2 = 9")
    print("  如果reverse_dilation=True，则顺序反转: [9, 3, 1]")
    
    print("\n6. 在RVQ-VAE中的应用")
    print("-" * 40)
    print("在RVQ-VAE编码器中:")
    print("  - Block1: ResNet1D(深度=3, 膨胀率=1)")
    print("  - Block2: ResNet1D(深度=3, 膨胀率=3)")
    print("\n在RVQ-VAE解码器中:")
    print("  - Block1: ResNet1D(深度=3, 膨胀率=3)")
    print("  - Block2: ResNet1D(深度=3, 膨胀率=1)")
    
    print("\n7. 创建一个简单的ResNet1D示例")
    print("-" * 40)
    
    # 创建一个ResNet1D实例
    resnet1d = Resnet1D(
        n_in=64,           # 输入通道数
        n_depth=3,         # 3个ResConv1DBlock
        dilation_growth_rate=3,  # 膨胀率增长率为3
        reverse_dilation=True,   # 反转膨胀率顺序
        activation='relu',  # 使用ReLU激活函数
        norm=None          # 不使用归一化
    )
    
    print(f"ResNet1D结构:")
    print(resnet1d)
    
    print("\n8. 测试ResNet1D")
    print("-" * 40)
    
    # 创建测试数据
    batch_size = 2
    channels = 64
    seq_length = 32
    x = torch.randn(batch_size, channels, seq_length)
    
    print(f"输入数据形状: {x.shape}")
    
    # 前向传播
    output = resnet1d(x)
    print(f"输出数据形状: {output.shape}")
    print("注意: ResNet1D不改变输入数据的形状，只进行特征提取")
    
    print("\n9. 单个ResConv1DBlock测试")
    print("-" * 40)
    
    # 创建单个ResConv1DBlock
    res_block = ResConv1DBlock(
        n_in=64,           # 输入通道数
        n_state=128,       # 中间状态通道数
        dilation=3,        # 膨胀率为3
        activation='relu', # 使用ReLU激活函数
        norm=None          # 不使用归一化
    )
    
    print(f"ResConv1DBlock结构:")
    print(res_block)
    
    # 测试单个块
    x_block = torch.randn(batch_size, 64, seq_length)
    output_block = res_block(x_block)
    print(f"输入形状: {x_block.shape}")
    print(f"输出形状: {output_block.shape}")
    print("注意: ResConv1DBlock也不改变输入数据的形状")
    
    print("\n10. 膨胀率对感受野的影响")
    print("-" * 40)
    
    # 测试不同膨胀率的卷积
    input_seq = torch.randn(1, 1, 21)  # 长度为21的序列
    print("输入序列长度: 21")
    
    dilations = [1, 3, 5, 7]
    for dilation in dilations:
        # 创建卷积层
        conv = nn.Conv1d(1, 1, kernel_size=3, dilation=dilation, padding=dilation)
        with torch.no_grad():
            conv.weight.fill_(1.0)  # 设置权重为1
            conv.bias.fill_(0.0)     # 设置偏置为0
        
        # 计算输出
        output = conv(input_seq)
        # 计算感受野
        receptive_field = (3 - 1) * dilation + 1
        
        print(f"膨胀率 {dilation}: 输出长度 {output.shape[2]}, 感受野 {receptive_field}")
    
    print("\n11. ResNet1D的优势")
    print("-" * 40)
    print("1. 梯度流动: 残差连接使梯度可以直接流向浅层")
    print("2. 特征重用: 原始特征可以直接传递到深层")
    print("3. 多尺度感受野: 不同膨胀率捕获不同时间尺度的模式")
    print("4. 参数效率: 相比纯堆叠卷积层，ResNet更参数高效")
    
    print("\n12. 在运动序列处理中的作用")
    print("-" * 40)
    print("1. 短期模式: 小膨胀率捕获局部运动模式")
    print("2. 长期依赖: 大膨胀率捕获长距离运动关系")
    print("3. 层次特征: 不同层提取不同抽象层次的特征")
    print("4. 时序一致性: 残差连接保持时序信息的连续性")
    
    print("\n" + "=" * 80)
    print("ResNet1D 解释完毕")
    print("=" * 80)

def demonstrate_residual_connection():
    """
    演示残差连接的作用
    """
    print("\n" + "=" * 80)
    print("残差连接演示")
    print("=" * 80)
    
    # 创建一个简单的网络，带和不带残差连接
    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
            
        def forward(self, x):
            residual = x
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            return x + residual
    
    # 创建测试数据
    x = torch.randn(1, 1, 10)
    
    # 创建网络
    simple_conv = SimpleConv()
    simple_resnet = SimpleResNet()
    
    # 设置相同的权重
    with torch.no_grad():
        simple_resnet.conv1.weight.copy_(simple_conv.conv1.weight)
        simple_resnet.conv1.bias.copy_(simple_conv.conv1.bias)
        simple_resnet.conv2.weight.copy_(simple_conv.conv2.weight)
        simple_resnet.conv2.bias.copy_(simple_conv.conv2.bias)
        simple_resnet.conv3.weight.copy_(simple_conv.conv3.weight)
        simple_resnet.conv3.bias.copy_(simple_conv.conv3.bias)
    
    # 前向传播
    out_conv = simple_conv(x)
    out_resnet = simple_resnet(x)
    
    print(f"输入数据: {x[0, 0].detach().numpy()}")
    print(f"普通卷积输出: {out_conv[0, 0].detach().numpy()}")
    print(f"残差网络输出: {out_resnet[0, 0].detach().numpy()}")
    print(f"残差连接效果: {out_resnet[0, 0].detach().numpy() - out_conv[0, 0].detach().numpy()}")
    print("注意: 残差网络的输出 = 普通卷积输出 + 原始输入")

if __name__ == "__main__":
    visualize_resnet1d()
    demonstrate_residual_connection()