import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/embodied/yangchenyu/GestureLSM')
from models.vq.resnet import ResConv1DBlock, Resnet1D

def visualize_resnet1d_structure():
    """
    可视化ResNet1D的结构和数据处理流程
    """
    print("=" * 80)
    print("ResNet1D 结构可视化演示")
    print("=" * 80)
    
    # 创建一个简单的运动序列数据
    batch_size = 1
    channels = 3  # 3个通道，代表x,y,z坐标
    seq_length = 20  # 20个时间步
    
    # 创建一个简单的正弦波运动数据
    t = np.linspace(0, 4*np.pi, seq_length)
    x_data = np.sin(t)  # x坐标
    y_data = np.cos(t)  # y坐标
    z_data = np.sin(2*t)  # z坐标
    
    # 转换为PyTorch张量
    motion_data = torch.tensor(np.stack([x_data, y_data, z_data]), dtype=torch.float32).unsqueeze(0)
    print(f"输入运动数据形状: {motion_data.shape}")
    print("(batch_size, channels, sequence_length)")
    
    # 创建一个ResNet1D
    resnet1d = Resnet1D(
        n_in=channels,
        n_depth=3,
        dilation_growth_rate=2,
        reverse_dilation=False,
        activation='relu',
        norm=None
    )
    
    print("\nResNet1D结构:")
    print(resnet1d)
    
    # 通过ResNet1D处理数据
    with torch.no_grad():
        output = resnet1d(motion_data)
    
    print(f"\n输出数据形状: {output.shape}")
    print("注意: ResNet1D不改变数据的形状，只提取特征")
    
    # 可视化输入和输出
    plt.figure(figsize=(12, 8))
    
    # 绘制输入数据
    plt.subplot(2, 1, 1)
    plt.plot(t, x_data, 'r-', label='X坐标', linewidth=2)
    plt.plot(t, y_data, 'g-', label='Y坐标', linewidth=2)
    plt.plot(t, z_data, 'b-', label='Z坐标', linewidth=2)
    plt.title('输入运动数据 (原始信号)', fontsize=14, fontweight='bold')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('坐标值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 绘制输出数据
    plt.subplot(2, 1, 2)
    output_np = output[0].detach().numpy()
    plt.plot(t, output_np[0], 'r--', label='通道1输出', linewidth=2)
    plt.plot(t, output_np[1], 'g--', label='通道2输出', linewidth=2)
    plt.plot(t, output_np[2], 'b--', label='通道3输出', linewidth=2)
    plt.title('ResNet1D输出特征 (处理后)', fontsize=14, fontweight='bold')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('特征值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/resnet1d_visualization.png')
    print(f"\n可视化结果已保存到: /home/embodied/yangchenyu/GestureLSM/visualizations/resnet1d_visualization.png")
    
    # 解释ResNet1D的工作原理
    print("\n" + "=" * 80)
    print("ResNet1D 工作原理详解")
    print("=" * 80)
    
    print("\n1. 基本构建块: ResConv1DBlock")
    print("-" * 40)
    print("每个ResConv1DBlock包含:")
    print("  - 归一化层 (可选)")
    print("  - 激活函数 (ReLU)")
    print("  - 膨胀卷积层")
    print("  - 1x1卷积层")
    print("  - Dropout层")
    print("  - 残差连接 (输入 + 输出)")
    
    print("\n2. 膨胀卷积的作用")
    print("-" * 40)
    print("膨胀卷积通过在卷积核元素之间插入空白来扩大感受野:")
    print("  - dilation=1: 标准卷积，感受野=3")
    print("  - dilation=2: 感受野=5")
    print("  - dilation=4: 感受野=9")
    print("\n这允许网络在不增加参数的情况下捕获更长时间范围的依赖关系")
    
    print("\n3. 残差连接的作用")
    print("-" * 40)
    print("残差连接将输入直接加到输出上:")
    print("  - 解决梯度消失问题")
    print("  - 允许信息直接流动")
    print("  - 使网络更容易学习恒等映射")
    
    print("\n4. 多层堆叠的效果")
    print("-" * 40)
    print("多个ResConv1DBlock堆叠形成ResNet1D:")
    print("  - 每个块有不同膨胀率，捕获不同时间尺度的模式")
    print("  - 浅层捕获局部模式")
    print("  - 深层捕获全局模式")
    print("  - 残差连接保持原始信息")
    
    print("\n5. 在RVQ-VAE中的应用")
    print("-" * 40)
    print("在RVQ-VAE中，ResNet1D用于:")
    print("  - 编码器: 提取运动序列的层次特征")
    print("  - 解码器: 从量化特征重建运动序列")
    print("  - 不同膨胀率捕获不同时间尺度的运动模式")
    print("  - 残差连接保持运动连续性")

def demonstrate_dilation_effect():
    """
    演示膨胀率对卷积效果的影响
    """
    print("\n" + "=" * 80)
    print("膨胀率效果演示")
    print("=" * 80)
    
    # 创建一个简单的脉冲信号
    seq_length = 50
    signal = torch.zeros(1, 1, seq_length)
    signal[0, 0, 25] = 1.0  # 在中心位置放置一个脉冲
    
    print("输入信号: 在中心位置有一个脉冲")
    
    # 创建不同膨胀率的卷积层
    dilations = [1, 2, 4, 8]
    outputs = []
    
    plt.figure(figsize=(12, 10))
    
    # 绘制输入信号
    plt.subplot(len(dilations)+1, 1, 1)
    plt.plot(signal[0, 0].detach().numpy())
    plt.title('输入信号')
    plt.grid(True)
    
    # 对每个膨胀率进行卷积
    for i, dilation in enumerate(dilations):
        # 创建卷积层
        conv = nn.Conv1d(1, 1, kernel_size=3, dilation=dilation, padding=dilation)
        with torch.no_grad():
            conv.weight.fill_(1.0/3.0)  # 平均权重
            conv.bias.fill_(0.0)
        
        # 卷积操作
        output = conv(signal)
        outputs.append(output)
        
        # 计算感受野
        receptive_field = (3 - 1) * dilation + 1
        
        # 绘制输出
        plt.subplot(len(dilations)+1, 1, i+2)
        plt.plot(output[0, 0].detach().numpy())
        plt.title(f'膨胀率={dilation}, 感受野={receptive_field}')
        plt.grid(True)
        
        print(f"膨胀率 {dilation}: 感受野大小 {receptive_field}")
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/dilation_effect.png')
    print(f"\n膨胀率效果可视化已保存到: /home/embodied/yangchenyu/GestureLSM/visualizations/dilation_effect.png")
    
    print("\n膨胀率对卷积效果的影响:")
    print("- 膨胀率越大，卷积核覆盖的范围越大")
    print("- 大膨胀率可以捕获长距离依赖")
    print("- 小膨胀率专注于局部模式")
    print("- 在ResNet1D中，不同层使用不同膨胀率来捕获多尺度特征")

def demonstrate_residual_connection():
    """
    演示残差连接的效果
    """
    print("\n" + "=" * 80)
    print("残差连接效果演示")
    print("=" * 80)
    
    # 创建一个简单的信号
    seq_length = 30
    signal = torch.sin(torch.linspace(0, 4*np.pi, seq_length)).unsqueeze(0).unsqueeze(0)
    
    # 创建一个简单的卷积网络
    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x
    
    # 创建一个带残差连接的网络
    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2)
            
        def forward(self, x):
            residual = x
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x + residual
    
    # 创建网络实例
    simple_conv = SimpleConv()
    simple_resnet = SimpleResNet()
    
    # 设置相同的权重
    with torch.no_grad():
        simple_resnet.conv1.weight.copy_(simple_conv.conv1.weight)
        simple_resnet.conv1.bias.copy_(simple_conv.conv1.bias)
        simple_resnet.conv2.weight.copy_(simple_conv.conv2.weight)
        simple_resnet.conv2.bias.copy_(simple_conv.conv2.bias)
    
    # 前向传播
    with torch.no_grad():
        output_conv = simple_conv(signal)
        output_resnet = simple_resnet(signal)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 输入信号
    plt.subplot(3, 1, 1)
    plt.plot(signal[0, 0].detach().numpy())
    plt.title('输入信号')
    plt.grid(True)
    
    # 普通卷积输出
    plt.subplot(3, 1, 2)
    plt.plot(output_conv[0, 0].detach().numpy())
    plt.title('普通卷积输出')
    plt.grid(True)
    
    # 残差网络输出
    plt.subplot(3, 1, 3)
    plt.plot(output_resnet[0, 0].detach().numpy())
    plt.title('残差网络输出')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/embodied/yangchenyu/GestureLSM/visualizations/residual_connection.png')
    print(f"残差连接效果可视化已保存到: /home/embodied/yangchenyu/GestureLSM/visualizations/residual_connection.png")
    
    print("\n残差连接的效果:")
    print("- 保留原始信号的特征")
    print("- 防止信息丢失")
    print("- 使网络更容易学习")
    print("- 解决梯度消失问题")

if __name__ == "__main__":
    visualize_resnet1d_structure()
    demonstrate_dilation_effect()
    demonstrate_residual_connection()