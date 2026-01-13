import numpy as np
import os
import glob

# 设置数据目录
data_dir = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction'

# 收集所有npz文件路径
npz_files = glob.glob(os.path.join(data_dir, '**/*.npz'), recursive=True)
print(f"找到 {len(npz_files)} 个npz文件")

# 采样前5个文件进行分析
sample_files = npz_files[:5]
z_values = []

for i, npz_file in enumerate(sample_files):
    try:
        data = np.load(npz_file)
        
        # 检查必要的键是否存在
        if 'smplh:translation' not in data:
            print(f"跳过文件 {npz_file}，缺少translation键")
            continue
        
        # 获取平移数据
        translation = data['smplh:translation']
        
        # 将平移数据从厘米转换为米
        translation = translation / 100.0
        
        print(f"\n文件 {i+1}: {os.path.basename(npz_file)}")
        print(f"平移数据形状: {translation.shape}")
        
        # 显示前10帧的平移数据
        frames_to_show = min(10, translation.shape[0])
        print(f"前{frames_to_show}帧的平移数据:")
        print(translation[:frames_to_show])
        
        # 计算并显示z轴统计信息
        z_values.extend(translation[:, 2].tolist())
        print(f"z轴范围: [{translation[:, 2].min():.4f}, {translation[:, 2].max():.4f}]")
        print(f"z轴均值: {translation[:, 2].mean():.4f}")
        print(f"z轴标准差: {translation[:, 2].std():.4f}")
        
    except Exception as e:
        print(f"处理文件 {npz_file} 时出错: {str(e)}")

# 计算所有采样文件的z轴整体统计
if z_values:
    z_values = np.array(z_values)
    print(f"\n=== 所有采样文件z轴数据统计 ===")
    print(f"z轴范围: [{z_values.min():.4f}, {z_values.max():.4f}]")
    print(f"z轴均值: {z_values.mean():.4f}")
    print(f"z轴标准差: {z_values.std():.4f}")
    
    # 查看z轴值的分布
    print(f"\nz轴值分布示例:")
    print(f"小于0的比例: {np.sum(z_values < 0) / len(z_values) * 100:.2f}%")
    print(f"0-10之间的比例: {np.sum((z_values >= 0) & (z_values <= 10)) / len(z_values) * 100:.2f}%")
    print(f"10-30之间的比例: {np.sum((z_values > 10) & (z_values <= 30)) / len(z_values) * 100:.2f}%")
    print(f"30-50之间的比例: {np.sum((z_values > 30) & (z_values <= 50)) / len(z_values) * 100:.2f}%")
    print(f"大于50的比例: {np.sum(z_values > 50) / len(z_values) * 100:.2f}%")

print("\n分析完成。")