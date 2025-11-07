import time
import torch
from torch.utils.data import DataLoader
from dataloaders.seamless_interaction import SeamlessInteractionDataset

def test_data_loading_time():
    """测试加载单个数据样本的时间"""
    print("开始测试数据加载时间...")
    
    # 设置数据路径
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train"
    
    # 创建基础数据集
    print("1. 创建基础数据集...")
    start_time = time.time()
    base_dataset = SeamlessInteractionDataset(
        data_path=data_path,
        split="train",
        load_video=False,
        load_audio=False
    )
    end_time = time.time()
    print(f"   基础数据集创建完成，耗时: {end_time - start_time:.4f} 秒")
    print(f"   找到 {len(base_dataset)} 个数据样本")
    
    # 测试加载单个原始样本的时间
    print("\n2. 测试加载单个原始样本的时间...")
    start_time = time.time()
    sample = base_dataset[0]
    end_time = time.time()
    loading_time = end_time - start_time
    print(f"   加载单个原始样本耗时: {loading_time:.4f} 秒")
    
    # 打印样本信息
    print("\n3. 样本信息:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")
    
    # 测试加载多个原始样本的平均时间
    print("\n4. 测试加载多个原始样本的平均时间...")
    num_samples = min(5, len(base_dataset))
    total_time = 0
    
    for i in range(num_samples):
        print(f"   加载样本 {i+1}/{num_samples}...")
        start_time = time.time()
        sample = base_dataset[i]
        end_time = time.time()
        sample_time = end_time - start_time
        total_time += sample_time
        print(f"   样本 {i+1} 加载耗时: {sample_time:.4f} 秒")
    
    avg_time = total_time / num_samples
    print(f"   加载 {num_samples} 个原始样本的平均耗时: {avg_time:.4f} 秒")
    
    # 测试DataLoader的批次加载时间
    print("\n5. 测试DataLoader的批次加载时间...")
    limited_dataset = torch.utils.data.Subset(base_dataset, range(min(5, len(base_dataset))))
    
    # 先检查样本长度是否一致
    print("   检查样本长度...")
    sample_lengths = []
    for i in range(min(3, len(limited_dataset))):
        sample = limited_dataset[i]
        if 'pose' in sample:
            sample_lengths.append(len(sample['pose']))
        elif 'keypoints' in sample:
            sample_lengths.append(len(sample['keypoints']))
    
    print(f"   样本长度: {sample_lengths}")
    
    # 使用自定义的collate函数处理不同长度的序列
    def custom_collate_fn(batch):
        """处理不同长度序列的自定义collate函数"""
        result = {}
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['pose', 'keypoints', 'emotion_scores', 'expression']:
                # 对于序列数据，不进行批次化，保持为列表
                result[key] = [item[key] for item in batch]
            else:
                # 对于非序列数据，尝试正常批次化
                try:
                    result[key] = torch.stack([item[key] for item in batch])
                except:
                    result[key] = [item[key] for item in batch]
        
        return result
    
    dataloader = DataLoader(limited_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    print("   创建DataLoader完成，开始加载批次...")
    start_time = time.time()
    batch = next(iter(dataloader))
    end_time = time.time()
    
    batch_loading_time = end_time - start_time
    print(f"   加载一个批次(2个样本)耗时: {batch_loading_time:.4f} 秒")
    
    print("\n6. 批次信息:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"   {key}: 列表，长度 {len(value)}")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"      第一个元素形状: {value[0].shape} ({value[0].dtype})")
        elif isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")
    
    # 测试手动创建时间窗口
    print("\n7. 测试手动创建时间窗口...")
    sample = base_dataset[0]
    
    # 获取序列长度
    if 'pose' in sample:
        seq_len = len(sample['pose'])
        print(f"   姿态序列长度: {seq_len}")
    elif 'keypoints' in sample:
        seq_len = len(sample['keypoints'])
        print(f"   关键点序列长度: {seq_len}")
    else:
        print("   未找到序列数据")
        return
    
    # 设置窗口参数
    window_size = int(2.0 * 30)  # 2秒 * 30fps
    window_stride = int(0.5 * 30)  # 0.5秒 * 30fps
    print(f"   窗口大小: {window_size} 帧")
    print(f"   窗口步长: {window_stride} 帧")
    
    # 计算窗口数量
    num_windows = (seq_len - window_size) // window_stride + 1
    print(f"   可创建 {num_windows} 个窗口")
    
    # 测试创建第一个窗口
    print("\n8. 测试创建第一个窗口...")
    start_time = time.time()
    
    window_data = {}
    start = 0
    end = start + window_size
    
    # 复制非序列数据
    for key, value in sample.items():
        if key not in ['pose', 'keypoints', 'emotion_scores', 'expression', 'audio']:
            window_data[key] = value
    
    # 提取序列数据的窗口
    if 'pose' in sample:
        window_data['pose'] = sample['pose'][start:end]
        
    if 'keypoints' in sample:
        window_data['keypoints'] = sample['keypoints'][start:end]
        
    if 'emotion_scores' in sample:
        window_data['emotion_scores'] = sample['emotion_scores'][start:end]
        
    if 'expression' in sample:
        window_data['expression'] = sample['expression'][start:end]
    
    end_time = time.time()
    window_creation_time = end_time - start_time
    print(f"   创建单个窗口耗时: {window_creation_time:.4f} 秒")
    
    print("\n9. 窗口信息:")
    for key, value in window_data.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")

if __name__ == "__main__":
    test_data_loading_time()