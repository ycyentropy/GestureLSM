import time
import torch
from torch.utils.data import DataLoader
from dataloaders.seamless_interaction import SeamlessInteractionDataset, SeamlessInteractionWindowDataset

def test_data_loading_time():
    """测试加载单个数据样本的时间"""
    print("创建数据集...")
    
    # 创建数据集，禁用音视频加载以加快速度
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised/train"
    
    # 先创建基础数据集并限制大小
    base_dataset = SeamlessInteractionDataset(
        data_path=data_path,
        split="train",
        load_video=False,
        load_audio=False
    )
    
    # 限制基础数据集大小为5个样本
    limited_base_dataset = torch.utils.data.Subset(base_dataset, range(min(5, len(base_dataset))))
    print(f"基础数据集大小限制为: {len(limited_base_dataset)}")
    
    # 创建窗口数据集，基于限制后的基础数据集
    print("创建窗口数据集...")
    dataset = SeamlessInteractionWindowDataset(
        data_path=data_path,
        split="train",
        load_video=False,
        load_audio=False
    )
    
    # 限制窗口数据集大小
    limited_dataset = torch.utils.data.Subset(dataset, range(min(10, len(dataset))))
    print(f"窗口数据集大小限制为: {len(limited_dataset)}")
    
    print(f"数据集创建完成，共有 {len(limited_dataset)} 个样本")
    
    # 测试加载单个样本的时间
    print("\n测试加载单个样本的时间...")
    start_time = time.time()
    
    sample = limited_dataset[0]
    
    end_time = time.time()
    loading_time = end_time - start_time
    
    print(f"加载单个样本耗时: {loading_time:.4f} 秒")
    
    # 打印样本信息
    print("\n样本信息:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # 测试加载多个样本的平均时间
    print("\n测试加载多个样本的平均时间...")
    num_samples = min(5, len(limited_dataset))
    total_time = 0
    
    for i in range(num_samples):
        start_time = time.time()
        sample = limited_dataset[i]
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_samples
    print(f"加载 {num_samples} 个样本的平均耗时: {avg_time:.4f} 秒")
    
    # 测试DataLoader的批次加载时间
    print("\n测试DataLoader的批次加载时间...")
    dataloader = DataLoader(limited_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    start_time = time.time()
    batch = next(iter(dataloader))
    end_time = time.time()
    
    batch_loading_time = end_time - start_time
    print(f"加载一个批次(2个样本)耗时: {batch_loading_time:.4f} 秒")
    
    print("\n批次信息:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")

if __name__ == "__main__":
    test_data_loading_time()