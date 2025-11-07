#!/usr/bin/env python3
"""
分布式训练架构调试脚本
测试torch.distributed的各个组件是否正常工作
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import time
import traceback

def test_distributed_init(rank, world_size):
    """测试分布式初始化"""
    print(f"[进程 {rank}] 开始测试分布式初始化...")

    try:
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"[进程 {rank}] ✓ 进程组初始化成功")

        # 测试设备设置
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        print(f"[进程 {rank}] ✓ 设备设置成功: {device}")

        # 测试进程间通信
        if rank == 0:
            tensor = torch.tensor([rank], device=device)
            print(f"[进程 {rank}] 发送张量: {tensor}")
        else:
            tensor = torch.zeros(1, device=device)

        # 广播测试
        dist.broadcast(tensor, src=0)
        print(f"[进程 {rank}] ✓ 广播测试成功，接收到: {tensor}")

        # 清理
        dist.destroy_process_group()
        print(f"[进程 {rank}] ✓ 进程组销毁成功")

        return True

    except Exception as e:
        print(f"[进程 {rank}] ✗ 分布式初始化失败: {e}")
        traceback.print_exc()
        return False

def test_distributed_model(rank, world_size):
    """测试分布式模型包装"""
    print(f"[进程 {rank}] 开始测试分布式模型...")

    try:
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # 创建简单模型
        device = torch.device(f"cuda:{rank}")
        model = torch.nn.Linear(156, 156).to(device)

        # 包装为DDP模型
        ddp_model = DDP(model, device_ids=[rank])
        print(f"[进程 {rank}] ✓ DDP模型包装成功")

        # 测试前向传播
        batch_size = 4
        seq_len = 64
        input_tensor = torch.randn(batch_size, seq_len, 156, device=device)

        output = ddp_model(input_tensor)
        print(f"[进程 {rank}] ✓ DDP前向传播成功，输出形状: {output.shape}")

        # 清理
        dist.destroy_process_group()
        return True

    except Exception as e:
        print(f"[进程 {rank}] ✗ 分布式模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_distributed_dataloader(rank, world_size):
    """测试分布式数据加载器"""
    print(f"[进程 {rank}] 开始测试分布式数据加载器...")

    try:
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # 创建数据集（使用少量数据快速测试）
        from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
            split="train",
            window_size=64,
            window_stride=20,
            multi_length_training=[1.0],  # 只用单长度避免复杂性
            load_video=False,
            load_audio=False,
            max_samples=10,  # 只用10个样本
            cache_path="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
        )

        print(f"[进程 {rank}] 数据集长度: {len(dataset)}")

        # 创建分布式采样器
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        print(f"[进程 {rank}] ✓ 分布式采样器创建成功")

        # 创建DataLoader
        def simple_collate_fn(batch):
            max_len = max(item['pose'].shape[0] for item in batch)
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]

            poses = torch.zeros(batch_size, max_len, pose_dim)
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = pose

            return {'pose': poses, 'mask': torch.ones(batch_size, max_len, dtype=torch.bool)}

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            num_workers=0,
            drop_last=True,
            collate_fn=simple_collate_fn,
            pin_memory=True
        )
        print(f"[进程 {rank}] ✓ DataLoader创建成功，批次数: {len(dataloader)}")

        # 测试获取一个批次
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            batch_time = time.time()
            print(f"[进程 {rank}] ✓ 批次 {i} 获取成功，耗时: {batch_time - start_time:.2f}秒")
            print(f"[进程 {rank}]   批次形状: {batch['pose'].shape}")
            break  # 只测试第一个批次

        # 清理
        dist.destroy_process_group()
        return True

    except Exception as e:
        print(f"[进程 {rank}] ✗ 分布式数据加载器测试失败: {e}")
        traceback.print_exc()
        return False

def test_full_distributed_training(rank, world_size):
    """测试完整的分布式训练流程"""
    print(f"[进程 {rank}] 开始测试完整分布式训练...")

    try:
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12358'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")

        print(f"[进程 {rank}] ✓ 进程组初始化成功")

        # 创建模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(156, 156)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(device)
        ddp_model = DDP(model, device_ids=[rank])
        print(f"[进程 {rank}] ✓ 模型创建和DDP包装成功")

        # 创建优化器
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
        print(f"[进程 {rank}] ✓ 优化器创建成功")

        # 创建数据集和数据加载器
        from lazy_window_dataset import CachedLazySeamlessInteractionWindowDataset

        dataset = CachedLazySeamlessInteractionWindowDataset(
            data_path="/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction",
            split="train",
            window_size=64,
            window_stride=20,
            multi_length_training=[1.0],
            load_video=False,
            load_audio=False,
            max_samples=10,
            cache_path="datasets/window_params/window_params_train_ws64_ws20_fixed.pkl"
        )

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        def simple_collate_fn(batch):
            max_len = max(item['pose'].shape[0] for item in batch)
            batch_size = len(batch)
            pose_dim = batch[0]['pose'].shape[1]
            poses = torch.zeros(batch_size, max_len, pose_dim)
            for i, item in enumerate(batch):
                pose = item['pose']
                length = pose.shape[0]
                poses[i, :length] = pose
            return {'pose': poses, 'mask': torch.ones(batch_size, max_len, dtype=torch.bool)}

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            num_workers=0,
            collate_fn=simple_collate_fn
        )
        print(f"[进程 {rank}] ✓ 数据加载器创建成功")

        # 训练循环测试（2个迭代）
        print(f"[进程 {rank}] 开始训练循环测试...")
        for epoch in range(1):
            sampler.set_epoch(epoch)

            for i, batch in enumerate(dataloader):
                if i >= 2:  # 只测试2个批次
                    break

                start_time = time.time()

                # 数据移到GPU
                gt_motion = batch['pose'].to(device)

                # 前向传播
                pred_motion = ddp_model(gt_motion)

                # 计算损失
                loss = torch.nn.functional.mse_loss(pred_motion, gt_motion)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end_time = time.time()
                print(f"[进程 {rank}] ✓ 迭代 {i} 完成，损失: {loss.item():.5f}，耗时: {end_time - start_time:.2f}秒")

        # 清理
        dist.destroy_process_group()
        print(f"[进程 {rank}] ✓ 完整分布式训练测试成功")
        return True

    except Exception as e:
        print(f"[进程 {rank}] ✗ 完整分布式训练测试失败: {e}")
        traceback.print_exc()
        return False

def run_test(test_func, world_size, test_name):
    """运行测试函数"""
    print(f"\n=== {test_name} ===")
    print(f"使用 {world_size} 个GPU进行测试...")

    try:
        # 检查GPU数量
        if torch.cuda.device_count() < world_size:
            print(f"❌ 可用GPU数量({torch.cuda.device_count()})少于所需数量({world_size})")
            return False

        # 启动多进程测试
        mp.spawn(test_func, args=(world_size,), nprocs=world_size, join=True)
        print(f"✅ {test_name} 完成")
        return True

    except Exception as e:
        print(f"❌ {test_name} 失败: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== 分布式训练架构调试 ===")

    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return

    print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")

    # 测试不同数量的GPU
    test_world_sizes = [2, 4]  # 先测试小规模，避免资源不足

    for world_size in test_world_sizes:
        print(f"\n{'='*50}")
        print(f"测试 {world_size} GPU 分布式训练")
        print(f"{'='*50}")

        # 1. 测试分布式初始化
        if not run_test(test_distributed_init, world_size, f"{world_size}GPU 分布式初始化测试"):
            continue

        # 2. 测试分布式模型
        if not run_test(test_distributed_model, world_size, f"{world_size}GPU 分布式模型测试"):
            continue

        # 3. 测试分布式数据加载器
        if not run_test(test_distributed_dataloader, world_size, f"{world_size}GPU 分布式数据加载器测试"):
            continue

        # 4. 测试完整分布式训练
        if not run_test(test_full_distributed_training, world_size, f"{world_size}GPU 完整分布式训练测试"):
            continue

    print(f"\n{'='*50}")
    print("🎉 所有分布式训练组件测试完成")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()