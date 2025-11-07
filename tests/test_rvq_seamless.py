#!/usr/bin/env python3
"""
测试rvq_seamless_multi_gpu.py的功能
"""
import os
import sys
import subprocess
import time

def test_single_gpu():
    """测试单GPU训练"""
    print("=" * 50)
    print("测试单GPU训练")
    print("=" * 50)
    
    # 设置环境变量，禁用分布式训练
    env = os.environ.copy()
    env.pop('RANK', None)
    env.pop('WORLD_SIZE', None)
    env.pop('LOCAL_RANK', None)
    env.pop('MASTER_ADDR', None)
    env.pop('MASTER_PORT', None)
    
    # 构建命令
    cmd = [
        "python", "rvq_seamless_multi_gpu.py",
        "--batch_size", "4",
        "--total_iter", "3",
        "--max_samples", "10",
        "--window_size", "32",
        "--window_stride", "16",
        "--cache_train", "datasets/window_params/window_params_train_ws64_ws20_fixed.pkl",
        "--cache_val", "datasets/window_params/window_params_val_ws64_ws20_fixed.pkl",
        "--print_iter", "1",
        "--eval_iter", "2"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时输出
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    # 等待进程完成
    return_code = process.wait()
    
    if return_code == 0:
        print("✅ 单GPU测试成功!")
        return True
    else:
        print(f"❌ 单GPU测试失败，返回码: {return_code}")
        return False

def test_multi_gpu():
    """测试多GPU训练"""
    print("=" * 50)
    print("测试多GPU训练")
    print("=" * 50)
    
    # 检查GPU数量
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], 
                              capture_output=True, text=True)
        gpu_count = int(result.stdout.strip())
        print(f"检测到 {gpu_count} 个GPU")
        
        if gpu_count < 2:
            print("⚠️ GPU数量少于2，跳过多GPU测试")
            return True
    except Exception as e:
        print(f"❌ 检测GPU数量失败: {e}")
        return False
    
    # 构建命令
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node", "2",
        "--master_port", "29501",
        "rvq_seamless_multi_gpu.py",
        "--batch_size", "4",
        "--total_iter", "3",
        "--max_samples", "10",
        "--window_size", "32",
        "--window_stride", "16",
        "--cache_train", "datasets/window_params/window_params_train_ws64_ws20_fixed.pkl",
        "--cache_val", "datasets/window_params/window_params_val_ws64_ws20_fixed.pkl",
        "--print_iter", "1",
        "--eval_iter", "2"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时输出
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    # 等待进程完成
    return_code = process.wait()
    
    if return_code == 0:
        print("✅ 多GPU测试成功!")
        return True
    else:
        print(f"❌ 多GPU测试失败，返回码: {return_code}")
        return False

def main():
    """主函数"""
    print("开始测试rvq_seamless_multi_gpu.py...")
    
    # 检查必要文件是否存在
    required_files = [
        "rvq_seamless_multi_gpu.py",
        "datasets/window_params/window_params_train_ws64_ws20_fixed.pkl",
        "datasets/window_params/window_params_val_ws64_ws20_fixed.pkl",
        "mean_std/seamless_smplh_mean.npy",
        "mean_std/seamless_smplh_std.npy"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return
    
    print("✅ 所有必要文件都存在")
    
    # 测试单GPU
    single_gpu_success = test_single_gpu()
    
    # 等待一下
    time.sleep(2)
    
    # 测试多GPU
    multi_gpu_success = test_multi_gpu()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"单GPU测试: {'✅ 成功' if single_gpu_success else '❌ 失败'}")
    print(f"多GPU测试: {'✅ 成功' if multi_gpu_success else '❌ 失败'}")
    
    if single_gpu_success and multi_gpu_success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()