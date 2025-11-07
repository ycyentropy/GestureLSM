#!/usr/bin/env python3
"""
GPU显存监控脚本
使用方法: python monitor_gpu.py [interval_seconds]
"""

import time
import subprocess
import sys
import argparse
from datetime import datetime

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return None
        
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 6:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total': int(parts[2]),
                    'memory_used': int(parts[3]),
                    'memory_free': int(parts[4]),
                    'utilization': int(parts[5])
                })
        
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None

def print_gpu_memory():
    """打印GPU显存使用情况"""
    gpu_info = get_gpu_memory()
    if not gpu_info:
        print("无法获取GPU信息或没有可用的GPU")
        return
    
    print("\n" + "="*80)
    print(f"{'GPU ID':<8} {'名称':<25} {'总计':<10} {'已用':<10} {'可用':<10} {'利用率':<10}")
    print("="*80)
    
    for gpu in gpu_info:
        print(f"{gpu['index']:<8} {gpu['name']:<25} {gpu['memory_total']:<10}MB {gpu['memory_used']:<10}MB {gpu['memory_free']:<10}MB {gpu['utilization']:<10}%")
    
    print("="*80)
    
    # 检查GPU 0是否空闲
    gpu0 = next((gpu for gpu in gpu_info if gpu['index'] == 0), None)
    if gpu0:
        if gpu0['memory_used'] < 1000 and gpu0['utilization'] < 5:
            print("\n✓ GPU 0 空闲，可供其他用户使用")
        else:
            print("\n⚠ GPU 0 正在使用中")

def main():
    parser = argparse.ArgumentParser(description='GPU显存监控脚本')
    parser.add_argument('interval', type=int, nargs='?', default=5, 
                       help='监控间隔（秒），默认为5秒')
    parser.add_argument('--count', type=int, default=0,
                       help='监控次数，0表示无限循环，默认为0')
    
    args = parser.parse_args()
    
    print(f"开始监控GPU显存，间隔 {args.interval} 秒")
    if args.count > 0:
        print(f"将监控 {args.count} 次")
    else:
        print("将持续监控，按Ctrl+C停止")
    
    try:
        count = 0
        while args.count == 0 or count < args.count:
            print_gpu_memory()
            count += 1
            if args.count == 0 or count < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()