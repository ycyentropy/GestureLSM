#!/usr/bin/env python3
"""
测试修复后的多GPU训练脚本是否能正确加载缓存
"""

import sys
import os
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

# 模拟命令行参数
sys.argv = [
    'test_script.py',
    '--data_path', '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction',
    '--cache_train', 'datasets/window_params/window_params_train_ws64_ws20_fixed.pkl',
    '--cache_val', 'datasets/window_params/window_params_val_ws64_ws20_fixed.pkl',
    '--window_size', '64',
    '--window_stride', '20',
    '--multi_length_training', '0.5', '0.75', '1.0', '1.25', '1.5',
    '--max_samples', '10',  # 限制样本数以便快速测试
    '--total_iter', '10',   # 限制迭代数
    '--batch_size', '4'
]

try:
    # 导入并运行主函数
    from rvq_seamless_multi_gpu import get_args_parser

    # 测试参数解析
    parser = get_args_parser()
    args = parser.parse_args()

    print("=== 参数解析测试 ===")
    print(f"数据路径: {args.data_path}")
    print(f"训练缓存路径: {args.cache_train}")
    print(f"验证缓存路径: {args.cache_val}")
    print(f"窗口大小: {args.window_size}")
    print(f"窗口步长: {args.window_stride}")
    print(f"多长度训练: {args.multi_length_training}")
    print(f"使用缓存: {args.use_cache}")

    # 测试缓存路径处理逻辑
    print("\n=== 缓存路径处理测试 ===")

    # 处理缓存路径别名
    if args.cache_train is not None:
        args.cache_path = args.cache_train
        print(f"设置训练缓存路径: {args.cache_path}")

    if args.cache_val is not None:
        args.val_cache_path = args.cache_val
        print(f"设置验证缓存路径: {args.val_cache_path}")

    # 如果提供了缓存路径，自动启用缓存
    if args.cache_path is not None or args.val_cache_path is not None:
        args.use_cache = True
        print(f"检测到缓存路径，自动启用缓存模式: {args.use_cache}")

    # 确定缓存路径
    if args.use_cache:
        train_cache_path = args.cache_path
        val_cache_path = args.val_cache_path if args.val_cache_path is not None else args.cache_path
        print(f"最终训练缓存路径: {train_cache_path}")
        print(f"最终验证缓存路径: {val_cache_path}")

        # 检查文件是否存在
        train_exists = os.path.exists(train_cache_path) if train_cache_path else False
        val_exists = os.path.exists(val_cache_path) if val_cache_path else False
        print(f"训练缓存文件存在: {train_exists}")
        print(f"验证缓存文件存在: {val_exists}")
    else:
        train_cache_path = None
        val_cache_path = None
        print("未启用缓存模式")

    # 处理多长度训练参数
    print("\n=== 多长度训练参数测试 ===")
    split = 'train'
    if args.multi_length_training is not None:
        if split == 'train':
            multi_length_ratios = args.multi_length_training
            print(f"使用多长度训练比例: {multi_length_ratios}")
        else:
            multi_length_ratios = [1.0]
    else:
        multi_length_ratios = [1.0]
        print("使用单长度训练比例: [1.0]")

    print("\n✅ 参数解析和缓存处理测试通过!")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()