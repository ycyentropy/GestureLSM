def get_args_parser():
    parser = argparse.ArgumentParser(description='RVQ-VAE training for Seamless Interaction - Optimized Version',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据集参数
    data_group = parser.add_argument_group('数据集参数')
    data_group.add_argument('--dataname', type=str, default='seamless_interaction', help='数据集名称')
    data_group.add_argument('--data_path', type=str, default='/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction', help='数据集路径')
    data_group.add_argument('--window-size', type=int, default=64, help='窗口大小(帧数)')
    data_group.add_argument('--window-stride', type=int, default=20, help='窗口步长(帧数)')
    data_group.add_argument('--multi_length_training', type=float, nargs='+', default=None, help='多长度训练比例列表')
    data_group.add_argument('--body-part', type=str, default='upper', choices=['whole', 'upper', 'lower', 'hands', 'face'], help='训练的身体部位')
    data_group.add_argument('--max_samples', type=int, default=None, help='最大样本数，设置为None使用整个数据集')
    
    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument("--code-dim", type=int, default=512, help="嵌入维度")
    model_group.add_argument("--nb-code", type=int, default=1024, help="嵌入数量")
    model_group.add_argument("--down-t", type=int, default=2, help="下采样率")
    model_group.add_argument("--stride-t", type=int, default=2, help="步长大小")
    model_group.add_argument("--width", type=int, default=1024, help="网络宽度")
    model_group.add_argument("--depth", type=int, default=6, help="网络深度")
    model_group.add_argument("--dilation-growth-rate", type=int, default=3, help="膨胀增长率")
    model_group.add_argument("--output-emb-width", type=int, default=1024, help="输出嵌入宽度")
    model_group.add_argument('--vq-act', type=str, default='gelu', choices=['relu', 'silu', 'gelu'], help='激活函数')
    model_group.add_argument('--vq-norm', type=str, default='layer_norm', choices=['batch_norm', 'layer_norm', 'none'], help='归一化方法')
    model_group.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'], help="量化器类型")
    model_group.add_argument("--num_quantizers", type=int, default=12, help="量化器数量")
    model_group.add_argument("--shared_codebook", type=bool, default=False, help="是否使用共享码本")
    model_group.add_argument("--quantize_dropout_prob", type=float, default=0.2, help="量化丢弃概率")
    model_group.add_argument('--beta', type=float, default=1.0, help='标准VQ中的承诺损失')
    model_group.add_argument("--mu", type=float, default=0.99, help="更新码本的指数移动平均")
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--total-iter', default=300000, type=int, help='总迭代次数')
    train_group.add_argument('--warm-up-iter', default=1000, type=int, help='预热迭代次数')
    train_group.add_argument('--lr', default=1e-4, type=float, help='最大学习率')
    train_group.add_argument('--lr-scheduler', default=[200000, 250000], nargs="+", type=int, help="学习率调度(迭代)")
    train_group.add_argument('--gamma', default=0.1, type=float, help="学习率衰减")
    train_group.add_argument('--weight-decay', default=0.01, type=float, help='权重衰减')
    train_group.add_argument("--commit", type=float, default=0.02, help="承诺损失的超参数")
    train_group.add_argument('--loss-vel', type=float, default=0.1, help='速度损失的超参数')
    train_group.add_argument('--recons-loss', type=str, default='l1_smooth', help='重建损失')
    train_group.add_argument('--batch-size', default=64, type=int, help='批次大小')
    train_group.add_argument('--eval-batch-size', type=int, default=128, help='评估批次大小')
    train_group.add_argument('--print-iter', default=200, type=int, help='打印频率')
    train_group.add_argument('--eval-iter', default=2000, type=int, help='评估频率')
    train_group.add_argument('--seed', default=123, type=int, help='初始化训练的种子')
    train_group.add_argument('--mode', type=str, default='train', help='训练或评估')
    
    # 数据加载优化参数
    data_opt_group = parser.add_argument_group('数据加载优化参数')
    data_opt_group.add_argument('--num-workers', type=int, default=None, help='数据加载器工作进程数(默认自动优化)')
    data_opt_group.add_argument('--prefetch-factor', type=int, default=None, help='预取因子(默认自动优化)')
    data_opt_group.add_argument('--use-fast-collate', action='store_true', default=True, help='使用快速批处理函数')
    data_opt_group.add_argument('--pin-memory', action='store_true', default=True, help='使用固定内存')
    data_opt_group.add_argument('--non-blocking-transfer', action='store_true', default=True, help='使用非阻塞传输')
    data_opt_group.add_argument('--persistent-workers', action='store_true', default=True, help='保持工作进程活跃')
    data_opt_group.add_argument('--cache-path', type=str, default=None, help='缓存文件路径')
    data_opt_group.add_argument('--val-cache-path', type=str, default=None, help='验证集缓存文件路径')
    data_opt_group.add_argument('--cache-train', type=str, default=None, help='训练集缓存文件路径(别名)')
    data_opt_group.add_argument('--cache-val', type=str, default=None, help='验证集缓存文件路径(别名)')
    data_opt_group.add_argument('--use-cache', action='store_true', default=False, help='是否使用缓存')
    data_opt_group.add_argument('--use-cache-mean-std', action='store_true', default=False, help='是否使用缓存的均值和标准差')
    
    # GPU内存和计算优化参数
    gpu_opt_group = parser.add_argument_group('GPU内存和计算优化参数')
    gpu_opt_group.add_argument('--max-batch-size', type=int, default=256, help='最大批次大小')
    gpu_opt_group.add_argument('--auto-tune-batch-size', action='store_true', default=True, help='自动调整批次大小')
    gpu_opt_group.add_argument('--adaptive-batching', action='store_true', default=True, help='自适应批次大小')
    gpu_opt_group.add_argument('--batch-adjust-interval', type=int, default=500, help='批次大小调整间隔(迭代)')
    gpu_opt_group.add_argument('--target-gpu-util', type=float, default=85.0, help='目标GPU利用率(%)')
    gpu_opt_group.add_argument('--gradient-accumulation-steps', type=int, default=1, help='梯度累积步数')
    gpu_opt_group.add_argument('--gradient-checkpointing', action='store_true', default=False, help='启用梯度检查点')
    gpu_opt_group.add_argument('--use-amp', action='store_true', default=True, help='使用自动混合精度')
    gpu_opt_group.add_argument('--memory-fraction', type=float, default=0.9, help='GPU内存分配比例')
    
    # 分布式训练参数
    dist_group = parser.add_argument_group('分布式训练参数')
    dist_group.add_argument('--local-rank', type=int, default=0, help='本地进程排名')
    dist_group.add_argument('--find-unused-parameters', action='store_true', default=False, help='查找未使用参数')
    dist_group.add_argument('--benchmark-distributed', action='store_true', default=False, help='运行分布式基准测试')
    
    # 性能监控参数
    perf_group = parser.add_argument_group('性能监控参数')
    perf_group.add_argument('--auto-save-interval', type=int, default=3600, help='自动保存间隔(秒)')
    perf_group.add_argument('--performance-log-interval', type=int, default=100, help='性能日志记录间隔(迭代)')
    perf_group.add_argument('--auto-tune-performance', action='store_true', default=False, help='启用自动性能调优')
    
    # 恢复训练参数
    resume_group = parser.add_argument_group('恢复训练参数')
    resume_group.add_argument("--resume-pth", type=str, default=None, help='VQ恢复路径')
    
    # 输出目录参数
    output_group = parser.add_argument_group('输出目录参数')
    output_group.add_argument('--out-dir', type=str, default='outputs/rvqvae_seamless_optimized', help='输出目录')
    output_group.add_argument('--exp-name', type=str, default='RVQVAE_Seamless_Optimized', help='实验名称')
    
    return parser