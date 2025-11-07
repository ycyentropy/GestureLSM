def setup_performance_monitoring(args, logger=None):
    """
    设置性能监控
    
    参数:
        args: 训练参数
        logger: 日志记录器
        
    返回:
        perf_monitor: 性能监控器
    """
    if logger:
        logger.info("设置性能监控...")
    
    # 创建性能监控器
    perf_monitor = {
        'start_time': time.time(),
        'iter_times': [],
        'gpu_memory_history': [],
        'gpu_util_history': [],
        'throughput_history': [],
        'loss_history': [],
        'best_loss': float('inf'),
        'last_save_time': time.time(),
        'auto_save_interval': args.auto_save_interval if hasattr(args, 'auto_save_interval') else 3600,  # 默认1小时
        'performance_log_interval': args.performance_log_interval if hasattr(args, 'performance_log_interval') else 100,  # 默认100次迭代
        'auto_tune_enabled': args.auto_tune_performance if hasattr(args, 'auto_tune_performance') else False,
        'tune_history': []
    }
    
    if logger:
        logger.info("性能监控设置完成")
    
    return perf_monitor


def update_performance_monitor(perf_monitor, nb_iter, loss, batch_size, device, logger=None):
    """
    更新性能监控
    
    参数:
        perf_monitor: 性能监控器
        nb_iter: 当前迭代次数
        loss: 当前损失
        batch_size: 当前批次大小
        device: 设备
        logger: 日志记录器
        
    返回:
        perf_monitor: 更新后的性能监控器
    """
    current_time = time.time()
    
    # 计算迭代时间
    if len(perf_monitor['iter_times']) > 0:
        iter_time = current_time - perf_monitor['iter_times'][-1]
    else:
        iter_time = current_time - perf_monitor['start_time']
    
    perf_monitor['iter_times'].append(current_time)
    
    # 计算吞吐量
    throughput = batch_size / iter_time
    perf_monitor['throughput_history'].append(throughput)
    
    # 记录损失
    perf_monitor['loss_history'].append(loss)
    
    # 更新最佳损失
    if loss < perf_monitor['best_loss']:
        perf_monitor['best_loss'] = loss
    
    # 监控GPU内存
    gpu_memory = monitor_gpu_memory(device.index)
    if gpu_memory:
        perf_monitor['gpu_memory_history'].append({
            'time': current_time,
            'utilization': gpu_memory['utilization'],
            'used': gpu_memory['used'],
            'total': gpu_memory['total']
        })
    
    # 监控GPU利用率
    gpu_util = monitor_gpu_utilization(device)
    if gpu_util:
        perf_monitor['gpu_util_history'].append({
            'time': current_time,
            'utilization': gpu_util['utilization'],
            'memory_util': gpu_util['memory_util']
        })
    
    # 定期记录性能日志
    if nb_iter % perf_monitor['performance_log_interval'] == 0 and logger:
        avg_throughput = sum(perf_monitor['throughput_history'][-perf_monitor['performance_log_interval']:]) / min(len(perf_monitor['throughput_history']), perf_monitor['performance_log_interval'])
        avg_loss = sum(perf_monitor['loss_history'][-perf_monitor['performance_log_interval']:]) / min(len(perf_monitor['loss_history']), perf_monitor['performance_log_interval'])
        
        if perf_monitor['gpu_memory_history']:
            avg_gpu_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history'][-perf_monitor['performance_log_interval']:]]) / min(len(perf_monitor['gpu_memory_history']), perf_monitor['performance_log_interval'])
        else:
            avg_gpu_util = 0
        
        logger.info(f"性能监控 - 迭代 {nb_iter}: 吞吐量 {avg_throughput:.2f} samples/s, 平均损失 {avg_loss:.5f}, GPU利用率 {avg_gpu_util:.1f}%")
    
    # 自动保存检查点
    if current_time - perf_monitor['last_save_time'] > perf_monitor['auto_save_interval']:
        perf_monitor['last_save_time'] = current_time
        if logger:
            logger.info(f"定期自动保存检查点 (每 {perf_monitor['auto_save_interval']} 秒)")
    
    return perf_monitor


def auto_tune_performance(perf_monitor, args, model, optimizer, device, logger=None):
    """
    自动调优性能参数
    
    参数:
        perf_monitor: 性能监控器
        args: 训练参数
        model: 模型
        optimizer: 优化器
        device: 设备
        logger: 日志记录器
        
    返回:
        args: 更新后的训练参数
        tune_actions: 执行的调优操作列表
    """
    tune_actions = []
    
    if not perf_monitor['auto_tune_enabled']:
        return args, tune_actions
    
    # 只在足够的历史数据后进行调优
    if len(perf_monitor['throughput_history']) < 10:
        return args, tune_actions
    
    # 计算最近的平均吞吐量和GPU利用率
    recent_throughput = sum(perf_monitor['throughput_history'][-10:]) / 10
    recent_loss = sum(perf_monitor['loss_history'][-10:]) / 10
    
    if perf_monitor['gpu_util_history']:
        recent_gpu_util = sum([u['utilization'] for u in perf_monitor['gpu_util_history'][-10:]]) / 10
    else:
        recent_gpu_util = 0
    
    if perf_monitor['gpu_memory_history']:
        recent_mem_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history'][-10:]]) / 10
    else:
        recent_mem_util = 0
    
    # 调优规则
    # 1. 如果GPU利用率低且内存使用率低，尝试增加批次大小
    if recent_gpu_util < 70 and recent_mem_util < 70 and hasattr(args, 'batch_size'):
        if args.batch_size < args.max_batch_size if hasattr(args, 'max_batch_size') else args.batch_size * 2:
            old_batch_size = args.batch_size
            args.batch_size = min(args.batch_size * 2, args.max_batch_size if hasattr(args, 'max_batch_size') else args.batch_size * 2)
            tune_actions.append(f"增加批次大小: {old_batch_size} -> {args.batch_size}")
            
            if logger:
                logger.info(f"自动调优: 增加批次大小 {old_batch_size} -> {args.batch_size} (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
    
    # 2. 如果GPU利用率高但内存使用率高，尝试启用梯度检查点
    elif recent_gpu_util > 80 and recent_mem_util > 85 and not args.gradient_checkpointing:
        args.gradient_checkpointing = True
        tune_actions.append("启用梯度检查点")
        
        if logger:
            logger.info(f"自动调优: 启用梯度检查点 (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
        
        # 如果模型支持梯度检查点，启用它
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
    
    # 3. 如果GPU利用率低但内存使用率高，尝试减少批次大小
    elif recent_gpu_util < 50 and recent_mem_util > 85 and hasattr(args, 'batch_size') and args.batch_size > 8:
        old_batch_size = args.batch_size
        args.batch_size = max(args.batch_size // 2, 8)
        tune_actions.append(f"减少批次大小: {old_batch_size} -> {args.batch_size}")
        
        if logger:
            logger.info(f"自动调优: 减少批次大小 {old_batch_size} -> {args.batch_size} (GPU利用率: {recent_gpu_util:.1f}%, 内存利用率: {recent_mem_util:.1f}%)")
    
    # 4. 如果损失增加，尝试降低学习率
    elif len(perf_monitor['loss_history']) >= 20:
        recent_loss = sum(perf_monitor['loss_history'][-10:]) / 10
        older_loss = sum(perf_monitor['loss_history'][-20:-10]) / 10
        
        if recent_loss > older_loss * 1.1:  # 损失增加了10%以上
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.9  # 降低10%
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            tune_actions.append(f"降低学习率: {old_lr:.6f} -> {new_lr:.6f}")
            
            if logger:
                logger.info(f"自动调优: 降低学习率 {old_lr:.6f} -> {new_lr:.6f} (损失增加: {older_loss:.5f} -> {recent_loss:.5f})")
    
    # 记录调优历史
    if tune_actions:
        perf_monitor['tune_history'].append({
            'time': time.time(),
            'actions': tune_actions,
            'throughput': recent_throughput,
            'gpu_util': recent_gpu_util,
            'mem_util': recent_mem_util,
            'loss': recent_loss
        })
    
    return args, tune_actions


def save_performance_report(perf_monitor, args, save_path, logger=None):
    """
    保存性能报告
    
    参数:
        perf_monitor: 性能监控器
        args: 训练参数
        save_path: 保存路径
        logger: 日志记录器
    """
    if logger:
        logger.info(f"保存性能报告到 {save_path}")
    
    # 计算总体统计
    total_time = time.time() - perf_monitor['start_time']
    total_iters = len(perf_monitor['iter_times'])
    
    if total_iters > 0:
        avg_iter_time = total_time / total_iters
        avg_throughput = sum(perf_monitor['throughput_history']) / len(perf_monitor['throughput_history']) if perf_monitor['throughput_history'] else 0
        avg_loss = sum(perf_monitor['loss_history']) / len(perf_monitor['loss_history']) if perf_monitor['loss_history'] else 0
    else:
        avg_iter_time = 0
        avg_throughput = 0
        avg_loss = 0
    
    if perf_monitor['gpu_memory_history']:
        avg_gpu_mem_util = sum([m['utilization'] for m in perf_monitor['gpu_memory_history']]) / len(perf_monitor['gpu_memory_history'])
    else:
        avg_gpu_mem_util = 0
    
    if perf_monitor['gpu_util_history']:
        avg_gpu_util = sum([u['utilization'] for u in perf_monitor['gpu_util_history']]) / len(perf_monitor['gpu_util_history'])
    else:
        avg_gpu_util = 0
    
    # 创建报告
    report = {
        'training_args': vars(args),
        'performance_summary': {
            'total_time': total_time,
            'total_iters': total_iters,
            'avg_iter_time': avg_iter_time,
            'avg_throughput': avg_throughput,
            'avg_loss': avg_loss,
            'best_loss': perf_monitor['best_loss'],
            'avg_gpu_memory_util': avg_gpu_mem_util,
            'avg_gpu_util': avg_gpu_util,
        },
        'tune_history': perf_monitor['tune_history'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存报告
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4, sort_keys=True)
    
    if logger:
        logger.info(f"性能报告已保存: 平均吞吐量 {avg_throughput:.2f} samples/s, 平均GPU利用率 {avg_gpu_util:.1f}%, 最佳损失 {perf_monitor['best_loss']:.5f}")


def plot_performance_curves(perf_monitor, save_path, logger=None):
    """
    绘制性能曲线
    
    参数:
        perf_monitor: 性能监控器
        save_path: 保存路径
        logger: 日志记录器
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        if logger:
            logger.info(f"绘制性能曲线到 {save_path}")
        
        # 创建子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 吞吐量曲线
        if perf_monitor['throughput_history']:
            axs[0, 0].plot(perf_monitor['throughput_history'])
            axs[0, 0].set_title('Throughput (samples/s)')
            axs[0, 0].set_xlabel('Iteration')
            axs[0, 0].set_ylabel('Throughput')
            axs[0, 0].grid(True)
        
        # 2. 损失曲线
        if perf_monitor['loss_history']:
            axs[0, 1].plot(perf_monitor['loss_history'])
            axs[0, 1].set_title('Training Loss')
            axs[0, 1].set_xlabel('Iteration')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].grid(True)
        
        # 3. GPU内存利用率曲线
        if perf_monitor['gpu_memory_history']:
            times = [m['time'] - perf_monitor['start_time'] for m in perf_monitor['gpu_memory_history']]
            utils = [m['utilization'] for m in perf_monitor['gpu_memory_history']]
            axs[1, 0].plot(times, utils)
            axs[1, 0].set_title('GPU Memory Utilization')
            axs[1, 0].set_xlabel('Time (s)')
            axs[1, 0].set_ylabel('Utilization (%)')
            axs[1, 0].grid(True)
        
        # 4. GPU利用率曲线
        if perf_monitor['gpu_util_history']:
            times = [u['time'] - perf_monitor['start_time'] for u in perf_monitor['gpu_util_history']]
            utils = [u['utilization'] for u in perf_monitor['gpu_util_history']]
            axs[1, 1].plot(times, utils)
            axs[1, 1].set_title('GPU Utilization')
            axs[1, 1].set_xlabel('Time (s)')
            axs[1, 1].set_ylabel('Utilization (%)')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        if logger:
            logger.info(f"性能曲线已保存")
    except ImportError:
        if logger:
            logger.warning("matplotlib未安装，无法绘制性能曲线")
    except Exception as e:
        if logger:
            logger.error(f"绘制性能曲线时出错: {e}")


if __name__ == "__main__":
    main()