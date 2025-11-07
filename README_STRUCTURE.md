# GestureLSM 项目结构说明

## 目录结构

### 核心代码
- `models/` - 模型相关代码
- `dataloaders/` - 数据加载器
- `datasets/` - 数据集
- `utils/` - 工具函数
- `optimizers/` - 优化器
- `configs/` - 配置文件

### 脚本文件
- `scripts/` - 所有运行脚本 (.sh文件)
  - 训练脚本
  - 测试脚本
  - 多GPU运行脚本

### 测试文件
- `tests/` - 所有测试文件 (test_*.py)
  - 单元测试
  - 集成测试
  - 性能测试

### 工具文件
- `tools/` - 工具类和调试文件
  - 调试工具 (debug_*.py)
  - 监控工具 (monitor_*.py)
  - 计算工具 (calculate_*.py, compute_*.py)
  - 诊断工具 (diagnose_*.py)
  - 性能分析工具 (performance_*.py)
  - 其他工具脚本

### 文档
- `docs/` - 所有文档和README文件
  - 项目说明文档
  - 使用指南
  - 技术文档

### 日志
- `logs/` - 日志文件
  - 分布式调试日志
  - 多GPU错误日志

### 数据和输出
- `data/` - 数据文件
- `ckpt/` - 模型检查点
- `output/`, `outputs/`, `custom_output/` - 输出文件
- `results/` - 结果文件

### 演示和示例
- `demo/` - 演示相关文件
- `demo_scripts/` - 演示脚本
- `static/` - 静态资源

### 其他
- `experiments/` - 实验相关文件
- `preprocess/` - 预处理脚本
- `visualizations/` - 可视化相关

## 主要运行文件

根目录下保留的主要Python文件：
- `train.py` - 主训练脚本
- `demo.py` - 演示脚本
- `gesture_inference.py` - 推理脚本
- `inference_cli.py` - 命令行推理脚本
- `get_args_parser.py` - 参数解析
- `requirements.txt` - 依赖包列表

## 使用说明

1. 运行训练：使用 `scripts/` 目录下的相应脚本
2. 运行测试：使用 `tests/` 目录下的测试文件
3. 使用工具：查看 `tools/` 目录下的工具脚本
4. 查看文档：参考 `docs/` 目录下的文档

整理日期：$(date)