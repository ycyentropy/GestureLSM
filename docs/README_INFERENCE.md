# GestureLSM 推理脚本使用指南

本文档介绍如何使用纯推理脚本和视频渲染脚本，无需Gradio界面。

## 脚本概述

### 1. `inference.py` - 纯推理脚本
- **功能**: 从音频文件生成手势动作数据
- **输入**: 音频文件 (.wav, .mp3等)
- **输出**: NPZ格式的SMPL-X运动数据文件
- **特点**: 无需Gradio，纯命令行操作

### 2. `render_video.py` - 视频渲染脚本
- **功能**: 将NPZ运动数据渲染为MP4视频
- **输入**: NPZ运动数据文件 + 音频文件
- **输出**: MP4视频文件
- **特点**: 支持简单和高级两种渲染模式

### 3. `run_example.py` - 完整流程脚本
- **功能**: 自动执行推理+渲染的完整流程
- **输入**: 音频文件
- **输出**: 最终的MP4视频文件

## 使用方法

### 方法1: 分步执行

#### 步骤1: 生成手势数据
```bash
python inference.py \
    --config configs/shortcut_rvqvae_128_hf.yaml \
    --audio /path/to/your/audio.wav \
    --output ./outputs/ \
    --checkpoint ./ckpt/new_540_shortcut.bin
```

#### 步骤2: 渲染视频
```bash
python render_video.py \
    --input /path/to/result.npz \
    --audio /path/to/your/audio.wav \
    --output ./output_video.mp4 \
    --config configs/shortcut_rvqvae_128_hf.yaml \
    --method simple
```

### 方法2: 一键执行
```bash
python run_example.py \
    --audio /path/to/your/audio.wav \
    --output ./example_output/ \
    --video generated_gesture.mp4
```

## 参数说明

### inference.py 参数
- `--config, -c`: 配置文件路径 (默认: configs/shortcut_rvqvae_128_hf.yaml)
- `--audio, -a`: **必需** 输入音频文件路径
- `--output, -o`: 输出目录 (默认: ./outputs/)
- `--checkpoint`: 模型检查点路径 (默认: ./ckpt/new_540_shortcut.bin)

### render_video.py 参数
- `--input, -i`: **必需** 输入NPZ运动数据文件路径
- `--audio, -a`: **必需** 音频文件路径
- `--output, -o`: 输出视频文件路径 (默认: output_video.mp4)
- `--config, -c`: 配置文件路径
- `--method, -m`: 渲染方法 (simple/advanced, 默认: simple)

### run_example.py 参数
- `--audio, -a`: **必需** 输入音频文件路径
- `--output, -o`: 输出目录 (默认: ./example_output/)
- `--video, -v`: 输出视频文件名 (默认: generated_gesture.mp4)

## 输出文件说明

### 推理输出结构
```
outputs/inference/timestamp_audio_name/
├── input.wav              # 处理后的音频文件
├── input.lab              # 语音识别文本
├── input.TextGrid         # MFA强制对齐数据
├── alignment_analysis.csv # 对齐质量报告
└── result/
    └── result_000.npz     # SMPL-X运动数据文件
```

### NPZ文件内容
生成的NPZ文件包含以下数据：
- `poses`: (N, 165) - 身体姿态参数
- `trans`: (N, 3) - 身体位移
- `expressions`: (N, 100) - 面部表情参数
- `betas`: (300,) - 人体形状参数
- `model`: 'smplx' - 模型类型
- `gender`: 'neutral' - 性别
- `mocap_frame_rate`: 30 - 帧率

## 示例

### 使用已有音频文件测试
```bash
# 假设有一个测试音频文件 test_audio.wav
python run_example.py --audio test_audio.wav --output ./test_output/
```

### 仅生成运动数据（不渲染视频）
```bash
python inference.py --audio test_audio.wav --output ./motion_only/
```

### 使用已有运动数据渲染视频
```bash
python render_video.py --input motion/result_000.npz --audio test_audio.wav --output final_video.mp4
```

## 依赖要求

确保已安装以下依赖：
- PyTorch
- transformers (for Whisper ASR)
- librosa, soundfile (audio processing)
- numpy, scipy
- trimesh, pyrender (for advanced rendering)
- FFmpeg (for video encoding)

## 注意事项

1. **音频格式**: 推荐使用WAV格式，其他格式可能需要转换
2. **音频时长**: 建议音频时长在10秒到5分钟之间
3. **模型检查点**: 确保检查点文件存在且路径正确
4. **MFA工具**: 首次运行需要安装Montreal Forced Aligner
5. **GPU内存**: 推理需要至少4GB GPU内存
6. **渲染时间**: 视频渲染可能需要几分钟时间

## 故障排除

### 常见问题
1. **MFA对齐失败**: 确保已安装MFA并正确配置
2. **模型加载失败**: 检查检查点文件路径
3. **GPU内存不足**: 减少batch_size或使用CPU
4. **FFmpeg错误**: 确保FFmpeg正确安装且有H.264编码器

### 日志查看
脚本运行时会输出详细日志，包括：
- 模型加载状态
- 音频处理进度
- 推理时间统计
- 错误信息（如有）

## 性能优化

- 使用GPU加速推理
- 批量处理多个音频文件
- 调整渲染分辨率以平衡质量和速度
- 使用简单渲染模式获得更快的速度