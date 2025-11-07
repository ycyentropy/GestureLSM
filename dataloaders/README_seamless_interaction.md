# Seamless_Interaction数据集解析器

这个解析器为Seamless_Interaction数据集提供了完整的加载和处理功能，支持多模态数据（文本、音频、视频和姿态）的加载、预处理和特征提取。

## 文件结构

- `seamless_interaction.py` - 主数据集加载器
- `seamless_interaction_features.py` - 特征提取工具
- `example_seamless_interaction.py` - 使用示例

## 数据集概述

Seamless_Interaction数据集是一个多模态交互数据集，包含11个交互片段，每个片段包含：
- 文本转录（JSON文件）
- 语音活动检测（VAD）信息
- 人体姿态参数（SMPL-H格式）
- 2D关键点
- 情绪分数
- 面部表情参数
- 音频数据
- 视频数据

## 使用方法

### 基本用法

```python
from dataloaders.seamless_interaction import SeamlessInteractionDataset

# 创建数据集实例
dataset = SeamlessInteractionDataset(
    data_path="/path/to/seamless_interaction",
    split="train",
    sample_rate=16000,
    pose_fps=30,
    audio_fps=16000,
    window_size=2.0,
    window_stride=0.5,
    load_video=False,
    load_audio=True,
    normalize=True
)

# 加载一个样本
sample = dataset[0]
```

### 时间窗口数据集

```python
from dataloaders.seamless_interaction import SeamlessInteractionWindowDataset

# 创建时间窗口数据集实例
window_dataset = SeamlessInteractionWindowDataset(
    data_path="/path/to/seamless_interaction",
    split="train",
    sample_rate=16000,
    pose_fps=30,
    audio_fps=16000,
    window_size=2.0,
    window_stride=0.5,
    load_video=False,
    load_audio=True,
    normalize=True
)

# 加载一个时间窗口样本
window_sample = window_dataset[0]
```

### 特征提取

```python
from dataloaders.seamless_interaction_features import (
    extract_text_features, extract_vad_features, extract_pose_features,
    extract_keypoint_features, extract_emotion_features, extract_expression_features,
    align_audio_to_motion, extract_audio_features
)

# 提取文本特征
text_features = extract_text_features(transcript, max_words=50)

# 提取VAD特征
vad_features = extract_vad_features(vad, sample_rate=30.0, max_frames=120)

# 提取姿态特征
pose_features = extract_pose_features(pose, joint_mask=joint_mask, normalize=True)

# 提取关键点特征
keypoint_features = extract_keypoint_features(keypoints, normalize=True, confidence_threshold=0.5)

# 提取情绪特征
emotion_features = extract_emotion_features(emotion_scores, normalize=True)

# 提取表情特征
expression_features = extract_expression_features(expression, normalize=True)

# 对齐音频和运动数据
aligned_audio = align_audio_to_motion(
    audio, motion_length=len(pose), audio_fps=16000, motion_fps=30
)

# 提取音频特征
audio_features = extract_audio_features(aligned_audio, sample_rate=16000, n_mels=80)
```

### 数据加载器

```python
from torch.utils.data import DataLoader

# 创建数据加载器
dataloader = DataLoader(
    window_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# 加载一个批次的数据
batch = next(iter(dataloader))
```

## 数据格式

### 输入数据

- **JSON文件**：包含文本转录（`metadata:transcript`）和语音活动检测（`metadata:vad`）信息
- **NPZ文件**：包含多模态数据，分为三大类：
  - `boxes_and_keypoints`：边界框和2D关键点
  - `movement`：运动/情绪特征
  - `smplh`：人体姿态参数
- **WAV文件**：音频数据
- **MP4文件**：视频数据

### 输出数据

每个样本包含以下字段：

- `id`：样本ID
- `transcript`：文本转录列表，每个元素包含单词、开始时间、结束时间和置信度
- `vad`：语音活动检测列表，每个元素包含开始时间和结束时间
- `pose`：人体姿态参数，形状为`(T, 156)`，其中T是时间步
- `keypoints`：2D关键点，形状为`(T, 133, 3)`
- `emotion_scores`：情绪分数，形状为`(T, 8)`
- `expression`：面部表情参数，形状为`(T, 128)`
- `audio`：音频数据，形状为`(N,)`，其中N是采样点数
- `audio_sample_rate`：音频采样率

## 特征提取

### 文本特征

- `word_ids`：单词ID，形状为`(max_words,)`
- `word_times`：单词时间戳，形状为`(max_words, 2)`
- `word_scores`：单词置信度，形状为`(max_words,)`
- `transcripts`：单词列表

### VAD特征

- 语音活动检测特征，形状为`(max_frames,)`

### 姿态特征

- 人体姿态特征，形状为`(T, num_joints * 3)`，其中`num_joints`是关节数量

### 关键点特征

- 2D关键点特征，形状为`(T, num_keypoints * 2)`，其中`num_keypoints`是关键点数量

### 情绪特征

- 情绪分数特征，形状为`(T, 8)`

### 表情特征

- 面部表情特征，形状为`(T, 128)`

### 音频特征

- 音频梅尔频谱图特征，形状为`(n_mels, T')`，其中`T'`是音频帧数

## 示例输出

运行示例脚本后，您将看到类似以下输出：

```
数据集大小: 11 个样本
窗口数据集大小: 220 个窗口

样本信息:
  id: V00_S0039_I00000581_P0061
  json_path: /path/to/V00_S0039_I00000581_P0061.json
  npz_path: /path/to/V00_S0039_I00000581_P0061.npz
  wav_path: /path/to/V00_S0039_I00000581_P0061.wav
  video_path: /path/to/V00_S0039_I00000581_P0061.mp4
  pose_fps: 30
  audio_fps: 16000
  pose_length: 3000
  audio_length: 1600000
  duration: 100.0

样本数据:
  id: V00_S0039_I00000581_P0061
  transcript: 列表，长度 10
  vad: 列表，长度 23
  pose: torch.Size([3000, 156]) torch.float32
  keypoints: torch.Size([3000, 133, 3]) torch.float32
  emotion_scores: torch.Size([3000, 8]) torch.float32
  expression: torch.Size([3000, 128]) torch.float32
  audio: torch.Size([1600000]) torch.float32
  audio_sample_rate: <class 'int'>

文本特征:
  word_ids: torch.Size([50]) torch.int64
  word_times: torch.Size([50, 2]) torch.float32
  word_scores: torch.Size([50]) torch.float32
  transcripts: <class 'list'>

VAD特征: torch.Size([120]) torch.float32

姿态特征: torch.Size([3000, 156]) torch.float32

关键点特征: torch.Size([3000, 266]) torch.float32

情绪特征: torch.Size([3000, 8]) torch.float32

表情特征: torch.Size([3000, 128]) torch.float32

对齐音频和运动数据...
原始音频: torch.Size([1600000])
对齐后音频: torch.Size([1600000])
音频特征: torch.Size([80, 6251]) torch.float32
```

## 注意事项

1. 数据集路径需要正确设置，指向包含`improvised/train/0000/0000`目录的父目录
2. 视频数据加载默认关闭，可以通过设置`load_video=True`启用
3. 时间窗口大小和步长可以根据需要调整
4. 关节掩码可以根据需要自定义，选择特定的关节进行处理
5. 数据归一化默认启用，可以通过设置`normalize=False`禁用

## 扩展功能

1. 可以添加更多的特征提取方法，如MFCC、频谱图等
2. 可以添加数据增强功能，如随机裁剪、旋转等
3. 可以添加更多的数据可视化功能
4. 可以添加更多的数据预处理功能，如滤波、降噪等