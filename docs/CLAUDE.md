# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

```bash
# Create environment
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh

# Download pretrained models (Google Drive)
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw?usp=drive_link -O ./ckpt --folder

# Download SMPL model
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

## Core Training and Evaluation Commands

### RVQ-VAE Training (Required prerequisite)
```bash
# Train all body parts RVQ-VAEs sequentially
bash train_rvq.sh
```

### Model Training
```bash
# Train Shortcut model (main approach)
python train.py -c configs/shortcut_rvqvae_128.yaml

# Train Diffusion model (baseline)
python train.py -c configs/diffuser_rvqvae_128.yaml
```

### Evaluation
```bash
# Evaluate Shortcut model (20-step generation)
python test.py -c configs/shortcut_rvqvae_128.yaml

# Evaluate Shortcut-Reflow model (2-step generation)
python test.py -c configs/shortcut_reflow_test.yaml

# Evaluate Diffusion model
python test.py -c configs/diffuser_rvqvae_128.yaml
```

### Demo
```bash
# Run web demo (Gradio interface)
python demo.py -c configs/shortcut_rvqvqe_128_hf.yaml
```

### Pure Inference Scripts (No Gradio Required)

#### Command-Line Gesture Generation
```bash
# Step 1: Generate motion data from audio
python gesture_inference.py \
    --config configs/shortcut_rvqvae_128_hf.yaml \
    --audio /path/to/your/audio.wav \
    --output ./outputs/

# Step 2: Render video from motion data
python render_video.py \
    --input /path/to/result.npz \
    --audio /path/to/your/audio.wav \
    --output ./final_video.mp4 \
    --method simple
```

#### One-Click Pipeline
```bash
# Complete pipeline from audio to video
python run_example.py \
    --audio /path/to/your/audio.wav \
    --output ./pipeline_output/ \
    --video gesture_video.mp4
```

#### Advanced Usage
```bash
# Generate motion data with custom checkpoint
python gesture_inference.py \
    --config configs/shortcut_rvqvae_128_hf.yaml \
    --audio input_audio.wav \
    --output ./custom_output/ \
    --checkpoint ./ckpt/new_540_shortcut.bin

# Render video with different method
python render_video.py \
    --input motion_data.npz \
    --audio source_audio.wav \
    --output high_quality_video.mp4 \
    --method advanced
```

## Architecture Overview

### Model Components
- **RVQ-VAE**: Residual Vector Quantized VAE for motion compression (separate models for upper/lower/hands/face)
- **Latent Shortcut Model**: Main generation model with specialized time sampling (`models/LSM.py`)
- **GestureDiffusion**: Diffusion-based baseline (`models/GestureDiffuse.py`)
- **ModalityEncoder**: Multi-modal encoder processing audio, text, and facial features (`models/modality_encoder.py`)

### Data Flow
1. **Input Processing**: Audio → WavLM features, Text → FastText embeddings, Facial expressions → SMPLX parameters
2. **RVQ Encoding**: Motion sequences → Quantized latent representations
3. **Generation**: Latent shortcuts/diffusion → Motion latent codes
4. **Decoding**: Latent codes → RVQ-VAE decoder → SMPLX pose parameters

### Key Architectural Patterns
- **Configuration-driven**: All models instantiated via `instantiate_from_config()` using YAML configs
- **Multi-body-part training**: Separate RVQ-VAEs for upper_body, lower_body, hands, face, lower_trans
- **Time sampling strategies**: Exponential and Beta distributions for shortcut generation (`models/LSM.py:31-89`)
- **Hierarchical training**: RVQ-VAEs must be trained before generative models

### Configuration System
- **Main configs**: `configs/shortcut_rvqvae_128.yaml`, `configs/diffuser_rvqvae_128.yaml`
- **Model configs**: `configs/sc_model_config.yaml` (defines LSM architecture)
- **Speaker settings**: Modify `training_speakers: [2]` for single-speaker or expand for multi-speaker

### Training Pipeline
1. **RVQ-VAE Training**: Motion compression using `rvq_beatx_train.py` (4 separate models)
2. **Generator Training**: Main training using `train.py` with custom trainers
   - `shortcut_rvqvae_trainer.py`: Latent Shortcut training
   - `diffuser_rvqvae_trainer.py`: Diffusion training

### Critical Implementation Details
- **Motion representation**: SMPLX format with rotation matrices (330 dimensions)
- **Frame rates**: 30 FPS motion, 16kHz audio processed to 30 FPS features
- **Sequence length**: 128 frames with stride 20 for training
- **Data splits**: BEAT dataset with speaker ID 2 ('scott') for single-speaker evaluation

## Common Development Patterns

### Adding New Models
1. Create model class in `models/`
2. Add corresponding YAML config in `configs/`
3. Create trainer in `train.py` or separate trainer file
4. Update `models/config.py` if new instantiation patterns needed

### Working with Data
- Motion data stored as SMPLX parameters in BEAT dataset format
- Audio features extracted using WavLM or custom encoders
- Use `dataloaders/data_tools.py` for SMPLX joint transformations
- Cache preprocessing in `datasets/beat_cache/`

### Debugging Training
- Check RVQ-VAE reconstruction quality first
- Verify latent dimensions match between encoder and decoder
- Monitor classifier-free guidance scale for conditional generation
- Use WandB/TensorBoard logging via `stat: ts` or `stat: wandb` configs

## New Script Files

### `gesture_inference.py` - Pure Inference Script
**Purpose**: Command-line gesture generation without Gradio interface

**Key Features**:
- Audio processing with Whisper ASR
- MFA forced alignment for precise timing
- Generates NPZ SMPL-X motion data files
- No web interface required
- Suitable for batch processing

**Output Structure**:
```
outputs/inference/timestamp_audio_name/
├── input.wav              # Processed audio
├── input.lab              # ASR transcript
├── input.TextGrid         # MFA alignment data
├── alignment_analysis.csv # Quality metrics
└── result/
    └── result_000.npz     # SMPL-X motion parameters
```

### `render_video.py` - Video Rendering Script
**Purpose**: Convert NPZ motion data to MP4 videos

**Key Features**:
- Two rendering modes: simple (existing pipeline) and advanced (custom)
- Audio synchronization
- Multiple video resolutions
- FFmpeg with libopenh264 encoding
- GPU-accelerated rendering

**Rendering Methods**:
- **Simple**: Uses project's existing rendering pipeline
- **Advanced**: Custom PyRender-based renderer (requires additional dependencies)

### `run_example.py` - Complete Pipeline Script
**Purpose**: Automated end-to-end gesture generation

**Key Features**:
- One-click execution from audio to video
- Detailed progress reporting
- Error handling and validation
- File organization

### `README_INFERENCE.md` - Detailed Documentation
Comprehensive guide covering:
- Installation requirements
- Script usage examples
- Output file formats
- Troubleshooting guide
- Performance optimization tips

## Script Testing Status

### ✅ Tested and Working
- **Video rendering**: Successfully generates MP4 videos from NPZ files
- **FFmpeg integration**: libopenh264 encoding works correctly
- **Audio synchronization**: Original audio properly synced with generated motion
- **Output quality**: 960×720, 30fps, high-quality output

### ⚠️ Partial Issues (Known)
- **Inference script**: Works through audio processing and MFA alignment, model initialization has module import conflicts (resolvable with script renaming)

### 🚀 Recommended Usage Patterns

#### For Development/Testing
```bash
# Use existing generated data for testing rendering
python render_video.py \
    --input outputs/audio2pose/custom/*/999/result_*.npz \
    --audio outputs/audio2pose/custom/*/tmp.wav \
    --output test_video.mp4
```

#### For Production
```bash
# Batch process multiple audio files
for audio in audio/*.wav; do
    python gesture_inference.py --audio "$audio" --output ./batch_output/
    python render_video.py --input ./batch_output/*/result/result_000.npz --audio "$audio" --output "videos/${audio%.wav}.mp4"
done
```

```bash
python inference_cli.py --audio_path demo/examples/2_scott_0_2_2.wav --config configs/shortcut_rvqvae_128_hf.yaml --output_dir ./outputs --output_name test_output --render_video --save_npz
```