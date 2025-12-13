# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference

请用中文回答 - Please respond in Chinese when interacting with this repository.

## Overview

GestureLSM is a co-speech gesture generation system that uses latent shortcut-based modeling with spatial-temporal modeling. The project implements RVQ-VAEs for motion quantization and offers both shortcut and diffusion-based generation approaches.

## Installation and Setup

```bash
# Create conda environment
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh

# Download pretrained models (1-speaker and all-speaker)
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw?usp=drive_link -O ./ckpt --folder
# or
huggingface-cli download https://huggingface.co/pliu23/GestureLSM --local-dir ./ckpt

# Download SMPL model
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

## Common Commands

### Training
```bash
# Train RVQ-VAEs (1-speaker)
bash train_rvq.sh

# Train shortcut generator
python train.py -c configs/shortcut_rvqvae_128.yaml

# Train diffusion generator
python train.py -c configs/diffuser_rvqvae_128.yaml
```

### Evaluation and Testing
```bash
# Evaluate shortcut model (20 steps)
python test.py -c configs/shortcut_rvqvae_128.yaml

# Evaluate shortcut-reflow model (2-step)
python test.py -c configs/shortcut_reflow_test.yaml

# Evaluate diffusion model
python test.py -c configs/diffuser_rvqvae_128.yaml
```

### Demo
```bash
# Run web demo
python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```

## Architecture Overview

### Core Components

1. **RVQ-VAE Models**: Residual Vector Quantization VAEs for motion compression
   - Separate models for upper body, hands, lower body, and translation
   - Located in `models/vq/` directory

2. **Generation Models**:
   - **Shortcut Model**: Fast generation approach in `models/LSM.py`
   - **Diffusion Model**: Diffusion-based approach in `models/GestureDiffuse.py`

3. **Data Loaders**: Dataset handling for BEAT dataset
   - Main dataloaders: `beat_sep.py`, `beat_sep_lower.py`, `beat_sep_single.py`
   - Located in `dataloaders/` directory

4. **Feature Extractors**:
   - Audio processing: `dataloaders/utils/audio_features.py`
   - Text processing: `dataloaders/utils/text_features.py`
   - Motion processing: `dataloaders/utils/motion_rep_transfer.py`

### Key Directories

- `models/`: Core model implementations including VAEs, generators, and encoders
- `dataloaders/`: Dataset classes and feature extraction utilities
- `optimizers/`: Custom optimizers and learning rate schedulers
- `utils/`: Utility functions for metrics, data processing, and visualization
- `configs/`: YAML configuration files for different model variants
- `scripts/`: Analysis and visualization tools

### Motion Representation

The system uses SMPL-X body representation with:
- 330 dimensions for full body pose
- Separate processing for upper body, hands, and lower body
- 30 FPS output frame rate
- Support for both rotation matrices and 6D rotation representations

### Training Configuration

Training uses a distributed setup with:
- Configurable speakers (single speaker: [2], all speakers: [1-30])
- Batch size: 128
- Learning rate: 2e-4
- Mixed precision training support
- WandB and TensorBoard logging

## Model Architecture

### RVQ-VAE Structure
- **Encoder**: Convolutional architectures (VQEncoderV3, VQEncoderV5) in `models/motion_encoder.py`
- **Decoder**: Corresponding decoders with symmetric architectures
- **Quantization**: Residual Vector Quantization with configurable codebook sizes (typically 256)
- **Body Partitioning**: Separate VAEs for upper body, hands, lower body, and translation

### Generation Models
- **Shortcut Model**: Implements rectified flow with ODE solver for fast sampling
- **Diffusion Model**: Traditional diffusion process with denoising transformer
- **Reflow Model**: 2-step rectified flow for ultra-fast inference

### Motion Representation System
- Uses SMPL-X body model with 330-dimensional pose vectors
- Joint masking system for different body regions (upper_body_mask, hands_body_mask, lower_body_mask)
- 6D rotation representation for more stable training
- Motion sequence length: 128 frames at 30 FPS

## Configuration Files

Configuration files are in YAML format and control:
- Model architecture parameters (VAE layers, codebook sizes, latent dimensions)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths and speaker IDs ([2] for single speaker, [1-30] for all speakers)
- Feature extraction settings (audio sampling rates, text embeddings)
- Evaluation parameters (test lengths, inference steps)

Key configs:
- `shortcut_rvqvae_128.yaml`: Main shortcut model configuration
- `diffuser_rvqvae_128.yaml`: Diffusion model configuration
- `shortcut_reflow_test.yaml`: 2-step reflow model for fast inference
- `sc_model_config.yaml`: Shared model architecture configuration

## Data Processing Pipeline

1. **Audio Features**:
   - Raw audio at 16kHz
   - Onset detection and amplitude extraction
   - Optional Wav2Vec2 features for enhanced representation

2. **Text Processing**:
   - FastText embeddings (300 dimensions)
   - TextGrid alignment for word-level timing
   - Vocabulary building for semantic features

3. **Motion Data**:
   - SMPL-X pose parameters with joint masking
   - Normalization using pre-computed mean/std statistics
   - Sliding window processing with 20-frame stride
   - Body region separation for specialized VAEs

4. **Training Pipeline**:
   - Distributed data loading with LMDB caching
   - Dynamic batching and sequence padding
   - Mixed precision training support