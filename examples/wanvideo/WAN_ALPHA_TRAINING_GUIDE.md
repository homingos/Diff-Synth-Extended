# Wan-Alpha Training Guide for DiffSynth-Studio

Complete guide to train Wan-Alpha RGBA video generation models.

## 📋 Overview

Wan-Alpha training consists of **TWO stages**:

1. **Stage 1**: Train FeatureMergeBlock (lightweight, ~10-20M params)
2. **Stage 2**: Train DiT with DoRA (using trained FeatureMergeBlock)

## 🔧 Prerequisites

### Required Files

- ✅ `Wan-Alpha/vae/decoder.bin` - Pre-trained VAE decoder LoRAs
- ✅ Base Wan2.1 model (auto-downloads)
- ✅ Your RGBA dataset (RGB + alpha video pairs)

### Dataset Format

Your dataset should have:

```
dataset_30FPS/
├── 001.mp4        # RGB video
├── 001_seg.mp4    # Alpha/segmentation video
├── 001.txt        # Caption
├── 002.mp4
├── 002_seg.mp4
├── 002.txt
└── ...
```

## 🚀 Complete Training Pipeline

### Step 1: Prepare Dataset

```bash
python prepare_rgba_dataset.py \
  --source_dir dataset_30FPS \
  --output_dir data/rgba_videos
```

This creates:

```
data/rgba_videos/
├── metadata.csv
├── 001_rgb.mp4
├── 001_alpha.mp4
├── 002_rgb.mp4
├── 002_alpha.mp4
└── ...
```

### Step 2: Train FeatureMergeBlock (Stage 1)

```bash
accelerate launch examples/wanvideo/model_training/train_feature_merge.py \
  --dataset_base_path data/rgba_videos \
  --dataset_metadata_path data/rgba_videos/metadata.csv \
  --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --height 480 --width 832 --num_frames 81 \
  --learning_rate 1e-4 --num_epochs 10 \
  --output_path ./models/train/feature_merge
```

**What it trains**:

- 🔥 FeatureMergeBlock only (~10-20M parameters)
- 🔒 Frozen: VAE encoder
- 🔒 Frozen: Dual decoders (from decoder.bin)

**Output**: `models/train/feature_merge/feature_merge_epoch_9.safetensors`

### Step 3: Validate FeatureMergeBlock

```bash
python examples/wanvideo/model_training/validate_feature_merge.py \
  --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --feature_merge_checkpoint models/train/feature_merge/feature_merge_epoch_9.safetensors \
  --test_rgb_video data/rgba_videos/001_rgb.mp4 \
  --test_alpha_video data/rgba_videos/001_alpha.mp4 \
  --output_prefix validation
```

**Outputs**:

- `validation_gt_rgb.mp4` - Original RGB
- `validation_gt_alpha.mp4` - Original alpha
- `validation_pred_rgb.mp4` - Reconstructed RGB
- `validation_pred_alpha.mp4` - Reconstructed alpha
- Reconstruction error metrics

### Step 4: Train DiT with DoRA (Stage 2)

```bash
bash examples/wanvideo/model_training/train_dit_dora_rgba.sh
```

Or manually:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --rgba_mode \
  --dataset_base_path data/rgba_videos \
  --dataset_metadata_path data/rgba_videos/metadata.csv \
  --data_file_keys "video" \
  --height 480 --width 832 --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --feature_merge_checkpoint models/train/feature_merge/feature_merge_epoch_9.safetensors \
  --use_dora \
  --dora_rank 32 \
  --learning_rate 1e-4 --num_epochs 15 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan-Alpha-DiT-DoRA" \
  --trainable_models "dit"
```

**What it trains**:

- 🔥 DiT with DoRA
- 🔒 Frozen: VAE (encoder + decoder.bin)
- 🔒 Frozen: Trained FeatureMergeBlock

**Output**: `models/train/Wan-Alpha-DiT-DoRA/epoch-14.safetensors`

### Step 5: Generate RGBA Videos (Inference)

```bash
python examples/wanvideo/model_inference/Wan-Alpha-RGBA-Inference.py
```

**Outputs**:

- `output_rgb.mp4` - RGB video only
- `output_alpha.mp4` - Alpha channel video
- `output_rgba_frames/` - PNG frames with alpha
- `output_composite.mp4` - RGB on checkerboard background

## 📊 Training Progress Tracking

### Stage 1: FeatureMergeBlock

- **Epochs**: 10-20 epochs
- **Time**: ~1-2 hours (depending on dataset size)
- **VRAM**: ~24GB (single GPU)
- **Monitor**: Reconstruction losses (L_α, L_rgb^s, L_rgb^h)

### Stage 2: DiT DoRA

- **Epochs**: 10-15 epochs
- **Time**: Several hours to days
- **VRAM**: 24-80GB (depends on model size)
- **Monitor**: Diffusion training loss

## 🔍 Validation & Testing

### Quick Test (2 videos)

```bash
# Create small test dataset
python prepare_rgba_dataset.py \
  --source_dir dataset_30FPS \
  --output_dir data/rgba_test

# Edit metadata.csv to keep only first 2 rows

# Run quick Feature Merge training (1 epoch)
accelerate launch examples/wanvideo/model_training/train_feature_merge.py \
  --dataset_base_path data/rgba_test \
  --dataset_metadata_path data/rgba_test/metadata.csv \
  --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --height 480 --width 832 --num_frames 17 \
  --learning_rate 1e-4 --num_epochs 1 \
  --output_path ./models/train/feature_merge_test
```

## 📁 Directory Structure

```
DiffSynth-Studio/
├── data/
│   └── rgba_videos/
│       ├── metadata.csv
│       ├── 001_rgb.mp4
│       ├── 001_alpha.mp4
│       └── ...
├── Wan-Alpha/
│   └── vae/
│       └── decoder.bin
├── models/
│   ├── train/
│   │   ├── feature_merge/          # Stage 1 outputs
│   │   │   └── feature_merge_epoch_9.safetensors
│   │   └── Wan-Alpha-DiT-DoRA/     # Stage 2 outputs
│   │       └── epoch-14.safetensors
│   └── Wan-AI/                     # Auto-downloaded
│       └── Wan2.1-T2V-14B/
│           ├── Wan2.1_VAE.pth
│           ├── diffusion_pytorch_model*.safetensors
│           └── models_t5_umt5-xxl-enc-bf16.pth
└── examples/
    └── wanvideo/
        ├── model_training/
        │   ├── train_feature_merge.py      # Stage 1
        │   ├── train.py                    # Stage 2
        │   ├── train_dit_dora_rgba.sh      # Stage 2 script
        │   └── validate_feature_merge.py   # Validation
        └── model_inference/
            └── Wan-Alpha-RGBA-Inference.py # Inference
```

## 🎯 Key Implementation Details

### FeatureMergeBlock Architecture

- Input: RGB latent (16-dim) + Alpha latent (16-dim)
- Concat: 32-dim
- Causal 3D Conv: 32-dim → 192-dim
- 3× (Residual Block + Attention)
- Causal 3D Conv: 192-dim → 16-dim
- Output: Merged latent (16-dim)

### Dual Decoder System

- `model_fgr`: RGB decoder with vae_fgr LoRA (from decoder.bin)
- `model_pha`: Alpha decoder with vae_pha LoRA (from decoder.bin)
- Both decode the same merged latent Z

### DoRA vs LoRA

- DoRA uses weight decomposition: W' = (m / ||W + ΔW||) × (W + ΔW)
- Better semantic alignment and quality than LoRA
- Slightly more memory but worth it

## 💡 Tips & Tricks

### Reducing VRAM

- Use `--use_gradient_checkpointing_offload` for Stage 2
- Reduce `--num_frames` to 49 or 33 for testing
- Use smaller resolution (e.g., 480×480)

### Speeding Up Training with Cached Latents

For faster training, pre-cache VAE latents to avoid repeated encoding:

#### Step 1: Cache Latents

```bash
python examples/wanvideo/model_training/cache_latents.py \
    --dataset_base_path data/rgba_videos \
    --dataset_metadata_path data/rgba_videos/metadata.csv \
    --base_vae_path /path/to/Wan2.1_VAE.pth \
    --vae_lora_path /path/to/decoder.bin \
    --output_dir ./cached_latents \
    --height 480 --width 832 --num_frames 81
```

#### Step 2: Train with Cached Latents

```bash
accelerate launch examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /path/to/Wan2.1_VAE.pth \
    --vae_lora_path /path/to/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size 4 --num_epochs 10 \
    --output_path ./models/train/feature_merge_cached
```

Or use the convenience script:

```bash
bash examples/wanvideo/model_training/run_cached_training.sh
```

Benefits:

- **5-10x faster training** (no VAE encoding per epoch)
- **Lower GPU memory** (no encoder in memory)
- **Consistent latents** across epochs

### Improving Quality

- Train FeatureMergeBlock for more epochs (15-20)
- Use larger dataset
- Increase DoRA rank to 64 or 128
- Fine-tune learning rates

### Debugging

- Check reconstruction error in validation (should be < 20 MAE)
- Monitor loss curves (should decrease steadily)
- Verify RGBA outputs have proper alpha channel

## ⚠️ Common Issues

### Issue: "Base VAE not found"

**Solution**: The model will auto-download from ModelScope. Ensure internet connection.

### Issue: "CUDA out of memory"

**Solution**:

- Reduce `--num_frames`
- Add `--use_gradient_checkpointing_offload`
- Use smaller batch size

### Issue: "Reconstruction error too high"

**Solution**:

- Train FeatureMergeBlock for more epochs
- Check if decoder.bin loaded correctly
- Verify dataset quality

## 📧 Support

If you encounter issues, check:

1. All imports work: `python -c "from diffsynth.models.wan_video_vae import RGBAlphaVAE"`
2. decoder.bin loads: Verify with validation script
3. Dataset format: metadata.csv has correct columns

## 🎓 Citation

If you use this implementation, please cite the Wan-Alpha paper:

```bibtex
@misc{dong2025wanalpha,
  title={Wan-Alpha: High-Quality Text-to-Video Generation with Alpha Channel},
  author={Haotian Dong and Wenjing Wang and Chen Li and Di Lin},
  year={2025},
  eprint={2509.24979},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
