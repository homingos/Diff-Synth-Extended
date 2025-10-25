#!/bin/bash

# Training script for Feature Merge Block with minimal memory usage

echo "=========================================
Training Feature Merge Block with minimal memory usage...
========================================="

# Set environment variables for aggressive memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1  # Reduce CPU memory overhead

# Clear any existing GPU memory
python -c "import torch; torch.cuda.empty_cache(); print(f'GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated')"

# Run training with minimal memory settings
accelerate launch --mixed_precision bf16 \
    examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth \
    --vae_lora_path /workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --num_workers 0 \
    --output_path ./models/train/feature_merge_cached \
    --save_interval 1

echo "=========================================
Training complete!
========================================="
