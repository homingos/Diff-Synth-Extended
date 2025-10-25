#!/bin/bash

# Training script for Feature Merge Block with extreme memory optimizations

echo "=========================================
Training Feature Merge Block with extreme memory optimizations...
========================================="

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1  # Helps with memory fragmentation

# Clear any existing GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run training with extreme memory optimizations
accelerate launch --mixed_precision bf16 \
    examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth \
    --vae_lora_path /workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --num_workers 0 \
    --output_path ./models/train/feature_merge_cached \
    --save_interval 1

echo "=========================================
Training complete!
========================================="
