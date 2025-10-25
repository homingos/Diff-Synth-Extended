#!/bin/bash

# Optimized training script for H200 GPU with 140GB memory

echo "=========================================
H200 Optimized Training for Feature Merge Block
========================================="

# Set environment variables for H200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0  # Disable for better performance on H200

# Clear GPU memory
python -c "
import torch
torch.cuda.empty_cache()
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'Currently allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
"

echo -e "\nStarting training with optimized settings for H200..."

# With 140GB memory, we can use larger batch sizes
# Let's start conservatively and increase if it works
accelerate launch \
    --mixed_precision bf16 \
    examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth \
    --vae_lora_path /workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --output_path ./models/train/feature_merge_cached \
    --save_interval 1

echo "=========================================
Training complete!
========================================="
