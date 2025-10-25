#!/bin/bash

# Multi-GPU training script for Feature Merge Block

echo "=========================================
Multi-GPU Training for Feature Merge Block
========================================="

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your available GPUs

# Check available GPUs
echo "Available GPUs:"
python -c "import torch; print(f'Found {torch.cuda.device_count()} GPUs')"

# Clear GPU memory on all devices
python -c "
import torch
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()
print('Cleared GPU memory on all devices')
"

# Run multi-GPU training
# Note: batch_size is per GPU, so total batch = batch_size * num_gpus
accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes $(python -c "import torch; print(torch.cuda.device_count())") \
    examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth \
    --vae_lora_path /workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --num_workers 2 \
    --output_path ./models/train/feature_merge_cached \
    --save_interval 1

echo "=========================================
Multi-GPU training complete!
========================================="
