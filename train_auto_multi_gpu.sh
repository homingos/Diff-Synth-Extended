#!/bin/bash

# Auto-detect and use all available GPUs for training

echo "=========================================
Auto Multi-GPU Training for Feature Merge Block
========================================="

# Detect number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs found!"
    exit 1
fi

echo "Found $NUM_GPUS GPUs"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the GPU check script
echo -e "\nGPU Information:"
python /workspace/Diff-Synth-Extended/check_gpus.py

echo -e "\nStarting training on $NUM_GPUS GPUs..."

# Calculate total effective batch size
BATCH_PER_GPU=1
GRAD_ACCUM=1
TOTAL_BATCH=$((NUM_GPUS * BATCH_PER_GPU * GRAD_ACCUM))
echo "Batch size per GPU: $BATCH_PER_GPU"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "Total effective batch size: $TOTAL_BATCH"

# Run multi-GPU training
accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path /workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth \
    --vae_lora_path /workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin \
    --cache_dir ./cached_latents \
    --batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --num_workers 2 \
    --output_path ./models/train/feature_merge_cached \
    --save_interval 1

echo "=========================================
Multi-GPU training complete!
========================================="
