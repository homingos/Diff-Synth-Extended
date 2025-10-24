#!/bin/bash
# Script to run cached latent training pipeline

# Set paths
DATASET_BASE_PATH="data/rgba_videos"
DATASET_METADATA_PATH="data/rgba_videos/metadata.csv"
BASE_VAE_PATH="/workspace/Diff-Synth-Extended/weights/Wan2.1_VAE.pth"
VAE_LORA_PATH="/workspace/Diff-Synth-Extended/WanAlpha-VAE/decoder.bin"
CACHE_DIR="./cached_latents"
OUTPUT_PATH="./models/train/feature_merge_cached"

# Video specifications
HEIGHT=480
WIDTH=832
NUM_FRAMES=81

# Training parameters
BATCH_SIZE=4
NUM_EPOCHS=10
LEARNING_RATE=1e-4

# Step 1: Cache latents (if not already cached)
if [ ! -d "$CACHE_DIR" ]; then
    echo "========================================="
    echo "Step 1: Caching VAE latents..."
    echo "========================================="
    
    python examples/wanvideo/model_training/cache_latents.py \
        --dataset_base_path $DATASET_BASE_PATH \
        --dataset_metadata_path $DATASET_METADATA_PATH \
        --base_vae_path $BASE_VAE_PATH \
        --vae_lora_path $VAE_LORA_PATH \
        --output_dir $CACHE_DIR \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --batch_size 1 \
        --num_workers 4
    
    if [ $? -ne 0 ]; then
        echo "Error: Latent caching failed!"
        exit 1
    fi
else
    echo "Cached latents already exist at $CACHE_DIR"
    
    # Verify cache
    echo "Verifying cached latents..."
    python examples/wanvideo/model_training/cache_latents.py \
        --dataset_base_path $DATASET_BASE_PATH \
        --dataset_metadata_path $DATASET_METADATA_PATH \
        --base_vae_path $BASE_VAE_PATH \
        --vae_lora_path $VAE_LORA_PATH \
        --output_dir $CACHE_DIR \
        --verify_only
fi

echo ""
echo "========================================="
echo "Step 2: Training Feature Merge Block with cached latents..."
echo "========================================="

# Run training with accelerate
accelerate launch examples/wanvideo/model_training/train_feature_merge_cached.py \
    --base_vae_path $BASE_VAE_PATH \
    --vae_lora_path $VAE_LORA_PATH \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_workers 16 \
    --output_path $OUTPUT_PATH \
    --save_interval 1

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo "Cached latents: $CACHE_DIR"
echo "Trained model: $OUTPUT_PATH"
