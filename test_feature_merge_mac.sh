#!/bin/bash
# Quick test script for Mac M3 (MPS/CPU)
# Tests Feature Merge Block training with minimal resources

echo "🍎 Mac M3 Test - Feature Merge Block Training"
echo "=============================================="
echo ""
echo "⚠️  Note: This will use MPS (Apple Silicon GPU)"
echo "   Training will be slower than CUDA but should work!"
echo ""

# Use small settings for testing
python examples/wanvideo/model_training/train_feature_merge.py \
  --dataset_base_path data/rgba_videos \
  --dataset_metadata_path data/rgba_videos/metadata.csv \
  --base_vae_path weights/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --height 272 \
  --width 272 \
  --num_frames 17 \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --dataset_repeat 1 \
  --num_workers 0 \
  --output_path ./models/train/feature_merge_test

echo ""
echo "✅ Test completed!"
echo "Check: ./models/train/feature_merge_test/feature_merge_epoch_0.safetensors"

