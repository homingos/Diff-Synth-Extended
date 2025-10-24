#!/bin/bash
# Wan-Alpha DiT DoRA Training Script (Stage 2)
# This trains the DiT with DoRA using trained FeatureMergeBlock

# Prerequisites:
# 1. Trained FeatureMergeBlock from Stage 1
# 2. RGBA video dataset with metadata.csv
# 3. Base Wan2.1 models
# 4. decoder.bin (VAE LoRAs)

accelerate launch examples/wanvideo/model_training/train.py \
  --rgba_mode \
  --dataset_base_path data/rgba_videos \
  --dataset_metadata_path data/rgba_videos/metadata.csv \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --vae_lora_path Wan-Alpha/vae/decoder.bin \
  --feature_merge_checkpoint models/train/feature_merge/feature_merge_epoch_9.safetensors \
  --use_dora \
  --dora_rank 32 \
  --learning_rate 1e-4 \
  --num_epochs 15 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan-Alpha-DiT-DoRA" \
  --trainable_models "dit"

