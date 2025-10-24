"""
Feature Merge Block training using cached latents.
This is much faster than encoding on the fly during training.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
import wandb
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from diffsynth.models.wan_video_vae import RGBAlphaVAE, FeatureMergeBlock
from diffsynth.trainers.rgba_losses import RGBAReconstructionLoss
from diffsynth.trainers.cached_latent_dataset import CachedLatentDataset, CachedLatentDataLoader


class CachedFeatureMergeTrainer(nn.Module):
    """
    Feature Merge Block trainer using cached latents.
    Much faster than encoding videos on the fly.
    """
    
    def __init__(
        self,
        base_vae_path,
        vae_lora_path,
        cached_latent_metadata,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        
        # Load metadata to get latent dimensions
        with open(cached_latent_metadata, "r") as f:
            metadata = json.load(f)
        
        latent_shape = metadata["latent_shape"]
        z_dim = latent_shape[0]  # Channel dimension of latents
        
        # Create only the feature merge block (we don't need the full VAE)
        self.feature_merge = FeatureMergeBlock(input_dim=z_dim, num_res_blocks=3)
        self.feature_merge.to(device)
        
        # We need the VAE decoders for reconstruction
        self.rgba_vae = RGBAlphaVAE(
            base_vae_path=base_vae_path,
            vae_lora_path=vae_lora_path,
            z_dim=z_dim,
            dtype=dtype,
            device=device,
            with_feature_merge=False  # We use our own feature merge
        )
        self.rgba_vae.eval()  # Decoders in eval mode
        self.rgba_vae.freeze_decoders()
        
        # Loss function
        self.loss_fn = RGBAReconstructionLoss()
    
    def forward(self, batch_data):
        """
        Forward pass using cached latents.
        
        Args:
            batch_data: Dict containing cached latents and ground truth
        
        Returns:
            loss: Total loss
            loss_dict: Detailed loss breakdown
        """
        # Extract cached latents
        rgb_latents = batch_data["rgb_latents"].to(self.dtype)
        alpha_latents = batch_data["alpha_latents"].to(self.dtype)
        
        # Extract ground truth tensors
        rgb_tensors = batch_data["rgb_tensors"]
        alpha_tensors = batch_data["alpha_tensors"]
        hard_rgb_tensors = batch_data["hard_rgb_tensors"]
        soft_rgb_tensors = batch_data["soft_rgb_tensors"]
        
        # Colors for rendering
        hard_colors = batch_data["hard_colors"]
        soft_colors = batch_data["soft_colors"]
        
        # Apply feature merge block
        merged_latents = self.feature_merge(rgb_latents, alpha_latents)
        
        # Decode with dual decoders (convert to list format expected by VAE)
        merged_latents_list = [merged_latents[i] for i in range(merged_latents.shape[0])]
        
        with torch.no_grad():  # Decoders are frozen
            pred_rgb_list, pred_alpha_list = self.rgba_vae.decode(
                merged_latents_list, 
                tiled=True,  # Use tiling for memory efficiency
                tile_size=(34, 34), 
                tile_stride=(18, 16)
            )
        
        # Stack predictions back to batch tensor
        pred_rgb = torch.stack(pred_rgb_list)
        pred_alpha = torch.stack(pred_alpha_list)
        
        # Apply rendering for loss calculation
        batch_size = pred_rgb.shape[0]
        pred_rgb_soft_list = []
        pred_rgb_hard_list = []
        
        for i in range(batch_size):
            # Soft rendering: R_s = RGB * α + c * (1 - α)
            color_tensor = torch.tensor(soft_colors[i], device=self.device).view(3, 1, 1, 1)
            pred_rgb_soft = pred_rgb[i] * pred_alpha[i] + color_tensor * (1 - pred_alpha[i])
            pred_rgb_soft_list.append(pred_rgb_soft)
            
            # Hard rendering: R_h = RGB if α > 0.5 else c
            color_tensor = torch.tensor(hard_colors[i], device=self.device).view(3, 1, 1, 1)
            mask = (pred_alpha[i] > 0.0).float()
            pred_rgb_hard = pred_rgb[i] * mask + color_tensor * (1 - mask)
            pred_rgb_hard_list.append(pred_rgb_hard)
        
        pred_rgb_soft = torch.stack(pred_rgb_soft_list)
        pred_rgb_hard = torch.stack(pred_rgb_hard_list)
        
        # Calculate losses
        total_loss, loss_dict = self.loss_fn(
            pred_alpha=pred_alpha,
            gt_alpha=alpha_tensors,
            pred_rgb_soft=pred_rgb_soft,
            gt_rgb_soft=soft_rgb_tensors,
            pred_rgb_hard=pred_rgb_hard,
            gt_rgb_hard=hard_rgb_tensors,
        )
        
        return total_loss, loss_dict


def train_epoch(model, dataloader, optimizer, accelerator, epoch, use_wandb=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process
    )
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Forward pass
        loss, loss_dict = model(batch_data)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        batch_size = batch_data["rgb_latents"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = total_loss / total_samples
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                **{k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            })
            
            # Log to wandb
            if use_wandb and accelerator.is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    **{f"train/{k}": v.item() for k, v in loss_dict.items()},
                    "train/epoch": epoch,
                    "train/step": epoch * len(dataloader) + batch_idx
                })
    
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train Feature Merge Block with cached latents")
    
    # Model arguments
    parser.add_argument("--base_vae_path", type=str, required=True,
                        help="Path to base Wan2.1_VAE.pth")
    parser.add_argument("--vae_lora_path", type=str, required=True,
                        help="Path to decoder.bin (VAE LoRAs)")
    
    # Dataset arguments
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory containing cached latents")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # Output arguments
    parser.add_argument("--output_path", type=str, default="./models/train/feature_merge_cached",
                        help="Path to save trained model")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="wan-alpha-feature-merge",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        log_with="wandb" if args.use_wandb else None,
    )
    
    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args
        )
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading cached latent dataset from {args.cache_dir}")
    dataset = CachedLatentDataset(
        cache_dir=args.cache_dir,
        phase="train",
        deterministic=False,
        load_tensors=True,  # Need tensors for loss calculation
        load_latents=True   # Need latents for feature merge
    )
    
    # Print dataset statistics
    if accelerator.is_main_process:
        stats = dataset.get_latent_statistics()
        print("Latent statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
    
    # Create dataloader
    dataloader = CachedLatentDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=accelerator.device
    )
    
    # Get metadata path
    cached_latent_metadata = os.path.join(args.cache_dir, "metadata.json")
    
    # Create model
    model = CachedFeatureMergeTrainer(
        base_vae_path=args.base_vae_path,
        vae_lora_path=args.vae_lora_path,
        cached_latent_metadata=cached_latent_metadata,
        device=accelerator.device,
        dtype=torch.bfloat16,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.feature_merge.parameters(),  # Only optimize feature merge
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Prepare for distributed training
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Training loop
    print(f"Starting Feature Merge Block training with cached latents...")
    print(f"Output path: {output_dir}")
    
    for epoch in range(args.num_epochs):
        # Train for one epoch
        avg_loss = train_epoch(
            model, dataloader, optimizer, accelerator, epoch, args.use_wandb
        )
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_path = output_dir / f"feature_merge_epoch_{epoch+1}.pt"
                
                # Get the unwrapped model
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Save only the feature merge block
                torch.save({
                    "epoch": epoch + 1,
                    "feature_merge_state_dict": unwrapped_model.feature_merge.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args)
                }, checkpoint_path)
                
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if accelerator.is_main_process:
        final_path = output_dir / "feature_merge_final.pt"
        unwrapped_model = accelerator.unwrap_model(model)
        
        torch.save({
            "epoch": args.num_epochs,
            "feature_merge_state_dict": unwrapped_model.feature_merge.state_dict(),
            "args": vars(args)
        }, final_path)
        
        print(f"✅ Training complete! Final model saved to {final_path}")
        
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
