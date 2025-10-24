"""
Pre-cache VAE latents for the RGBA dataset to speed up training.
This script encodes all videos in the dataset and saves the latents to disk.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from diffsynth.models.wan_video_vae import RGBAlphaVAE
from diffsynth.trainers.unified_dataset import UnifiedRGBADataset


class LatentCacher:
    """Cache VAE latents for RGBA dataset"""
    
    def __init__(
        self,
        dataset_base_path,
        dataset_metadata_path,
        base_vae_path,
        vae_lora_path,
        output_dir,
        height=480,
        width=832,
        num_frames=81,
        batch_size=1,
        num_workers=4,
        device="cuda",
        dtype=torch.bfloat16,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16)
    ):
        self.dataset_base_path = dataset_base_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.dtype = dtype
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        
        # Load VAE (without feature merge block since we're just encoding)
        print("Loading VAE...")
        self.vae = RGBAlphaVAE(
            base_vae_path=base_vae_path,
            vae_lora_path=vae_lora_path,
            z_dim=16,
            dtype=dtype,
            device=device,
            with_feature_merge=False  # Don't need feature merge for caching
        )
        self.vae.eval()
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = UnifiedRGBADataset(
            base_path=dataset_base_path,
            metadata_path=dataset_metadata_path,
            height=height,
            width=width,
            num_frames=num_frames,
            phase="train",
            deterministic=True  # Ensure reproducibility
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for easy indexing
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Metadata storage
        self.metadata = {
            "dataset_base_path": dataset_base_path,
            "dataset_metadata_path": dataset_metadata_path,
            "base_vae_path": base_vae_path,
            "vae_lora_path": vae_lora_path,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "dtype": str(dtype),
            "latent_shape": None,  # Will be filled after first encoding
            "num_samples": len(self.dataset),
            "video_metadata": []
        }
    
    def pil_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor [C, T, H, W] in range [-1, 1]"""
        import torchvision.transforms as transforms
        
        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img) for img in pil_images]
        video_tensor = torch.stack(tensors, dim=1)  # [C, T, H, W]
        video_tensor = video_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        return video_tensor.to(self.device)
    
    def cache_all_latents(self):
        """Cache latents for all videos in the dataset"""
        print(f"Caching latents for {len(self.dataset)} videos...")
        
        with torch.no_grad():
            for idx, data in enumerate(tqdm(self.dataloader, desc="Caching latents")):
                # Extract data
                video_path = data["video_path"][0]  # Batch size 1
                rgb_video = data["rgb_video"]
                alpha_video = data["alpha_video"]
                hard_rgb_video = data["hard_rgb_video"]
                soft_rgb_video = data["soft_rgb_video"]
                
                # Convert PIL to tensors
                rgb_tensor = self.pil_to_tensor(rgb_video)
                alpha_tensor = self.pil_to_tensor(alpha_video)
                hard_rgb_tensor = self.pil_to_tensor(hard_rgb_video)
                soft_rgb_tensor = self.pil_to_tensor(soft_rgb_video)
                
                # Encode videos
                # Use hard_rgb for encoding (as in training) to prevent color/transparency confusion
                rgb_latents = self.vae.encode(
                    [hard_rgb_tensor],
                    tiled=self.tiled,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride
                )
                
                alpha_latents = self.vae.encode(
                    [alpha_tensor],
                    tiled=self.tiled,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride
                )
                
                # Save latent shape metadata (first time only)
                if self.metadata["latent_shape"] is None:
                    self.metadata["latent_shape"] = list(rgb_latents[0].shape)
                
                # Create output filename based on video path
                relative_path = os.path.relpath(video_path, self.dataset_base_path)
                latent_filename = relative_path.replace("/", "_").replace(".mp4", ".pt")
                latent_path = self.output_dir / latent_filename
                
                # Save latents and ground truth tensors
                latent_data = {
                    "rgb_latent": rgb_latents[0].cpu(),  # Remove batch dimension
                    "alpha_latent": alpha_latents[0].cpu(),
                    "rgb_tensor": rgb_tensor.cpu(),
                    "alpha_tensor": alpha_tensor.cpu(),
                    "hard_rgb_tensor": hard_rgb_tensor.cpu(),
                    "soft_rgb_tensor": soft_rgb_tensor.cpu(),
                    "hard_color": data["hard_color"],
                    "soft_color": data["soft_color"],
                    "video_path": video_path,
                    "index": idx
                }
                
                torch.save(latent_data, latent_path)
                
                # Store metadata
                self.metadata["video_metadata"].append({
                    "index": idx,
                    "video_path": video_path,
                    "latent_path": str(latent_path),
                    "relative_path": relative_path
                })
                
                # Clear GPU cache periodically
                if idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Cached {len(self.dataset)} latents to {self.output_dir}")
        print(f"✅ Metadata saved to {metadata_path}")
    
    def verify_cache(self):
        """Verify that all cached latents are valid"""
        print("Verifying cached latents...")
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        errors = []
        for video_meta in tqdm(metadata["video_metadata"], desc="Verifying"):
            latent_path = video_meta["latent_path"]
            
            try:
                # Load and check latent
                latent_data = torch.load(latent_path, map_location="cpu")
                
                # Check all required keys
                required_keys = [
                    "rgb_latent", "alpha_latent", "rgb_tensor", "alpha_tensor",
                    "hard_rgb_tensor", "soft_rgb_tensor", "hard_color", "soft_color"
                ]
                for key in required_keys:
                    if key not in latent_data:
                        errors.append(f"{latent_path}: Missing key {key}")
                
                # Check latent shapes
                if list(latent_data["rgb_latent"].shape) != metadata["latent_shape"]:
                    errors.append(f"{latent_path}: RGB latent shape mismatch")
                
                if list(latent_data["alpha_latent"].shape) != metadata["latent_shape"]:
                    errors.append(f"{latent_path}: Alpha latent shape mismatch")
                    
            except Exception as e:
                errors.append(f"{latent_path}: {str(e)}")
        
        if errors:
            print(f"❌ Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
        else:
            print(f"✅ All {len(metadata['video_metadata'])} cached latents are valid!")


def main():
    parser = argparse.ArgumentParser(description="Cache VAE latents for RGBA dataset")
    
    # Dataset arguments
    parser.add_argument("--dataset_base_path", type=str, required=True,
                        help="Base path to RGBA video dataset")
    parser.add_argument("--dataset_metadata_path", type=str, required=True,
                        help="Path to dataset metadata CSV")
    
    # Model arguments
    parser.add_argument("--base_vae_path", type=str, required=True,
                        help="Path to base Wan2.1_VAE.pth")
    parser.add_argument("--vae_lora_path", type=str, required=True,
                        help="Path to decoder.bin (VAE LoRAs)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./cached_latents",
                        help="Directory to save cached latents")
    
    # Video specifications
    parser.add_argument("--height", type=int, default=480,
                        help="Video height")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames per video")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing cache")
    
    # Tiling arguments
    parser.add_argument("--no_tiling", action="store_true",
                        help="Disable tiled encoding")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[34, 34],
                        help="Tile size for tiled encoding")
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[18, 16],
                        help="Tile stride for tiled encoding")
    
    args = parser.parse_args()
    
    # Create cacher
    cacher = LatentCacher(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path,
        base_vae_path=args.base_vae_path,
        vae_lora_path=args.vae_lora_path,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        tiled=not args.no_tiling,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride)
    )
    
    if args.verify_only:
        cacher.verify_cache()
    else:
        cacher.cache_all_latents()
        cacher.verify_cache()


if __name__ == "__main__":
    main()
