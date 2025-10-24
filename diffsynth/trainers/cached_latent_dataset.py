"""
Cached Latent Dataset for fast RGBA training.
Loads pre-cached VAE latents instead of encoding on the fly.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class CachedLatentDataset(Dataset):
    """
    Dataset that loads pre-cached VAE latents for RGBA training.
    This avoids repeated VAE encoding during training and significantly speeds up the process.
    """
    
    def __init__(
        self,
        cache_dir: str,
        phase: str = "train",
        deterministic: bool = False,
        load_tensors: bool = True,
        load_latents: bool = True
    ):
        """
        Args:
            cache_dir: Directory containing cached latents and metadata.json
            phase: Dataset phase (train/val/test)
            deterministic: If True, use fixed random seed
            load_tensors: If True, load ground truth tensors (for loss calculation)
            load_latents: If True, load cached latents
        """
        self.cache_dir = Path(cache_dir)
        self.phase = phase
        self.deterministic = deterministic
        self.load_tensors = load_tensors
        self.load_latents = load_latents
        
        # Load metadata
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Extract video metadata
        self.video_metadata = self.metadata["video_metadata"]
        self.num_samples = len(self.video_metadata)
        
        # Extract configuration
        self.height = self.metadata["height"]
        self.width = self.metadata["width"]
        self.num_frames = self.metadata["num_frames"]
        self.latent_shape = self.metadata["latent_shape"]
        
        # Set random seed if deterministic
        if deterministic:
            random.seed(42)
            torch.manual_seed(42)
        
        print(f"Loaded cached latent dataset with {self.num_samples} samples")
        print(f"Video shape: {self.num_frames}x{self.height}x{self.width}")
        print(f"Latent shape: {self.latent_shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Load cached latent and associated data"""
        video_meta = self.video_metadata[idx]
        latent_path = video_meta["latent_path"]
        
        # Load cached data
        latent_data = torch.load(latent_path, map_location="cpu")
        
        # Prepare output dictionary
        output = {
            "index": idx,
            "video_path": latent_data["video_path"],
            "hard_color": latent_data["hard_color"],
            "soft_color": latent_data["soft_color"]
        }
        
        # Add latents if requested
        if self.load_latents:
            output["rgb_latent"] = latent_data["rgb_latent"]
            output["alpha_latent"] = latent_data["alpha_latent"]
        
        # Add tensors if requested (for loss calculation)
        if self.load_tensors:
            output["rgb_tensor"] = latent_data["rgb_tensor"]
            output["alpha_tensor"] = latent_data["alpha_tensor"]
            output["hard_rgb_tensor"] = latent_data["hard_rgb_tensor"]
            output["soft_rgb_tensor"] = latent_data["soft_rgb_tensor"]
        
        return output
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of samples by indices"""
        batch_data = [self[idx] for idx in indices]
        
        # Stack tensors
        output = {
            "indices": torch.tensor(indices),
            "hard_colors": [d["hard_color"] for d in batch_data],
            "soft_colors": [d["soft_color"] for d in batch_data],
            "video_paths": [d["video_path"] for d in batch_data]
        }
        
        if self.load_latents:
            output["rgb_latents"] = torch.stack([d["rgb_latent"] for d in batch_data])
            output["alpha_latents"] = torch.stack([d["alpha_latent"] for d in batch_data])
        
        if self.load_tensors:
            output["rgb_tensors"] = torch.stack([d["rgb_tensor"] for d in batch_data])
            output["alpha_tensors"] = torch.stack([d["alpha_tensor"] for d in batch_data])
            output["hard_rgb_tensors"] = torch.stack([d["hard_rgb_tensor"] for d in batch_data])
            output["soft_rgb_tensors"] = torch.stack([d["soft_rgb_tensor"] for d in batch_data])
        
        return output
    
    def get_latent_statistics(self) -> Dict[str, torch.Tensor]:
        """Calculate statistics of cached latents for monitoring"""
        print("Calculating latent statistics...")
        
        rgb_latents = []
        alpha_latents = []
        
        # Load all latents
        for i in range(min(100, len(self))):  # Sample first 100 for efficiency
            data = self[i]
            if self.load_latents:
                rgb_latents.append(data["rgb_latent"])
                alpha_latents.append(data["alpha_latent"])
        
        if not rgb_latents:
            return {}
        
        # Stack latents
        rgb_latents = torch.stack(rgb_latents)
        alpha_latents = torch.stack(alpha_latents)
        
        # Calculate statistics
        stats = {
            "rgb_mean": rgb_latents.mean(),
            "rgb_std": rgb_latents.std(),
            "rgb_min": rgb_latents.min(),
            "rgb_max": rgb_latents.max(),
            "alpha_mean": alpha_latents.mean(),
            "alpha_std": alpha_latents.std(),
            "alpha_min": alpha_latents.min(),
            "alpha_max": alpha_latents.max(),
        }
        
        return stats


class CachedLatentDataLoader:
    """
    Custom dataloader for cached latents that supports efficient batching
    and prefetching of latent data.
    """
    
    def __init__(
        self,
        dataset: CachedLatentDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        device: str = "cuda"
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = device
        
        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        
        # Batch indices
        self.batches = []
        for i in range(0, len(self.indices), batch_size):
            batch_indices = self.indices[i:i + batch_size]
            self.batches.append(batch_indices)
        
        self.current_batch = 0
    
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            random.shuffle(self.batches)
        return self
    
    def __next__(self):
        if self.current_batch >= len(self.batches):
            raise StopIteration
        
        batch_indices = self.batches[self.current_batch]
        self.current_batch += 1
        
        # Get batch data
        batch_data = self.dataset.get_batch(batch_indices)
        
        # Move to device
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(self.device)
        
        return batch_data
