"""
Feature Merge Block Training for Wan-Alpha (Stage 1)

This script trains ONLY the FeatureMergeBlock using:
- Frozen: Base VAE encoder
- Frozen: RGB and Alpha decoder LoRAs (from decoder.bin)
- Trainable: FeatureMergeBlock

After training, the FeatureMergeBlock can be used for DiT DoRA training (Stage 2).
"""

import torch
import os
from tqdm import tqdm
from diffsynth.models.wan_video_vae import RGBAlphaVAE
from diffsynth.trainers.rgba_losses import RGBAReconstructionLoss
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    LoadRGBAVideoPair,
    HardRenderRGBA,
    SoftRenderRGBA,
    ImageCropAndResize,
    ToAbsolutePath,
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import argparse


class FeatureMergeTrainingModule(torch.nn.Module):
    """Training module for Feature Merge Block."""

    def __init__(
        self,
        base_vae_path,
        vae_lora_path,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()

        # Load RGBA VAE with feature merge block
        self.rgba_vae = RGBAlphaVAE(
            base_vae_path=base_vae_path,
            vae_lora_path=vae_lora_path,
            z_dim=16,
            dtype=dtype,
            device=device,
            with_feature_merge=True,
        )

        # Freeze decoders (we only train feature merge)
        self.rgba_vae.freeze_decoders()

        # Unfreeze feature merge block
        self.rgba_vae.unfreeze_feature_merge()

        # Loss function
        self.loss_fn = RGBAReconstructionLoss()

    def forward(self, data):
        """
        Forward pass for Feature Merge Block training.

        Args:
            data: Dict containing:
                - rgb_video: Original RGB frames
                - alpha_video: 3-channel alpha frames
                - hard_rgb_video: Hard-rendered RGB
                - soft_rgb_video: Soft-rendered RGB

        Returns:
            loss: Total VAE loss
            loss_dict: Detailed loss breakdown
        """
        # Extract data
        rgb_video = data["rgb_video"]  # List of PIL Images
        alpha_video = data["alpha_video"]  # List of PIL Images (3-channel)
        hard_rgb_video = data["hard_rgb_video"]  # List of PIL Images
        soft_rgb_video = data["soft_rgb_video"]  # List of PIL Images

        # Convert PIL to tensors [B, C, T, H, W]
        rgb_tensor = self.pil_to_tensor(rgb_video)
        alpha_tensor = self.pil_to_tensor(alpha_video)
        hard_rgb_tensor = self.pil_to_tensor(hard_rgb_video)
        soft_rgb_tensor = self.pil_to_tensor(soft_rgb_video)

        # Encode with feature merging
        # hard_rgb is used for encoding to prevent color/transparency confusion
        merged_latents = self.rgba_vae.encode_with_merge(
            [hard_rgb_tensor], [alpha_tensor], tiled=False
        )

        # Decode with dual decoders
        pred_rgb_list, pred_alpha_list = self.rgba_vae.decode(
            merged_latents, tiled=False
        )

        pred_rgb = pred_rgb_list[0]
        pred_alpha = pred_alpha_list[0]

        # Apply soft and hard rendering to predicted RGB for loss calculation
        # For soft rendering loss
        pred_rgb_soft = self.apply_soft_render(pred_rgb, pred_alpha, data["soft_color"])

        # For hard rendering loss
        pred_rgb_hard = self.apply_hard_render(pred_rgb, pred_alpha, data["hard_color"])

        # Calculate losses
        total_loss, loss_dict = self.loss_fn(
            pred_alpha=pred_alpha,
            gt_alpha=alpha_tensor,
            pred_rgb_soft=pred_rgb_soft,
            gt_rgb_soft=soft_rgb_tensor,
            pred_rgb_hard=pred_rgb_hard,
            gt_rgb_hard=hard_rgb_tensor,
        )

        return total_loss, loss_dict

    def pil_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor [B, C, T, H, W] in range [-1, 1]"""
        import torchvision.transforms as transforms

        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img) for img in pil_images]
        video_tensor = torch.stack(tensors, dim=1)  # [C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
        video_tensor = video_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        return video_tensor

    def apply_soft_render(self, rgb, alpha, color):
        """Apply soft rendering: R_s = RGB * α + c * (1 - α)"""
        # rgb, alpha in range [-1, 1], convert to [0, 1]
        rgb_01 = (rgb + 1.0) / 2.0
        alpha_01 = (alpha + 1.0) / 2.0

        # Color to tensor [1, 3, 1, 1, 1]
        color_tensor = (
            torch.tensor(color, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1, 1)
            / 255.0
        )

        # Take mean of alpha channels
        alpha_mean = alpha_01.mean(dim=1, keepdim=True)

        # Soft render
        rendered = rgb_01 * alpha_mean + color_tensor * (1 - alpha_mean)

        # Convert back to [-1, 1]
        return rendered * 2.0 - 1.0

    def apply_hard_render(self, rgb, alpha, color):
        """Apply hard rendering: R_h = RGB * α + c * (1 - α) where α is binarized"""
        # rgb, alpha in range [-1, 1], convert to [0, 1]
        rgb_01 = (rgb + 1.0) / 2.0
        alpha_01 = (alpha + 1.0) / 2.0

        # Color to tensor
        color_tensor = (
            torch.tensor(color, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1, 1)
            / 255.0
        )

        # Binarize alpha (hard mask)
        alpha_mean = alpha_01.mean(dim=1, keepdim=True)
        hard_alpha = (alpha_mean > 0.5).float()

        # Hard render
        rendered = rgb_01 * hard_alpha + color_tensor * (1 - hard_alpha)

        # Convert back to [-1, 1]
        return rendered * 2.0 - 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Feature Merge Block for Wan-Alpha (Stage 1)"
    )
    parser.add_argument(
        "--dataset_base_path", type=str, required=True, help="Base path of RGBA dataset"
    )
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        required=True,
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--base_vae_path", type=str, required=True, help="Path to base Wan2.1_VAE.pth"
    )
    parser.add_argument(
        "--vae_lora_path",
        type=str,
        required=True,
        help="Path to decoder.bin (VAE LoRAs)",
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/train/feature_merge",
        help="Output path",
    )
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Dataset repeat")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data workers"
    )

    args = parser.parse_args()

    # Create dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=["video"],
        main_data_operator=ToAbsolutePath(args.dataset_base_path)
        >> LoadRGBAVideoPair(
            num_frames=args.num_frames,
            frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16),
        )
        >> HardRenderRGBA()  # Apply hard rendering
        >> SoftRenderRGBA(),  # Apply soft rendering
    )

    # Create training module
    model = FeatureMergeTrainingModule(
        base_vae_path=args.base_vae_path,
        vae_lora_path=args.vae_lora_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Setup training
    optimizer = torch.optim.AdamW(
        model.rgba_vae.feature_merge.parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.num_workers
    )

    # Accelerator setup
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training loop
    print(f"Starting Feature Merge Block training...")
    print(f"Output path: {args.output_path}")

    for epoch_id in range(args.num_epochs):
        epoch_losses = []

        for data in tqdm(dataloader, desc=f"Epoch {epoch_id}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Forward pass
                loss, loss_dict = model(data)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                # Log
                epoch_losses.append(loss_dict)

        # Print epoch stats
        avg_loss = sum([d["total"] for d in epoch_losses]) / len(epoch_losses)
        print(f"Epoch {epoch_id} - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.output_path, exist_ok=True)

            # Save only feature merge block weights
            feature_merge_state = model.rgba_vae.feature_merge.state_dict()
            save_path = os.path.join(
                args.output_path, f"feature_merge_epoch_{epoch_id}.safetensors"
            )

            accelerator.save(feature_merge_state, save_path, safe_serialization=True)
            print(f"✅ Saved: {save_path}")

    print("✅ Feature Merge Block training completed!")
