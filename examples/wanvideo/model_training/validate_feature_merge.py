"""
Validation script for trained FeatureMergeBlock.
Tests reconstruction quality on a validation RGBA video.
"""

import torch
import numpy as np
from PIL import Image
from diffsynth.models.wan_video_vae import RGBAlphaVAE, load_decoder_lora
from diffsynth import load_state_dict
import imageio
import argparse


def validate_feature_merge(
    base_vae_path,
    vae_lora_path,
    feature_merge_checkpoint,
    test_rgb_video,
    test_alpha_video,
    output_prefix="validation",
):
    """
    Validate trained FeatureMergeBlock by reconstructing a test video.

    Args:
        base_vae_path: Path to Wan2.1_VAE.pth
        vae_lora_path: Path to decoder.bin
        feature_merge_checkpoint: Path to trained FeatureMergeBlock weights
        test_rgb_video: Path to test RGB video
        test_alpha_video: Path to test alpha video
        output_prefix: Prefix for output files
    """
    print("🧪 Validating FeatureMergeBlock...")

    # Load RGBA VAE
    rgba_vae = RGBAlphaVAE(
        base_vae_path=base_vae_path,
        vae_lora_path=vae_lora_path,
        z_dim=16,
        dtype=torch.bfloat16,
        device="cuda",
        with_feature_merge=True,
    )

    # Load trained FeatureMergeBlock
    print(f"Loading FeatureMergeBlock from {feature_merge_checkpoint}")
    merge_state = load_state_dict(feature_merge_checkpoint)
    rgba_vae.feature_merge.load_state_dict(merge_state)
    print("✅ FeatureMergeBlock loaded")

    # Load test videos
    print(f"Loading test videos...")
    rgb_reader = imageio.get_reader(test_rgb_video)
    alpha_reader = imageio.get_reader(test_alpha_video)

    num_frames = min(81, rgb_reader.count_frames())

    rgb_frames = []
    alpha_frames = []

    for i in range(num_frames):
        # RGB
        rgb_frame = rgb_reader.get_data(i)
        rgb_pil = Image.fromarray(rgb_frame).convert("RGB")
        rgb_frames.append(rgb_pil)

        # Alpha (duplicate to 3 channels)
        alpha_frame = alpha_reader.get_data(i)
        if len(alpha_frame.shape) == 3:
            alpha_frame = alpha_frame[:, :, 0]
        alpha_pil = Image.fromarray(alpha_frame).convert("L")
        alpha_pil_3ch = Image.merge("RGB", [alpha_pil, alpha_pil, alpha_pil])
        alpha_frames.append(alpha_pil_3ch)

    rgb_reader.close()
    alpha_reader.close()

    # Convert to tensors
    def pil_list_to_tensor(pil_images):
        import torchvision.transforms as transforms

        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img) for img in pil_images]
        video_tensor = torch.stack(tensors, dim=1)  # [C, T, H, W]
        video_tensor = video_tensor * 2.0 - 1.0  # [-1, 1]
        return video_tensor

    rgb_tensor = pil_list_to_tensor(rgb_frames)
    alpha_tensor = pil_list_to_tensor(alpha_frames)

    # Encode with feature merge
    print("Encoding with FeatureMergeBlock...")
    with torch.no_grad():
        merged_latents = rgba_vae.encode_with_merge(
            [rgb_tensor], [alpha_tensor], tiled=False
        )

    # Decode
    print("Decoding with dual decoders...")
    with torch.no_grad():
        pred_rgb_list, pred_alpha_list = rgba_vae.decode(merged_latents, tiled=False)

    pred_rgb = pred_rgb_list[0]
    pred_alpha = pred_alpha_list[0]

    # Convert back to numpy
    def tensor_to_video(tensor):
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        numpy_array = np.transpose(
            numpy_array, (1, 2, 3, 0)
        )  # [C, T, H, W] → [T, H, W, C]
        return numpy_array

    pred_rgb_numpy = tensor_to_video(pred_rgb)
    pred_alpha_numpy = tensor_to_video(pred_alpha)
    gt_rgb_numpy = tensor_to_video(rgb_tensor)
    gt_alpha_numpy = tensor_to_video(alpha_tensor)

    # Save outputs
    print("Saving outputs...")

    # Original
    imageio.mimsave(f"{output_prefix}_gt_rgb.mp4", gt_rgb_numpy, fps=15, quality=8)
    imageio.mimsave(f"{output_prefix}_gt_alpha.mp4", gt_alpha_numpy, fps=15, quality=8)

    # Reconstructed
    imageio.mimsave(f"{output_prefix}_pred_rgb.mp4", pred_rgb_numpy, fps=15, quality=8)
    imageio.mimsave(
        f"{output_prefix}_pred_alpha.mp4", pred_alpha_numpy, fps=15, quality=8
    )

    # Calculate reconstruction error
    rgb_error = np.mean(np.abs(pred_rgb_numpy - gt_rgb_numpy))
    alpha_error = np.mean(np.abs(pred_alpha_numpy - gt_alpha_numpy))

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)
    print(f"  RGB reconstruction error (MAE): {rgb_error:.4f}")
    print(f"  Alpha reconstruction error (MAE): {alpha_error:.4f}")
    print(f"\n  Output files:")
    print(f"    - {output_prefix}_gt_rgb.mp4")
    print(f"    - {output_prefix}_gt_alpha.mp4")
    print(f"    - {output_prefix}_pred_rgb.mp4")
    print(f"    - {output_prefix}_pred_alpha.mp4")
    print("=" * 60)

    return rgb_error, alpha_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate trained FeatureMergeBlock")
    parser.add_argument(
        "--base_vae_path", type=str, required=True, help="Path to base Wan2.1_VAE.pth"
    )
    parser.add_argument(
        "--vae_lora_path", type=str, required=True, help="Path to decoder.bin"
    )
    parser.add_argument(
        "--feature_merge_checkpoint",
        type=str,
        required=True,
        help="Path to trained FeatureMergeBlock checkpoint",
    )
    parser.add_argument(
        "--test_rgb_video",
        type=str,
        required=True,
        help="Path to test RGB video (e.g., data/rgba_videos/001_rgb.mp4)",
    )
    parser.add_argument(
        "--test_alpha_video",
        type=str,
        required=True,
        help="Path to test alpha video (e.g., data/rgba_videos/001_alpha.mp4)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="validation",
        help="Prefix for output files",
    )

    args = parser.parse_args()

    validate_feature_merge(
        args.base_vae_path,
        args.vae_lora_path,
        args.feature_merge_checkpoint,
        args.test_rgb_video,
        args.test_alpha_video,
        args.output_prefix,
    )
