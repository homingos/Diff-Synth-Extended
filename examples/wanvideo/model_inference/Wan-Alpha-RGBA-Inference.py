"""
Wan-Alpha RGBA Video Generation Inference Script

Generates RGBA videos (RGB + Alpha channel) using:
- Trained DiT with DoRA
- Trained FeatureMergeBlock
- Dual VAE decoders (decoder.bin)
"""

import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_vae import RGBAlphaVAE
import numpy as np
import imageio


# Load standard Wan pipeline with DoRA
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-14B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-14B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-14B",
            origin_file_pattern="Wan2.1_VAE.pth",
            offload_device="cpu",
        ),
    ],
)

# Load trained DiT DoRA weights
pipe.load_dora(
    pipe.dit, "models/train/Wan-Alpha-DiT-DoRA/epoch-14.safetensors", alpha=1.0
)

# Setup RGBA VAE with trained FeatureMergeBlock
rgba_vae = RGBAlphaVAE(
    base_vae_path="models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
    vae_lora_path="Wan-Alpha/vae/decoder.bin",
    z_dim=16,
    dtype=torch.bfloat16,
    device="cuda",
    with_feature_merge=False,  # For inference, we don't need merge block
)

# Replace standard VAE with RGBA VAE
pipe.vae = rgba_vae

pipe.enable_vram_management()

# Generate RGBA video
prompt = "This video has a transparent background. Close-up shot. A colorful butterfly flying. Realistic style."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Generate latent with DiT
print("Generating latent with DiT...")
# Note: Standard pipeline generation, but VAE will produce RGBA
latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42,
    tiled=True,
    return_latent=True,  # Return latent instead of decoded video
)

# Decode with dual decoders
print("Decoding to RGB and Alpha...")
rgb_videos, alpha_videos = rgba_vae.decode([latent], tiled=True)

rgb_video = rgb_videos[0]  # [C, T, H, W]
alpha_video = alpha_videos[0]  # [C, T, H, W]


# Convert to numpy for saving
def tensor_to_numpy(tensor):
    """Convert tensor [-1, 1] to numpy [0, 255]"""
    tensor = (tensor + 1.0) / 2.0  # [-1, 1] → [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    # [C, T, H, W] → [T, H, W, C]
    numpy_array = np.transpose(numpy_array, (1, 2, 3, 0))
    return numpy_array


rgb_numpy = tensor_to_numpy(rgb_video)
alpha_numpy = tensor_to_numpy(alpha_video)

# Take first channel of alpha (all 3 channels are same)
alpha_numpy = alpha_numpy[:, :, :, 0]

# Combine RGB + Alpha = RGBA
rgba_numpy = np.concatenate(
    [rgb_numpy, alpha_numpy[:, :, :, np.newaxis]], axis=3
)  # [T, H, W, 4]

# Save RGB video
print("Saving RGB video...")
save_video(VideoData(rgb_numpy), "output_rgb.mp4", fps=15, quality=5)

# Save alpha video
print("Saving alpha video...")
alpha_3ch = np.stack([alpha_numpy] * 3, axis=3)  # Convert to 3-channel for video
save_video(VideoData(alpha_3ch), "output_alpha.mp4", fps=15, quality=5)

# Save RGBA frames as PNG sequence
print("Saving RGBA frames...")
os.makedirs("output_rgba_frames", exist_ok=True)
for i, frame in enumerate(rgba_numpy):
    img = Image.fromarray(frame, mode="RGBA")
    img.save(f"output_rgba_frames/frame_{i:04d}.png")

# Create checkerboard composite for visualization
print("Creating checkerboard composite...")


def create_checkerboard(height, width, square_size=30):
    """Create checkerboard background"""
    checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
    color1, color2 = 140, 113

    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                checkerboard[i : i + square_size, j : j + square_size] = color1
            else:
                checkerboard[i : i + square_size, j : j + square_size] = color2

    return checkerboard


composite_frames = []
for rgb_frame, alpha_frame in zip(rgb_numpy, alpha_numpy):
    # Create checkerboard
    checkerboard = create_checkerboard(rgb_frame.shape[0], rgb_frame.shape[1])

    # Composite: RGB * alpha + checkerboard * (1 - alpha)
    alpha_3d = alpha_frame[:, :, np.newaxis] / 255.0
    composite = (rgb_frame * alpha_3d + checkerboard * (1 - alpha_3d)).astype(np.uint8)
    composite_frames.append(composite)

composite_numpy = np.stack(composite_frames, axis=0)
save_video(VideoData(composite_numpy), "output_composite.mp4", fps=15, quality=5)

print("✅ RGBA generation complete!")
print("  - output_rgb.mp4: RGB video")
print("  - output_alpha.mp4: Alpha video")
print("  - output_rgba_frames/: PNG frames with alpha channel")
print("  - output_composite.mp4: RGB composited on checkerboard")
