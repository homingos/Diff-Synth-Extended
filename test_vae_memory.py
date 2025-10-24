"""
Test script to verify VAE memory usage is reasonable
"""

import torch
import os
import sys

sys.path.append(os.path.dirname(__file__))

from diffsynth.models.wan_video_vae import VideoVAE_, RGBAlphaVAE


def test_vae_memory():
    """Test that VAE decode doesn't explode memory usage"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple VAE
    vae = VideoVAE_(z_dim=16)
    vae.to(device)
    vae.eval()

    # Create test data
    batch_size = 1
    channels = 16
    frames = 5  # Small number of frames
    height = 60  # Small spatial dimensions
    width = 104

    # Create random latent
    z = torch.randn(batch_size, channels, frames, height, width).to(device)

    # Test scale
    mean = torch.zeros(16).to(device)
    std = torch.ones(16).to(device)
    scale = [mean, 1.0 / std]

    print(f"Testing VAE decode with shape: {z.shape}")

    # Test decode WITHOUT caching (should be memory efficient)
    print("\nTesting decode without caching...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        output = vae.decode(z, scale, use_cache=False)

    if device == "cuda":
        end_mem = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"Memory usage: Start={start_mem:.1f}MB, End={end_mem:.1f}MB, Peak={peak_mem:.1f}MB"
        )
        print(f"Memory increase: {peak_mem - start_mem:.1f}MB")

    print(f"Output shape: {output.shape}")

    # Test tiled decode
    print("\nTesting tiled decode...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        output_tiled = vae.tiled_decode(
            z, scale, tile_size=(34, 34), tile_stride=(18, 16)
        )

    if device == "cuda":
        end_mem = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"Memory usage: Start={start_mem:.1f}MB, End={end_mem:.1f}MB, Peak={peak_mem:.1f}MB"
        )
        print(f"Memory increase: {peak_mem - start_mem:.1f}MB")

    print(f"Tiled output shape: {output_tiled.shape}")

    # Compare outputs
    if torch.allclose(output, output_tiled, rtol=1e-3, atol=1e-3):
        print("\n✅ Tiled and non-tiled outputs match!")
    else:
        print("\n❌ Outputs don't match!")
        print(f"Max difference: {(output - output_tiled).abs().max().item()}")

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_vae_memory()
