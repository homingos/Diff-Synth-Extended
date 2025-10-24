"""
Dataset preparation script for Wan-Alpha training.
Reorganizes dataset from format:
  - XXX.mp4 (RGB)
  - XXX_seg.mp4 (Alpha/Segmentation)
  - XXX.txt (Caption)

To format:
  - XXX_rgb.mp4
  - XXX_alpha.mp4
  - metadata.csv
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def prepare_rgba_dataset(
    source_dir="dataset_30FPS",
    output_dir="data/rgba_videos",
    copy_files=True,  # Set to False to create symlinks instead
):
    """
    Reorganize dataset and create metadata.csv

    Args:
        source_dir: Source directory with XXX.mp4, XXX_seg.mp4, XXX.txt files
        output_dir: Output directory for reorganized dataset
        copy_files: If True, copy files; if False, create symlinks
    """

    print(f"🔄 Preparing RGBA dataset...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all RGB videos (files ending with .mp4 but not _seg.mp4)
    source_path = Path(source_dir)
    all_mp4_files = sorted(source_path.glob("*.mp4"))

    rgb_files = [f for f in all_mp4_files if not f.name.endswith("_seg.mp4")]

    print(f"📊 Found {len(rgb_files)} RGB videos")

    # Prepare metadata
    metadata_rows = []
    processed_count = 0
    skipped_count = 0

    for rgb_file in tqdm(rgb_files, desc="Processing videos"):
        # Get base name (e.g., "001" from "001.mp4")
        base_name = rgb_file.stem  # e.g., "001"

        # Find corresponding segmentation and caption files
        seg_file = source_path / f"{base_name}_seg.mp4"
        txt_file = source_path / f"{base_name}.txt"

        # Check if all required files exist
        if not seg_file.exists():
            print(f"  ⚠️  Skipping {base_name}: No segmentation file {seg_file.name}")
            skipped_count += 1
            continue

        if not txt_file.exists():
            print(f"  ⚠️  Skipping {base_name}: No caption file {txt_file.name}")
            skipped_count += 1
            continue

        # Read caption
        with open(txt_file, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Define new filenames
        new_rgb_name = f"{base_name}_rgb.mp4"
        new_alpha_name = f"{base_name}_alpha.mp4"

        # Copy or symlink files
        rgb_dest = Path(output_dir) / new_rgb_name
        alpha_dest = Path(output_dir) / new_alpha_name

        if copy_files:
            # Copy files
            if not rgb_dest.exists():
                shutil.copy2(rgb_file, rgb_dest)
            if not alpha_dest.exists():
                shutil.copy2(seg_file, alpha_dest)
        else:
            # Create symlinks (faster, saves space)
            if not rgb_dest.exists():
                rgb_dest.symlink_to(rgb_file.absolute())
            if not alpha_dest.exists():
                alpha_dest.symlink_to(seg_file.absolute())

        # Add to metadata
        metadata_rows.append(
            {
                "video": base_name,  # Just the base name, LoadRGBAVideoPair will add suffixes
                "prompt": caption,
            }
        )

        processed_count += 1

    # Create metadata.csv
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = Path(output_dir) / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print()
    print("═══════════════════════════════════════════════════════")
    print("✅ DATASET PREPARATION COMPLETE")
    print("═══════════════════════════════════════════════════════")
    print(f"  Processed: {processed_count} videos")
    print(f"  Skipped: {skipped_count} videos")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata file: {metadata_path}")
    print()
    print("📝 Metadata structure:")
    print(metadata_df.head())
    print()
    print("📂 Sample files created:")
    sample_files = list(Path(output_dir).glob("001*"))[:4]
    for f in sorted(sample_files):
        print(f"  - {f.name}")
    print()
    print("🚀 Ready for training!")
    print(f"\nRun training with:")
    print(
        f"  accelerate launch examples/wanvideo/model_training/train_feature_merge.py \\"
    )
    print(f"    --dataset_base_path {output_dir} \\")
    print(f"    --dataset_metadata_path {metadata_path} \\")
    print(f"    --base_vae_path models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth \\")
    print(f"    --vae_lora_path Wan-Alpha/vae/decoder.bin \\")
    print(f"    --height 480 --width 832 --num_frames 81 \\")
    print(f"    --learning_rate 1e-4 --num_epochs 10 \\")
    print(f"    --output_path ./models/train/feature_merge")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare RGBA dataset for Wan-Alpha training"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="dataset_30FPS",
        help="Source directory with XXX.mp4, XXX_seg.mp4, XXX.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/rgba_videos",
        help="Output directory for reorganized dataset",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of creating symlinks"
    )

    args = parser.parse_args()

    prepare_rgba_dataset(
        source_dir=args.source_dir, output_dir=args.output_dir, copy_files=args.copy
    )
