#!/usr/bin/env python3
"""
Standalone YOLOv11-OBB Training Script

Run this script in terminal for faster training with multiple workers.
After training, load the model in the notebook for SAM evaluation.

Usage:
    python train_yolo_standalone.py

The trained weights will be saved to:
    dataset_split/runs/obb/polybag_yolov11/weights/best.pt
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO


# ===================== CONFIGURATION =====================
# Source data (raw, unsplit)
SOURCE_ROOT = Path("/home/xegi09mi/19-01-2026_Database/annotated_polyBag_customTool")
SOURCE_IMAGES = SOURCE_ROOT / "images/0000"
SOURCE_LABELS = SOURCE_ROOT / "output/labels"

# Working directory (where we'll create train/val split)
DATA_ROOT = SOURCE_ROOT / "dataset_split"

# Output paths
IMAGES_TRAIN = DATA_ROOT / "images/train"
IMAGES_VAL = DATA_ROOT / "images/val"
LABELS_TRAIN = DATA_ROOT / "labels/train"
LABELS_VAL = DATA_ROOT / "labels/val"
DATA_CONFIG = DATA_ROOT / "polybag_dataset.yaml"

# Model paths
PRETRAINED_WEIGHTS = Path("/home/xegi09mi/19-01-2026_Database/polybag_dataset_yolo_obb/yolo11s-obb.pt")
RUNS_DIR = DATA_ROOT / "runs"
OBB_RUN_DIR = RUNS_DIR / "obb"

# Training defaults
DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 0  # Use 0 to avoid multiprocessing issues (system has limited workers)
DEFAULT_PATIENCE = 25
DEFAULT_IMGSZ = 640
DEFAULT_CACHE = "disk"  # Use 'disk' instead of 'ram' to avoid OOM
# =========================================================


def split_dataset(
    source_images: Path,
    source_labels: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Split dataset into train/val sets.
    """
    print("=" * 60)
    print("📂 SPLITTING DATASET INTO TRAIN/VAL")
    print("=" * 60)
    
    # Create output directories
    train_img_dir = output_root / "images/train"
    val_img_dir = output_root / "images/val"
    train_lbl_dir = output_root / "labels/train"
    val_lbl_dir = output_root / "labels/val"
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(source_images.glob("*.png"))
    if not image_files:
        image_files = sorted(source_images.glob("*.jpg"))
    
    print(f"  Found {len(image_files)} images")
    
    # Filter to images that have corresponding labels
    valid_pairs = []
    for img_path in image_files:
        label_path = source_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    print(f"  Found {len(valid_pairs)} image-label pairs")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(valid_pairs)
    
    val_count = int(len(valid_pairs) * val_ratio)
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]
    
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Copy files
    for img_path, lbl_path in train_pairs:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(lbl_path, train_lbl_dir / lbl_path.name)
    
    for img_path, lbl_path in val_pairs:
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(lbl_path, val_lbl_dir / lbl_path.name)
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Train images: {train_img_dir}")
    print(f"   Val images: {val_img_dir}")
    
    return {
        "total": len(valid_pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
    }


def create_yolo_yaml(
    output_path: Path,
    data_root: Path,
    class_names: List[str],
) -> Path:
    """
    Create YOLO dataset configuration YAML file.
    """
    yaml_content = f"""# Polybag Dataset Configuration
# Auto-generated for YOLOv11-OBB training

path: {data_root}
train: images/train
val: images/val

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created YAML config: {output_path}")
    return output_path


def train_yolov11(
    data_yaml: Path,
    weights: Path,
    epochs: int = 100,
    imgsz: int = 640,
    device: int = 0,
    batch: int = 16,
    workers: int = 2,
    patience: int = 25,
    project: Path = None,
    name: str = "train",
    cache: str = "disk",  # 'disk', 'ram', or False
):
    """
    Train YOLOv11-OBB model.
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING YOLOV11-OBB TRAINING")
    print("=" * 60)
    print(f"  Dataset config:     {data_yaml}")
    print(f"  Pretrained weights: {weights}")
    print(f"  Epochs:             {epochs}")
    print(f"  Image size:         {imgsz}")
    print(f"  Batch size:         {batch}")
    print(f"  Workers:            {workers}")
    print(f"  Patience:           {patience}")
    print(f"  Device:             cuda:{device}")
    print(f"  Cache:              {cache}")
    print("=" * 60 + "\n")
    
    model = YOLO(str(weights))
    
    start_time = time.time()
    
    # Handle cache parameter
    cache_value = cache if cache in ("disk", "ram") else False
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        patience=patience,
        project=str(project) if project else None,
        name=name,
        cache=cache_value,
        verbose=True,
        save_period=10,
        exist_ok=True,      # Nominal batch size for loss normalization
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print("=" * 60)
    
    return results, model


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11-OBB on polybag dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of data loading workers (system max: 2)")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE, choices=["disk", "ram", "none"], 
                        help="Cache mode: 'disk' (default, saves to disk), 'ram' (faster but uses more memory), 'none' (no caching)")
    parser.add_argument("--skip-split", action="store_true", help="Skip data splitting (if already done)")
    args = parser.parse_args()
    
    print("\n" + "🎯" * 20)
    print("  YOLOv11-OBB TRAINING SCRIPT")
    print("🎯" * 20 + "\n")
    
    # Check GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    else:
        print("⚠️ WARNING: No GPU found, training will be slow!")
    
    # Step 1: Split dataset (if needed)
    if not args.skip_split:
        if IMAGES_TRAIN.exists() and len(list(IMAGES_TRAIN.glob("*.png"))) > 0:
            print(f"\n⚠️ Dataset already split! Skipping...")
            print(f"   Train images: {len(list(IMAGES_TRAIN.glob('*.png')))}")
            print(f"   Val images: {len(list(IMAGES_VAL.glob('*.png')))}")
        else:
            split_dataset(
                source_images=SOURCE_IMAGES,
                source_labels=SOURCE_LABELS,
                output_root=DATA_ROOT,
                val_ratio=0.2,
                seed=42,
            )
    else:
        print("\n⏭️ Skipping data split (--skip-split)")
    
    # Step 2: Create YAML config
    create_yolo_yaml(
        output_path=DATA_CONFIG,
        data_root=DATA_ROOT,
        class_names=["polybag"],
    )
    
    # Step 3: Create output directories
    OBB_RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Train
    cache_mode = args.cache if args.cache != "none" else False
    results, model = train_yolov11(
        data_yaml=DATA_CONFIG,
        weights=PRETRAINED_WEIGHTS,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=OBB_RUN_DIR,
        name="polybag_yolov11",
        cache=cache_mode,
    )
    
    # Find and print best weights path
    best_weights = sorted(OBB_RUN_DIR.glob("**/weights/best.pt"))
    if best_weights:
        print("\n" + "=" * 60)
        print("📦 TRAINED MODEL LOCATION:")
        print(f"   {best_weights[-1]}")
        print("\n   Use this path in the notebook to load the model:")
        print(f'   BEST_WEIGHTS = Path("{best_weights[-1]}")')
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
