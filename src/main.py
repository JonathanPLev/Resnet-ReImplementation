import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DATASET_CHOICE,
    DEVICE,
    DSB2018_TRAIN_ROOT,
    NUM_WORKERS,
    PHC_IMAGE_ROOT,
    PIN_MEMORY,
)
from dataset import (
    DataScienceBowlDataset,
    SegmentationDataset,
    transforms as dataset_transforms,
)
from train import train_u_net

# we are implementing the original U-net architecture for the PhC-C2DH-U373 dataset segmentation task.

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("Using GPU: ", torch.cuda.get_device_name(0))


def build_phc_loaders():
    mask_pattern = os.path.join(str(PHC_IMAGE_ROOT), "*_GT", "SEG", "man_seg*.tif")
    mask_paths = sorted(glob.glob(mask_pattern))

    rng = np.random.default_rng(seed=42)
    shuffled = mask_paths.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(0.2 * len(shuffled)))
    val_mask_paths = shuffled[:split_idx]
    train_mask_paths = shuffled[split_idx:]

    train_dataset = SegmentationDataset(
        image_root=str(PHC_IMAGE_ROOT),
        mask_paths=train_mask_paths,
        transforms=dataset_transforms,
    )
    val_dataset = SegmentationDataset(
        image_root=str(PHC_IMAGE_ROOT),
        mask_paths=val_mask_paths,
        transforms=dataset_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


def build_dsb_loaders():
    train_root = Path(DSB2018_TRAIN_ROOT)
    image_ids = sorted(
        [
            p.name
            for p in train_root.iterdir()
            if p.is_dir() and (p / "images").exists() and (p / "masks").exists()
        ]
    )
    if not image_ids:
        raise FileNotFoundError(
            f"No training folders found under {train_root}. "
            "Ensure the Kaggle stage1_train directory is extracted there."
        )

    rng = np.random.default_rng(seed=42)
    rng.shuffle(image_ids)
    split_idx = max(1, int(0.2 * len(image_ids)))
    val_ids = image_ids[:split_idx]
    train_ids = image_ids[split_idx:]

    train_dataset = DataScienceBowlDataset(
        dataset_root=train_root,
        image_ids=train_ids,
        transforms=dataset_transforms,
    )
    val_dataset = DataScienceBowlDataset(
        dataset_root=train_root,
        image_ids=val_ids,
        transforms=dataset_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


def run_phc_training():
    train_loader, val_loader = build_phc_loaders()
    train_u_net(train_loader, val_loader)


def run_dsb_training():
    train_loader, val_loader = build_dsb_loaders()
    train_u_net(train_loader, val_loader)


if __name__ == "__main__":
    if DATASET_CHOICE == "phc-u373":
        run_phc_training()
    elif DATASET_CHOICE == "data-science-bowl-2018":
        run_dsb_training()
    else:
        raise ValueError(
            f"Unsupported dataset '{DATASET_CHOICE}'. "
            "Use 'phc-u373' or 'data-science-bowl-2018'."
        )
