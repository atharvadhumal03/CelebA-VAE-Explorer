import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


def get_transforms(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class CelebADataset(Dataset):
    """
    Reads CelebA from the Kaggle CSV layout:
      root/
        img_align_celeba/  ← flat folder of 202,599 .jpg files
        list_attr_celeba.csv
        list_eval_partition.csv

    split: 'train' (partition=0), 'valid' (partition=1), 'test' (partition=2)
    Attributes are returned as float32 tensors with values 0.0 / 1.0.
    """

    SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}

    def __init__(self, root: str, split: str = "train", image_size: int = 128):
        root = Path(root)
        self.img_dir = root / "img_align_celeba"
        self.transform = get_transforms(image_size)

        # Load partition file
        partition_df = pd.read_csv(root / "list_eval_partition.csv")
        split_id = self.SPLIT_MAP[split]
        split_mask = partition_df["partition"] == split_id

        # Load attribute file
        attr_df = pd.read_csv(root / "list_attr_celeba.csv")

        # Align on image_id and filter to split
        merged = attr_df[attr_df["image_id"].isin(
            partition_df.loc[split_mask, "image_id"]
        )].copy()
        merged = merged.set_index("image_id")
        # Reorder to match partition order so indices are stable
        ordered_ids = partition_df.loc[split_mask, "image_id"].values
        merged = merged.loc[ordered_ids]

        self.image_ids = merged.index.tolist()
        # Convert -1/1 → 0/1
        attr_values = merged.values  # [N, 40]
        self.attributes = torch.tensor(
            (attr_values > 0).astype("float32"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.image_ids[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), self.attributes[idx]


def get_celeba_datasets(root: str, image_size: int = 128):
    train = CelebADataset(root, split="train", image_size=image_size)
    val   = CelebADataset(root, split="valid", image_size=image_size)
    test  = CelebADataset(root, split="test",  image_size=image_size)
    return train, val, test


def get_dataloaders(
    root: str,
    image_size: int = 128,
    batch_size: int = 128,
    num_workers: int = 4,
    smoke_test: bool = False,
    smoke_n: int = 1000,
):
    train_ds, val_ds, test_ds = get_celeba_datasets(root, image_size)

    if smoke_test:
        train_ds = Subset(train_ds, range(min(smoke_n, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(smoke_n // 5, len(val_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
