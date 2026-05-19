import os
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CelebA


def get_transforms(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_celeba_datasets(root: str, image_size: int = 128):
    tf = get_transforms(image_size)
    train = CelebA(root=root, split="train", target_type="attr", transform=tf, download=False)
    val   = CelebA(root=root, split="valid", target_type="attr", transform=tf, download=False)
    test  = CelebA(root=root, split="test",  target_type="attr", transform=tf, download=False)
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
