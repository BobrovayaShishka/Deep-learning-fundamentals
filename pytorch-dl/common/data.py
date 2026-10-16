from __future__ import annotations

from typing import Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split


MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class TensorClassificationDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _subset(dataset: Dataset, n: int, seed: int) -> Subset:
    n = min(n, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def _split_train_val(dataset: Dataset, val_ratio: float, seed: int):
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, device: str):
    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_vision_splits(
    dataset_name: str,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    smoke: bool = False,
    data_root: str = "./data",
):
    name = dataset_name.lower()
    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        )
        full_train = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
        input_size, channels, image_size = 28 * 28, 1, 28
    elif name in {"cifar", "cifar10"}:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
        )
        full_train = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
        input_size, channels, image_size = 32 * 32 * 3, 3, 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if smoke:
        full_train = _subset(full_train, 128, seed)
        test_ds = _subset(test_ds, 64, seed)
        val_ratio = 0.25

    train_ds, val_ds = _split_train_val(full_train, val_ratio, seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size=min(batch_size, len(train_ds)), device=device
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": 10,
        "input_size": input_size,
        "channels": channels,
        "image_size": image_size,
    }


def make_synthetic_splits(
    n_train: int = 64,
    n_val: int = 32,
    n_test: int = 32,
    channels: int = 1,
    image_size: int = 28,
    num_classes: int = 10,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cpu",
):
    generator = torch.Generator().manual_seed(seed)

    def _make(n):
        images = torch.randn(n, channels, image_size, image_size, generator=generator)
        labels = torch.randint(0, num_classes, (n,), generator=generator)
        return TensorClassificationDataset(images, labels)

    train_loader, val_loader, test_loader = make_loaders(
        _make(n_train), _make(n_val), _make(n_test), batch_size=batch_size, device=device
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": num_classes,
        "input_size": channels * image_size * image_size,
        "channels": channels,
        "image_size": image_size,
    }
