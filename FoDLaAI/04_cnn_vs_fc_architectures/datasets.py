import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import get_vision_splits


def get_mnist_loaders(batch_size=64, val_ratio=0.1, seed=42, device="cpu", smoke=False):
    splits = get_vision_splits(
        "mnist",
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        device=device,
        smoke=smoke,
    )
    return splits["train"], splits["val"], splits["test"], splits


def get_cifar_loaders(batch_size=64, val_ratio=0.1, seed=42, device="cpu", smoke=False):
    splits = get_vision_splits(
        "cifar",
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        device=device,
        smoke=smoke,
    )
    return splits["train"], splits["val"], splits["test"], splits
