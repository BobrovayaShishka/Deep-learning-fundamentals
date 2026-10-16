"""Fine-tune pretrained models: select on val, score once on test, repeat over seeds."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable
from common.train import evaluate_test_once, train_select_on_val
from utils.datasets import CustomImageDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def replace_head(model_name, model, num_classes):
    if model_name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(model_name)
    return model


def build_model(model_name, num_classes, device, smoke=False):
    weights = None if smoke else "IMAGENET1K_V1"
    factories = {
        "resnet18": models.resnet18,
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v3_small": models.mobilenet_v3_small,
    }
    model = factories[model_name](weights=weights)
    return replace_head(model_name, model, num_classes).to(device)


def plot_training_history(history, model_name, out_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_training_history.png"))
    plt.close()


def make_loaders(data_root, batch_size, device):
    train_tf, eval_tf = build_transforms()
    train_ds = CustomImageDataset(os.path.join(data_root, "train"), transform=train_tf)
    val_ds = CustomImageDataset(os.path.join(data_root, "val"), transform=eval_tf)
    test_ds = CustomImageDataset(os.path.join(data_root, "test"), transform=eval_tf)
    pin = str(device).startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader, train_ds.get_class_names()


def run_finetune(args, device, seeds):
    out_dir = "results/training"
    os.makedirs(out_dir, exist_ok=True)
    data_root = args.data_root
    for split in ("train", "val", "test"):
        path = os.path.join(data_root, split)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing {path}. Put images into data/train, data/val and data/test."
            )
    train_loader, val_loader, test_loader, class_names = make_loaders(
        data_root, args.batch_size, device
    )
    model_names = ["resnet18"] if args.smoke else ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]
    results = {}
    for model_name in model_names:
        seed_runs = {}
        for seed in seeds:
            print(f"\n=== {model_name} seed={seed} ===")
            set_seed(seed)
            model = build_model(model_name, len(class_names), device, smoke=args.smoke)
            start = time.time()
            history, best = train_select_on_val(
                model,
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                max_batches=2 if args.smoke else None,
            )
            test_metrics = evaluate_test_once(
                model, test_loader, device, max_batches=2 if args.smoke else None
            )
            seed_runs[seed] = {
                "history": history,
                "train_time": time.time() - start,
                "val_acc": best["val_acc"],
                "test_acc": test_metrics["test_acc"],
                "test_loss": test_metrics["test_loss"],
            }
            torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}_seed{seed}_best.pth"))
            plot_training_history(history, f"{model_name}_seed{seed}", out_dir)
        mean, std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[model_name] = {
            "seeds": seed_runs,
            "test_acc_mean": mean,
            "test_acc_std": std,
        }
        print(f"{model_name} test acc {format_mean_std(mean, std)}")

    names = list(results.keys())
    plt.figure(figsize=(10, 6))
    plt.bar(
        names,
        [results[n]["test_acc_mean"] for n in names],
        yerr=[results[n]["test_acc_std"] for n in names],
        capsize=4,
        color="skyblue",
    )
    plt.title("Model Comparison: Test Accuracy (mean ± std)")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(out_dir, "model_comparison_acc.png"))
    plt.close()
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained models")
    add_experiment_args(parser, default_epochs=10)
    parser.add_argument("--data-root", default="data")
    parser.set_defaults(batch_size=32, lr=1e-3)
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    run_finetune(args, device, seeds)


if __name__ == "__main__":
    main()
