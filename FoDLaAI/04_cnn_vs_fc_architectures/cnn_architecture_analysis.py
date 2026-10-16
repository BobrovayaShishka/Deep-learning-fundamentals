import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import cnn_models
from utils.training_utils import count_parameters, train_model
from utils.visualization_utils import (
    ensure_dir,
    plot_activations,
    plot_gradients,
    plot_model_comparison,
    plot_receptive_field,
    plot_training_history,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("architecture_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_gradients(model, data_loader, criterion, device):
    model.train()
    gradients = {}
    hooks = []

    def hook_fn(module, grad_input, grad_output, name):
        if grad_output[0] is not None:
            gradients[name] = grad_output[0].abs().mean().item()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(
                module.register_full_backward_hook(
                    lambda module, grad_input, grad_output, name=name: hook_fn(
                        module, grad_input, grad_output, name
                    )
                )
            )
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    model.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    for hook in hooks:
        hook.remove()
    return gradients


def _train_builder(builder, splits, args, device):
    model = builder().to(device)
    start = time.time()
    history = train_model(
        model,
        splits["train"],
        splits["val"],
        test_loader=splits["test"],
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        max_batches=2 if args.smoke else None,
    )
    return model, history, time.time() - start


def kernel_size_experiment(args, device, seeds):
    save_dir = "results/architecture_analysis/kernel_size"
    ensure_dir(save_dir)
    configs = [
        {"name": "3x3", "kernels": [3, 3, 3]},
        {"name": "5x5", "kernels": [5, 5, 5]},
        {"name": "7x7", "kernels": [7, 7, 7]},
        {"name": "1x1+3x3", "kernels": [1, 3, 3]},
    ]
    if args.smoke:
        configs = configs[:2]
    results = {}
    for cfg in configs:
        seed_runs = {}
        for seed in seeds:
            logger.info(f"Kernel {cfg['name']} seed={seed}")
            set_seed(seed)
            splits = get_vision_splits(
                "cifar",
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
            model, history, train_time = _train_builder(
                lambda: cnn_models.KernelSizeCNN(cfg["kernels"]), splits, args, device
            )
            seed_runs[seed] = {
                "history": history,
                "train_time": train_time,
                "params": count_parameters(model),
                "test_acc": history["final_test_acc"],
                "val_acc": history["best_val_acc"],
            }
            if seed == seeds[0]:
                rf_sizes, current_rf, stride_product = [], 1, 1
                for kernel in cfg["kernels"]:
                    current_rf = current_rf + (kernel - 1) * stride_product
                    rf_sizes.append(current_rf)
                    stride_product *= 2
                plot_receptive_field(rf_sizes, f"{cfg['name']} Kernel Receptive Field", save_dir)
                activations = []
                hook = model.conv1.register_forward_hook(
                    lambda module, inputs, output: activations.append(output.detach().cpu())
                )
                data, _ = next(iter(splits["train"]))
                with torch.no_grad():
                    model(data[:1].to(device))
                hook.remove()
                plot_activations(activations[0][0].numpy(), f"{cfg['name']} First Layer Activations", save_dir)
                plot_training_history(history, f"{cfg['name']} Kernel - Training History", save_dir)
        mean, std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[cfg["name"]] = {
            "seeds": seed_runs,
            "params": next(iter(seed_runs.values()))["params"],
            "test_acc_mean": mean,
            "test_acc_std": std,
        }
        logger.info(f"{cfg['name']} test acc {format_mean_std(mean, std)}")
    plot_model_comparison(
        [next(iter(res["seeds"].values()))["history"] for res in results.values()],
        list(results.keys()),
        title="Kernel Size Comparison (Val Accuracy)",
        save_dir=save_dir,
    )
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def depth_experiment(args, device, seeds):
    save_dir = "results/architecture_analysis/depth"
    ensure_dir(save_dir)
    configs = [
        {"name": "Shallow (2 blocks)", "depth": 2, "residual": False},
        {"name": "Medium (4 blocks)", "depth": 4, "residual": False},
        {"name": "Deep (6 blocks)", "depth": 6, "residual": False},
        {"name": "Residual (6 blocks)", "depth": 6, "residual": True},
    ]
    if args.smoke:
        configs = configs[:2]
    results = {}
    for cfg in configs:
        seed_runs = {}
        for seed in seeds:
            logger.info(f"Depth {cfg['name']} seed={seed}")
            set_seed(seed)
            splits = get_vision_splits(
                "cifar",
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
            model, history, train_time = _train_builder(
                lambda: cnn_models.DepthCNN(num_blocks=cfg["depth"], use_residual=cfg["residual"]),
                splits,
                args,
                device,
            )
            grads = get_gradients(model, splits["train"], nn.CrossEntropyLoss(), device)
            seed_runs[seed] = {
                "history": history,
                "train_time": train_time,
                "params": count_parameters(model),
                "test_acc": history["final_test_acc"],
                "val_acc": history["best_val_acc"],
                "grad_means": list(grads.values()),
            }
            if seed == seeds[0]:
                plot_gradients(list(grads.values()), list(grads.keys()), f"{cfg['name']} Gradient Analysis", save_dir)
                plot_training_history(history, f"{cfg['name']} - Training History", save_dir)
        mean, std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[cfg["name"]] = {
            "seeds": seed_runs,
            "params": next(iter(seed_runs.values()))["params"],
            "test_acc_mean": mean,
            "test_acc_std": std,
            "grad_means": next(iter(seed_runs.values()))["grad_means"],
        }
        logger.info(f"{cfg['name']} test acc {format_mean_std(mean, std)}")
    plot_model_comparison(
        [next(iter(res["seeds"].values()))["history"] for res in results.values()],
        list(results.keys()),
        title="Network Depth Comparison (Val Accuracy)",
        save_dir=save_dir,
    )
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        plt.plot(res["grad_means"], label=name, marker="o")
    plt.title("Gradient Magnitude Comparison")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Absolute Gradient (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_comparison.png"))
    plt.close()
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="CNN architecture analysis")
    add_experiment_args(parser, default_epochs=15)
    parser.add_argument("--task", choices=["kernel", "depth", "both"], default="both")
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    if args.task in {"kernel", "both"}:
        kernel_size_experiment(args, device, seeds)
    if args.task in {"depth", "both"}:
        if not args.smoke:
            args.epochs = max(args.epochs, 20)
        depth_experiment(args, device, seeds)


if __name__ == "__main__":
    main()
