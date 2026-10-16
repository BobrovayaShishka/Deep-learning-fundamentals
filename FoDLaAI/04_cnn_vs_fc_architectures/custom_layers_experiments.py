import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import cnn_models, custom_layers
from utils.training_utils import count_parameters, train_model
from utils.visualization_utils import plot_training_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable


def test_custom_layers(device):
    conv = custom_layers.CustomConv2d(3, 16, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    print("CustomConv output shape:", conv(x).shape)
    attn = custom_layers.AttentionBlock(64).to(device)
    print("Attention output shape:", attn(torch.randn(2, 64, 16, 16, device=device)).shape)
    print("Swish output:", custom_layers.Swish()(torch.randn(2, 64, device=device)).shape)
    pool = custom_layers.CustomMaxPool(2).to(device)
    print("CustomPool output shape:", pool(torch.randn(2, 3, 32, 32, device=device)).shape)


def _make_model(block):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.block1 = block(64, 128, stride=2)
            self.block2 = block(128, 256, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.pool(x)
            return self.fc(x.view(x.size(0), -1))

    return TestModel()


def residual_block_experiment(args, device, seeds):
    save_dir = "results/custom_layers/residual_blocks"
    os.makedirs(save_dir, exist_ok=True)
    blocks = {
        "Basic": cnn_models.ResidualBlock,
        "Bottleneck": custom_layers.BottleneckResidualBlock,
        "Wide": custom_layers.WideResidualBlock,
    }
    if args.smoke:
        blocks = {"Basic": cnn_models.ResidualBlock}
    results = {}
    for name, block in blocks.items():
        seed_runs = {}
        for seed in seeds:
            print(f"Training {name} residual block seed={seed}")
            set_seed(seed)
            splits = get_vision_splits(
                "cifar",
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
            model = _make_model(block).to(device)
            history = train_model(
                model,
                splits["train"],
                splits["val"],
                test_loader=splits["test"],
                epochs=args.epochs,
                device=device,
                max_batches=2 if args.smoke else None,
            )
            seed_runs[seed] = {
                "history": history,
                "params": count_parameters(model),
                "test_acc": history["final_test_acc"],
                "val_acc": history["best_val_acc"],
            }
            plot_training_history(history, f"{name} Residual Block Training seed {seed}", save_dir)
        mean, std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[name] = {
            "seeds": seed_runs,
            "params": next(iter(seed_runs.values()))["params"],
            "test_acc_mean": mean,
            "test_acc_std": std,
        }
        print(f"{name}: test acc {format_mean_std(mean, std)}")
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Custom CNN layers")
    add_experiment_args(parser, default_epochs=10)
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    test_custom_layers(device)
    residual_block_experiment(args, device, seeds)


if __name__ == "__main__":
    main()
