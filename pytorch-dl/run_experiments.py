"""One-command runner: smoke test on CPU, or CUDA/CPU experiment suites."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_cmd(args, cwd: Path):
    print(f"\n>>> {' '.join(args)}  (cwd={cwd})")
    result = subprocess.run(args, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with code {result.returncode}: {' '.join(args)}")


def run_smoke():
    print("Running CPU smoke tests...")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_smoke*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("Smoke tests failed")
    print("Smoke tests passed.")


EXPERIMENTS = {
    "mlp_depth_mnist": (
        ROOT / "03_mlp_depth_width_regularization",
        ["depth_experiments.py", "--dataset", "mnist"],
    ),
    "mlp_depth_cifar": (
        ROOT / "03_mlp_depth_width_regularization",
        ["depth_experiments.py", "--dataset", "cifar"],
    ),
    "mlp_width_mnist": (
        ROOT / "03_mlp_depth_width_regularization",
        ["width_experiments.py", "--dataset", "mnist"],
    ),
    "mlp_width_cifar": (
        ROOT / "03_mlp_depth_width_regularization",
        ["width_experiments.py", "--dataset", "cifar"],
    ),
    "mlp_reg_mnist": (
        ROOT / "03_mlp_depth_width_regularization",
        ["regularization_experiments.py", "--dataset", "mnist"],
    ),
    "mlp_reg_cifar": (
        ROOT / "03_mlp_depth_width_regularization",
        ["regularization_experiments.py", "--dataset", "cifar"],
    ),
    "cnn_vs_fc": (
        ROOT / "04_cnn_vs_fc_architectures",
        ["cnn_vs_fc_comparison.py", "--dataset", "both"],
    ),
    "cnn_arch": (
        ROOT / "04_cnn_vs_fc_architectures",
        ["cnn_architecture_analysis.py", "--task", "both"],
    ),
    "custom_layers": (
        ROOT / "04_cnn_vs_fc_architectures",
        ["custom_layers_experiments.py"],
    ),
    "finetune": (
        ROOT / "05_augmentations_and_transfer_learning",
        ["finetune_pretrained.py"],
    ),
}

DEFAULT_SUITE = [
    "mlp_depth_mnist",
    "mlp_depth_cifar",
    "mlp_width_mnist",
    "mlp_width_cifar",
    "mlp_reg_mnist",
    "mlp_reg_cifar",
    "cnn_vs_fc",
    "cnn_arch",
    "custom_layers",
]


def main():
    parser = argparse.ArgumentParser(description="Run pytorch-dl experiments")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "smoke"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--experiments", default=",".join(DEFAULT_SUITE))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.command == "smoke" or args.smoke:
        run_smoke()
        return

    names = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}. Choose from {list(EXPERIMENTS)}")

    extra = ["--device", args.device, "--seeds", args.seeds]
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]

    for name in names:
        cwd, script_args = EXPERIMENTS[name]
        if name == "finetune" and not (cwd / "data" / "train").is_dir():
            print(f"Skipping {name}: no data/train directory")
            continue
        run_cmd([PY, *script_args, *extra], cwd)


if __name__ == "__main__":
    main()
