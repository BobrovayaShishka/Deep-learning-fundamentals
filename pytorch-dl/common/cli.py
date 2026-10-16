from __future__ import annotations

import argparse

import torch

from .seed import parse_seeds


def add_experiment_args(parser: argparse.ArgumentParser, default_epochs: int = 10) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cuda:0 or cpu",
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds, one run each")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--smoke", action="store_true", help="CPU-friendly tiny run")
    return parser


def resolve_runtime(args):
    if args.smoke:
        args.device = "cpu"
        args.seeds = "42"
        args.epochs = min(getattr(args, "epochs", 1), 1)
        args.batch_size = min(getattr(args, "batch_size", 32), 16)
    device = torch.device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    seeds = parse_seeds(args.seeds)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Seeds: {seeds} | epochs: {args.epochs} | batch_size: {args.batch_size}")
    return device, seeds
