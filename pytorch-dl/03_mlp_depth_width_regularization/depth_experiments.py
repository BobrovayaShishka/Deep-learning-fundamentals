import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable
from common.train import evaluate_test_once, train_select_on_val
from utils.model_utils import count_parameters, create_model, generate_depth_config
from utils.visualization_utils import plot_comparison_bar, plot_training_history

RESULTS_DIR = "results/depth_experiments"
PLOTS_DIR = "plots/depth_experiments"
HIDDEN_SIZE = 256


def _loaders(dataset_name, args, seed, device):
    return get_vision_splits(
        dataset_name,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=seed,
        device=str(device),
        smoke=args.smoke,
    )


def run_one(dataset_name, depth, splits, device, args, use_dropout, use_batchnorm):
    config = generate_depth_config(depth, HIDDEN_SIZE, use_dropout, use_batchnorm)
    model = create_model(config, splits["input_size"], splits["num_classes"]).to(device)
    params = count_parameters(model)
    history, best = train_select_on_val(
        model,
        splits["train"],
        splits["val"],
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        max_batches=2 if args.smoke else None,
    )
    test_metrics = evaluate_test_once(
        model, splits["test"], device, max_batches=2 if args.smoke else None
    )
    return {
        "params": params,
        "config": config,
        "history": history,
        "val_acc": best["val_acc"],
        "test_acc": test_metrics["test_acc"],
        "test_loss": test_metrics["test_loss"],
    }


def run_depth_experiment(dataset_name, depths, args, device, seeds, use_dropout=False, use_batchnorm=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    tag = "reg" if use_dropout or use_batchnorm else "base"
    results = {}
    for depth in depths:
        seed_runs = {}
        for seed in seeds:
            print(f"\n=== {dataset_name} depth={depth} seed={seed} ({tag}) ===")
            set_seed(seed)
            splits = _loaders(dataset_name, args, seed, device)
            seed_runs[seed] = run_one(
                dataset_name, depth, splits, device, args, use_dropout, use_batchnorm
            )
        test_mean, test_std = mean_std(run["test_acc"] for run in seed_runs.values())
        val_mean, val_std = mean_std(run["val_acc"] for run in seed_runs.values())
        plot_training_history(
            next(iter(seed_runs.values()))["history"],
            f"{dataset_name} depth {depth} (seed {seeds[0]})",
            os.path.join(PLOTS_DIR, f"{dataset_name}_{tag}_depth_{depth}.png"),
        )
        results[str(depth)] = {
            "params": next(iter(seed_runs.values()))["params"],
            "config": next(iter(seed_runs.values()))["config"],
            "seeds": seed_runs,
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
            "val_acc_mean": val_mean,
            "val_acc_std": val_std,
        }
        print(f"Depth {depth} test acc {format_mean_std(test_mean, test_std)}")

    names = list(results.keys())
    plot_comparison_bar(
        [results[d]["test_acc_mean"] for d in names],
        names,
        f"{dataset_name}: test accuracy by depth ({tag})",
        "Test accuracy",
        os.path.join(PLOTS_DIR, f"{dataset_name}_{tag}_depth_accuracy.png"),
        yerr=[results[d]["test_acc_std"] for d in names],
    )
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{tag}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f"Saved {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MLP depth experiments")
    add_experiment_args(parser, default_epochs=20)
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    depths = [1, 2] if args.smoke else [1, 2, 3, 5, 7]
    run_depth_experiment(args.dataset, depths, args, device, seeds)
    if not args.smoke:
        run_depth_experiment(
            args.dataset, [5, 7], args, device, seeds, use_dropout=True, use_batchnorm=True
        )


if __name__ == "__main__":
    main()
