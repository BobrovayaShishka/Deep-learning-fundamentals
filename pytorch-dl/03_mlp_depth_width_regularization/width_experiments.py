import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable
from common.train import evaluate_test_once, train_select_on_val
from utils.model_utils import count_parameters, create_model, generate_width_config
from utils.visualization_utils import plot_comparison_bar, plot_heatmap, plot_training_history

RESULTS_DIR = "results/width_experiments"
PLOTS_DIR = "plots/width_experiments"


def _run_widths(dataset_name, widths, splits, device, args):
    config = generate_width_config(widths)
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
        "widths": widths,
        "history": history,
        "val_acc": best["val_acc"],
        "test_acc": test_metrics["test_acc"],
        "test_loss": test_metrics["test_loss"],
    }


def width_experiment(dataset_name, width_configs, config_names, args, device, seeds):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = {}
    for name, widths in zip(config_names, width_configs):
        seed_runs = {}
        for seed in seeds:
            print(f"\n=== {dataset_name} {name} {widths} seed={seed} ===")
            set_seed(seed)
            splits = get_vision_splits(
                dataset_name,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
            seed_runs[seed] = _run_widths(dataset_name, widths, splits, device, args)
        test_mean, test_std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[name] = {
            "params": next(iter(seed_runs.values()))["params"],
            "widths": widths,
            "seeds": seed_runs,
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
        }
        plot_training_history(
            next(iter(seed_runs.values()))["history"],
            f"{dataset_name} {name}",
            os.path.join(PLOTS_DIR, f"{dataset_name}_{name}.png"),
        )
        print(f"{name} test acc {format_mean_std(test_mean, test_std)}")

    names = list(results.keys())
    plot_comparison_bar(
        [results[n]["test_acc_mean"] for n in names],
        names,
        f"{dataset_name}: test accuracy by width",
        "Test accuracy",
        os.path.join(PLOTS_DIR, f"{dataset_name}_width_accuracy.png"),
        yerr=[results[n]["test_acc_std"] for n in names],
    )
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def grid_search_experiment(dataset_name, width_options, args, device, seeds, depth=3):
    n = len(width_options)
    acc_mean = np.zeros((n, n))
    acc_std = np.zeros((n, n))
    for i, w1 in enumerate(width_options):
        for j, w2 in enumerate(width_options):
            widths = [w1, w2, max(w2 // 2, 16)] if depth == 3 else [w1, w2]
            accs = []
            for seed in seeds:
                print(f"\n=== grid {dataset_name} {widths} seed={seed} ===")
                set_seed(seed)
                splits = get_vision_splits(
                    dataset_name,
                    batch_size=args.batch_size,
                    val_ratio=args.val_ratio,
                    seed=seed,
                    device=str(device),
                    smoke=args.smoke,
                )
                run = _run_widths(dataset_name, widths, splits, device, args)
                accs.append(run["test_acc"])
            mean, std = mean_std(accs)
            acc_mean[i, j] = mean
            acc_std[i, j] = std
            print(f"{widths} test acc {format_mean_std(mean, std)}")
    labels = [str(w) for w in width_options]
    plot_heatmap(
        acc_mean,
        labels,
        labels,
        f"{dataset_name}: grid test acc (mean over seeds)",
        os.path.join(PLOTS_DIR, f"{dataset_name}_grid_acc_heatmap.png"),
    )
    np.save(os.path.join(RESULTS_DIR, f"{dataset_name}_acc_mean.npy"), acc_mean)
    np.save(os.path.join(RESULTS_DIR, f"{dataset_name}_acc_std.npy"), acc_std)


def main():
    parser = argparse.ArgumentParser(description="MLP width experiments")
    add_experiment_args(parser, default_epochs=20)
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    width_configs = [[64, 32, 16], [256, 128, 64]] if args.smoke else [
        [64, 32, 16],
        [256, 128, 64],
        [1024, 512, 256],
        [2048, 1024, 512],
    ]
    names = ["narrow", "medium", "wide", "xwide"][: len(width_configs)]
    width_experiment(args.dataset, width_configs, names, args, device, seeds)
    if args.grid:
        options = [64, 128] if args.smoke else [64, 128, 256, 512]
        grid_search_experiment(args.dataset, options, args, device, seeds)


if __name__ == "__main__":
    main()
