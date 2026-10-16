import argparse
import json
import os
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable
from common.train import evaluate_test_once, train_select_on_val
from utils.model_utils import count_parameters, create_model, generate_config
from utils.visualization_utils import plot_comparison_bar, plot_training_history

RESULTS_DIR = "results/regularization_experiments"
PLOTS_DIR = "plots/regularization_experiments"
BASE_WIDTHS = [1024, 512, 256]


def _experiments(smoke: bool):
    configs = [
        {"name": "no_reg", "config": generate_config(BASE_WIDTHS)},
        {"name": "dropout_0.1", "config": generate_config(BASE_WIDTHS, dropout_rate=0.1)},
        {"name": "dropout_0.3", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3)},
        {"name": "dropout_0.5", "config": generate_config(BASE_WIDTHS, dropout_rate=0.5)},
        {"name": "batchnorm", "config": generate_config(BASE_WIDTHS, use_batchnorm=True)},
        {"name": "both", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True)},
        {"name": "l2_1e-4", "config": generate_config(BASE_WIDTHS, weight_decay=1e-4)},
        {"name": "l2_1e-3", "config": generate_config(BASE_WIDTHS, weight_decay=1e-3)},
        {"name": "l2_1e-2", "config": generate_config(BASE_WIDTHS, weight_decay=1e-2)},
    ]
    if smoke:
        return configs[:2]
    return configs


def run_one(exp, splits, device, args):
    model = create_model(exp["config"], splits["input_size"], splits["num_classes"]).to(device)
    if exp["config"].get("use_batchnorm", False):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = exp["config"].get("bn_momentum", 0.1)
    history, best = train_select_on_val(
        model,
        splits["train"],
        splits["val"],
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        weight_decay=exp["config"].get("weight_decay", 0.0),
        max_batches=2 if args.smoke else None,
    )
    test_metrics = evaluate_test_once(
        model, splits["test"], device, max_batches=2 if args.smoke else None
    )
    return {
        "params": count_parameters(model),
        "config": exp["config"],
        "history": history,
        "val_acc": best["val_acc"],
        "test_acc": test_metrics["test_acc"],
        "test_loss": test_metrics["test_loss"],
    }


def regularization_experiment(dataset_name, args, device, seeds):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = {}
    for exp in _experiments(args.smoke):
        seed_runs = {}
        for seed in seeds:
            print(f"\n=== {dataset_name} {exp['name']} seed={seed} ===")
            set_seed(seed)
            splits = get_vision_splits(
                dataset_name,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
            seed_runs[seed] = run_one(exp, splits, device, args)
        test_mean, test_std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[exp["name"]] = {
            "params": next(iter(seed_runs.values()))["params"],
            "config": exp["config"],
            "seeds": seed_runs,
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
        }
        plot_training_history(
            next(iter(seed_runs.values()))["history"],
            f"{dataset_name} {exp['name']}",
            os.path.join(PLOTS_DIR, f"{dataset_name}_{exp['name']}.png"),
        )
        print(f"{exp['name']} test acc {format_mean_std(test_mean, test_std)}")

    names = list(results.keys())
    plot_comparison_bar(
        [results[n]["test_acc_mean"] for n in names],
        names,
        f"{dataset_name}: regularization test accuracy",
        "Test accuracy",
        os.path.join(PLOTS_DIR, f"{dataset_name}_comparison.png"),
        yerr=[results[n]["test_acc_std"] for n in names],
    )
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="MLP regularization experiments")
    add_experiment_args(parser, default_epochs=30)
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    args = parser.parse_args()
    device, seeds = resolve_runtime(args)
    regularization_experiment(args.dataset, args, device, seeds)


if __name__ == "__main__":
    main()
