import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from models import cnn_models, fc_models
from utils.training_utils import count_parameters, train_model
from utils.visualization_utils import plot_confusion_matrix, plot_model_comparison, plot_training_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import add_experiment_args, resolve_runtime
from common.data import get_vision_splits
from common.seed import set_seed
from common.stats import format_mean_std, mean_std, to_jsonable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("comparison.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _builders(dataset_name):
    if dataset_name == "mnist":
        return {
            "FC": lambda: fc_models.FCNetworkMNIST(),
            "SimpleCNN": lambda: cnn_models.SimpleCNN(),
            "ResCNN": lambda: cnn_models.CNNWithResidual(),
        }
    return {
        "FC": lambda: fc_models.FCNetworkCIFAR(),
        "CNN": lambda: cnn_models.CIFARCNN(),
        "ResCNN": lambda: cnn_models.ResidualCIFAR(),
    }


def compare_dataset(dataset_name, args, device, seeds):
    save_dir = f"results/{dataset_name}_comparison"
    os.makedirs(save_dir, exist_ok=True)
    builders = _builders(dataset_name)
    results = {}
    for name, builder in builders.items():
        seed_runs = {}
        for seed in seeds:
            logger.info(f"Training {name} on {dataset_name} seed={seed}")
            set_seed(seed)
            splits = get_vision_splits(
                dataset_name,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=seed,
                device=str(device),
                smoke=args.smoke,
            )
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
            seed_runs[seed] = {
                "history": history,
                "train_time": time.time() - start,
                "params": count_parameters(model),
                "val_acc": history["best_val_acc"],
                "test_acc": history["final_test_acc"],
            }
            plot_training_history(history, f"{name} - {dataset_name} seed {seed}", save_dir)
            if dataset_name == "cifar" and seed == seeds[0] and not args.smoke:
                preds, targets = [], []
                model.eval()
                with torch.no_grad():
                    for data, target in splits["test"]:
                        data, target = data.to(device), target.to(device)
                        preds.extend(model(data).argmax(dim=1).cpu().tolist())
                        targets.extend(target.cpu().tolist())
                plot_confusion_matrix(
                    targets, preds, CIFAR_CLASSES, f"{name} - CIFAR-10 Confusion Matrix", save_dir
                )
        test_mean, test_std = mean_std(run["test_acc"] for run in seed_runs.values())
        results[name] = {
            "seeds": seed_runs,
            "params": next(iter(seed_runs.values()))["params"],
            "test_acc_mean": test_mean,
            "test_acc_std": test_std,
        }
        logger.info(f"{name} test acc {format_mean_std(test_mean, test_std)}")

    histories = [next(iter(res["seeds"].values()))["history"] for res in results.values()]
    plot_model_comparison(
        histories,
        list(results.keys()),
        metric="val_accs",
        title=f"{dataset_name.upper()} Val Accuracy Comparison",
        save_dir=save_dir,
    )
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="CNN vs FC comparison")
    add_experiment_args(parser, default_epochs=10)
    parser.add_argument("--dataset", choices=["mnist", "cifar", "both"], default="both")
    args = parser.parse_args()
    if args.dataset == "cifar":
        args.batch_size = args.batch_size or 128
    device, seeds = resolve_runtime(args)
    datasets = ["mnist", "cifar"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        compare_dataset(name, args, device, seeds)


if __name__ == "__main__":
    main()
