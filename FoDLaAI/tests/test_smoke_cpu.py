"""CPU smoke tests: train/val/test split, val selection, one-shot test metric."""

import matplotlib

matplotlib.use("Agg")

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import make_synthetic_splits
from common.seed import set_seed
from common.stats import mean_std
from common.train import evaluate_test_once, train_select_on_val


class TinyMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(channels, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 8)
        return self.fc(x.flatten(1))


class SmokeCPUTest(unittest.TestCase):
    def test_mlp_selects_on_val_and_scores_test_once(self):
        set_seed(42)
        splits = make_synthetic_splits(n_train=64, n_val=32, n_test=32, batch_size=16, seed=42)
        model = TinyMLP(splits["input_size"], splits["num_classes"])
        history, best = train_select_on_val(
            model, splits["train"], splits["val"], epochs=1, lr=1e-2, device="cpu"
        )
        test_metrics = evaluate_test_once(model, splits["test"], "cpu")
        self.assertEqual(len(history["val_acc"]), 1)
        self.assertNotIn("test_acc", history)
        self.assertGreaterEqual(best["val_acc"], 0.0)
        self.assertLessEqual(best["val_acc"], 1.0)
        self.assertGreaterEqual(test_metrics["test_acc"], 0.0)
        self.assertLessEqual(test_metrics["test_acc"], 1.0)

    def test_cnn_three_seeds_mean_std(self):
        accs = []
        for seed in (42, 43, 44):
            set_seed(seed)
            splits = make_synthetic_splits(
                n_train=48,
                n_val=24,
                n_test=24,
                channels=3,
                image_size=16,
                batch_size=8,
                seed=seed,
            )
            model = TinyCNN(3, 10)
            train_select_on_val(
                model, splits["train"], splits["val"], epochs=1, lr=1e-2, device="cpu"
            )
            accs.append(evaluate_test_once(model, splits["test"], "cpu")["test_acc"])
        mean, std = mean_std(accs)
        self.assertEqual(len(accs), 3)
        self.assertTrue(0.0 <= mean <= 1.0)
        self.assertGreaterEqual(std, 0.0)

    def test_cuda_not_required_for_smoke(self):
        self.assertEqual(str(torch.device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
