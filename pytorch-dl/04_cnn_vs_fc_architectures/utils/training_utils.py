import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.train import evaluate, evaluate_test_once, run_train_epoch, train_select_on_val


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def run_epoch(model, data_loader, criterion, optimizer=None, device="cpu", is_test=False):
    if is_test or optimizer is None:
        return evaluate(model, data_loader, criterion, device, desc="Eval")
    return run_train_epoch(model, data_loader, criterion, optimizer, device)


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=10,
    lr=0.001,
    device="cpu",
    weight_decay=0.0,
    max_batches=None,
):
    history, best = train_select_on_val(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
        max_batches=max_batches,
    )
    history["test_losses"] = history["val_losses"]
    history["test_accs"] = history["val_accs"]
    history["best_val_acc"] = best["val_acc"]
    if test_loader is not None:
        test_metrics = evaluate_test_once(model, test_loader, device, max_batches=max_batches)
        history["final_test_loss"] = test_metrics["test_loss"]
        history["final_test_acc"] = test_metrics["test_acc"]
    return history
