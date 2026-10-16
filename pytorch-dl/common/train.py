from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device,
    max_batches: Optional[int] = None,
    desc: str = "Eval",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def run_train_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    for data, target in tqdm(data_loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return total_loss / total, correct / total


def train_select_on_val(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int,
    lr: float,
    device,
    weight_decay: float = 0.0,
    max_batches: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, nn.Module], None]] = None,
):
    """Train on train, pick the best checkpoint by val accuracy, restore it."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_losses": [],
        "train_accs": [],
        "val_losses": [],
        "val_accs": [],
        "epoch_time": [],
    }
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss, train_acc = run_train_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=max_batches
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, max_batches=max_batches, desc="Val"
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f}"
        )
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch_callback is not None:
            epoch_callback(epoch, model)

    model.load_state_dict(best_state)
    return history, {"val_acc": best_val_acc, "val_loss": best_val_loss}


def evaluate_test_once(model, test_loader, device, max_batches: Optional[int] = None):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, max_batches=max_batches, desc="Test"
    )
    print(f"Final test | loss {test_loss:.4f} | acc {test_acc:.4f}")
    return {"test_loss": test_loss, "test_acc": test_acc}
