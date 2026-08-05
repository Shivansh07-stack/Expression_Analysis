"""Training loop, evaluation, and plotting helpers."""

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import EXPRESSIONS


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    checkpoint_path,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:5.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:5.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"No improvement in {patience} epochs, stopping early at epoch {epoch}.")
            break

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return history


@torch.no_grad()
def evaluate_model(model, test_loader, device, results_dir):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    for images, labels in tqdm(test_loader, desc="test", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    test_acc = 100.0 * correct / total
    class_names = [e.capitalize() for e in EXPRESSIONS]

    print(f"\nTest accuracy: {test_acc:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - FER2013 Test Set", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    return test_acc, cm


def plot_history(history, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_acc"], label="Train", linewidth=2)
    axes[0].plot(history["val_acc"], label="Validation", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train", linewidth=2)
    axes[1].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["lr"], linewidth=2, color="green")
    axes[2].set_yscale("log")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"), dpi=300)
    plt.close()

    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
