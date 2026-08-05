"""
Train the EmotionCNN on FER2013.

Usage:
    python train.py --data-dir /path/to/fer2013 [--epochs 100] [--batch-size 128]

Expects --data-dir to contain train/ and test/ subfolders (see dataset.py
for the exact layout).
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_data_loaders
from engine import evaluate_model, plot_history, train_model
from model import EmotionCNN, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ and test/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_fer2013_model.pth")

    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    plot_history(history, args.results_dir)

    # Evaluate the best checkpoint, not whatever's in memory after the last epoch
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint saved (val accuracy never improved) - evaluating last epoch instead.")

    test_acc, _ = evaluate_model(model, test_loader, device, args.results_dir)

    final_path = os.path.join(args.checkpoint_dir, "fer2013_final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
            "history": history,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
