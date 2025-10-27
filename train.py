import argparse
import os
from pathlib import Path

import torch
from torch import nn
from dataset import create_dataset, NUM_LABELS
import random
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

feature_sizes = {
    "slowfast_r50": 2304
}


def train_epoch(
        model: nn.Module,
        steps_per_epoch: int,
        dataloader_iter,
        optimizer,
        criterion,
        writer,
        device: str | torch.device = "cpu",
        global_step=0
    ):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for step in tqdm(range(steps_per_epoch)):
        batch = next(dataloader_iter)
        frames = [frame.to(device) for frame in batch["video"]]  # list: [slow_pathway, fast_pathway], each tensor is (B, C, T H, W)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log metrics to TensorBoard per step
        writer.add_scalar("Loss/Train", loss.item(), global_step + step)
        writer.add_scalar("Accuracy/Train", (preds == labels).float().mean().item(), global_step + step)

    avg_loss = running_loss / total if total else 0.0
    acc = running_correct / total if total else 0.0
    return avg_loss, acc, global_step + steps_per_epoch


def val_epoch(model, dataloader, criterion, writer, device, global_step=0):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            frames = [frame.to(device) for frame in batch["video"]]  # list: [slow_pathway, fast_pathway], each tensor is (B, T, C, H, W)
            labels = batch["label"].to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Log metrics to TensorBoard per step
            writer.add_scalar("Loss/Validation", loss.item(), global_step + step)
            writer.add_scalar("Accuracy/Validation", (preds == labels).float().mean().item(), global_step + step)

    avg_loss = running_loss / total if total else 0.0
    acc = running_correct / total if total else 0.0
    return avg_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal video classification trainer")
    parser.add_argument("--steps-per-epoch", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="model.pt")
    parser.add_argument("--model", type=str, default="slowfast_r50")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Create dataloaders
    train_loader, val_loader = create_dataset()
    train_loader_iter = iter(train_loader)

    # Create model
    model = torch.hub.load('facebookresearch/pytorchvideo', args.model, pretrained=False)
    model.blocks[6].proj = nn.Linear(feature_sizes[args.model], NUM_LABELS)

    criterion = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_acc = 0.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, global_step = train_epoch(model, args.steps_per_epoch, train_loader_iter, optimizer, criterion, writer, device, global_step)
        logger.info(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}")

        val_loss, val_acc = val_epoch(model, val_loader, criterion, writer, device, global_step)
        logger.info(f"Epoch {epoch}/{args.epochs} - val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'model': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'args': args,
                },
                args.save_path
            )
            logger.info(f"Saved best model to {args.save_path} (val_acc={best_val_acc:.4f})")

    # Close the TensorBoard writer
    writer.close()
