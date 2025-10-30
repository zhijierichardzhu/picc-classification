import argparse
import logging
from pathlib import Path
import random
import sys
from typing import Callable, Iterable
import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, Mean
from tqdm import tqdm
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR

from dataset import NUM_LABELS, create_dataset

logger = logging.getLogger(__name__)

feature_sizes = {
    "slowfast_r50": 2304,
    "x3d_m": 2048,
}


def create_network(name: str, num_classes: int=NUM_LABELS) -> nn.Module:
    assert name in feature_sizes, f"Model {name} not supported"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=True)
    model.blocks[-1].proj = nn.Linear(feature_sizes[name], num_classes)
    return model


def get_grad_norm(network: nn.Module):
    grads = [
        param.grad.detach().flatten()
        for param in network.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads).norm().item()


def create_optimizer(
        model: nn.Module,
        name: str,
        lr: float,
        momentum: float = 0.9
    ) -> Optimizer:
    if name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    return optimizer


def train_epoch(
        model: nn.Module,
        steps_per_epoch: int,
        dataloader_iter: Iterable,
        optimizer: Optimizer,
        criterion: nn.Module | Callable,
        writer: SummaryWriter,
        device: str | torch.device = "cpu",
        global_step=0,
        clip_grad: bool = False
    ):
    model.train()

    acc = MulticlassAccuracy(num_classes=NUM_LABELS).to(device)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=NUM_LABELS).to (device)
    mean_loss = Mean().to(device)

    for step in tqdm(range(steps_per_epoch)):
        batch = next(dataloader_iter)
        if isinstance(batch["video"], list):
            # SlowFast
            frames = [frame.to(device) for frame in batch["video"]]  # list: [slow_pathway, fast_pathway], each tensor is (B, C, T H, W)
        else:
            # X3D
            frames = batch["video"].to(device)  # tensor: (B, C, T, H, W)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(frames)
        loss: torch.Tensor = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        loss.backward()

        grad_norm = get_grad_norm(model)
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        mean_loss.update(torch.stack([loss] * labels.size(0)))
        acc.update(preds, labels)
        confusion_matrix.update(preds, labels)

        # Log metrics to TensorBoard per step
        writer.add_scalar("GradNorm/Train", grad_norm, global_step + step)
        writer.add_scalar("Loss/Train", loss.item(), global_step + step)

    return {
        'avg_loss': mean_loss.compute().item(),
        'accuracy': acc.compute().item(),
        'steps': global_step + steps_per_epoch,
        'confusion_matrix': confusion_matrix.compute().cpu()
    }


def val_epoch(
        model: nn.Module,
        dataloader: Iterable,
        criterion: nn.Module | Callable,
        epoch: int,
        writer: SummaryWriter,
        device: str | torch.device = "cpu"
    ):
    model.eval()
    mean_loss = Mean().to(device)

    acc = MulticlassAccuracy(num_classes=NUM_LABELS).to(device)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=NUM_LABELS).to(device)

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader.dataset)):  # Assume a batch size of 1 for validation
            if isinstance(batch["video"], list):
                frames = [frame.to(device) for frame in batch["video"]]  # list: [slow_pathway, fast_pathway], each tensor is (B, C, T, H, W)
            else:
                frames = batch["video"].to(device)  # shape: (B, C, T, H, W)
            labels = batch["label"].to(device)

            outputs = model(frames)
            preds = outputs.argmax(dim=1)

            loss = criterion(outputs, labels)

            mean_loss.update(torch.stack([loss] * labels.size(0)))
            acc.update(preds, labels)
            confusion_matrix.update(preds, labels)

    # Log metrics to TensorBoard per epoch
    writer.add_scalar("Loss/Validation", mean_loss.compute().item(), epoch)
    writer.add_scalar("Accuracy/Validation", acc.compute().item(), epoch)

    return {
        'avg_loss': mean_loss.compute().item(),
        'accuracy': acc.compute().item(),
        'confusion_matrix': confusion_matrix.compute().cpu()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal video classification trainer")

    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Name for the training run")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")

    # Data
    parser.add_argument("--batch-size", type=int, default=16)
    # Model
    parser.add_argument("--model", type=str, default="x3d_m", choices=feature_sizes.keys())

    # Optimizer
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD optimizer")
    parser.add_argument("--clip-grad", action="store_true", help="Whether to clip gradients")

    parser.add_argument("--use-lr-scheduler", action="store_true", help="Whether to use a learning rate scheduler")
    parser.add_argument("--lr-step-size", type=int, default=10, help="Step size for StepLR scheduler")

    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    logdir = Path(args.log_dir) / str(args.name)
    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logdir / "train.log"),
        ]
    )

    # Create dataloaders
    train_loader, val_loader = create_dataset(sample_pathways=args.model.startswith("slowfast"))
    train_loader_iter = iter(train_loader)

    # Create model and loss function
    model = create_network(args.model, NUM_LABELS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args.optimizer, args.lr, args.momentum)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1) if args.use_lr_scheduler else None

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=logdir.as_posix())

    best_val_acc = 0.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model=model,
            steps_per_epoch=args.steps_per_epoch,
            dataloader_iter=train_loader_iter,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            device=device,
            global_step=global_step,
            clip_grad=args.clip_grad
        )
        global_step = metrics['steps']
        logger.info(f"Epoch {epoch}/{args.epochs} - train_loss: {metrics['loss']:.4f} train_acc: {metrics['accuracy']:.4f}")

        if scheduler:
            scheduler.step()

        val_metrics = val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
            device=device
        )
        logger.info(f"Epoch {epoch}/{args.epochs} - val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint = {
                'model': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer': optimizer.state_dict(),
                'args': args,
                'epoch': epoch,
                'step': global_step,
                'train_accuracy': metrics['accuracy'],
                'val_accuracy': best_val_acc,
                'val_confusion_matrix': val_metrics['confusion_matrix']
            }
            torch.save(checkpoint, ckpt_dir / f"{args.name}.pth")
            logger.info(f"Saved best model to {(ckpt_dir / f'{args.name}.pth').as_posix()} (val_acc={best_val_acc:.4f})")

    # Close the TensorBoard writer
    writer.close()
