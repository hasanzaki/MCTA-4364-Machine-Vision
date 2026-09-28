"""Deep-learning helpers for the Machine Vision notebooks (Week 6 onwards).

Usage (after the setup cell):

    import sys
    sys.path.append("resources/scripts")
    from dlhelpers import hw_profile, fit, plot_history

Everything here is sized for a laptop with a 4 GB NVIDIA GPU, and falls back
to the CPU when no GPU is present.
"""

import platform
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from cvhelpers import SMOKE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hw_profile(verbose=True):
    """Detect the hardware and suggest safe settings for it.

    Returns a dict with `device`, `cuda`, `gpu_name`, `vram_gb`, `tier`
    ("cpu", "gpu-4gb" or "gpu"), `amp` (mixed precision on/off),
    `num_workers` (0 on Windows, where worker processes are fragile in
    notebooks) and `batch_scale` (multiply your base batch size by this).
    """
    cuda = torch.cuda.is_available()
    info = {"cuda": cuda, "device": torch.device("cuda" if cuda else "cpu"),
            "gpu_name": None, "vram_gb": 0.0, "smoke": SMOKE}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = props.total_memory / 1024 ** 3
    if not cuda:
        info.update(tier="cpu", amp=False, batch_scale=0.5)
    elif info["vram_gb"] < 6:
        info.update(tier="gpu-4gb", amp=True, batch_scale=1.0)
    else:
        info.update(tier="gpu", amp=True, batch_scale=2.0)
    info["num_workers"] = 0 if platform.system() == "Windows" else 2
    if verbose:
        print(f"PyTorch {torch.__version__} | device: {info['device']}", end="")
        if cuda:
            print(f" | {info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM)", end="")
        print(f" | tier: {info['tier']}" + (" | SMOKE mode (CI)" if SMOKE else ""))
        if not cuda:
            print("No NVIDIA GPU found: the notebook still runs on the CPU, "
                  "with smaller training runs.")
    return info


def set_seed(seed=0):
    """Make runs repeatable (up to GPU non-determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def layer_table(model, input_shape, device="cpu"):
    """Print each leaf layer with its output shape and parameter count."""
    rows, hooks = [], []

    def hook(module, _inp, out):
        shape = tuple(out.shape) if torch.is_tensor(out) else "-"
        rows.append((module.__class__.__name__, shape,
                     sum(p.numel() for p in module.parameters(recurse=False))))

    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    print(f"{'Layer':<20}{'Output shape':<24}{'Params':>10}")
    print("-" * 54)
    for name, shape, n in rows:
        print(f"{name:<20}{str(shape):<24}{n:>10,}")
    print("-" * 54)
    print(f"{'Total':<44}{count_params(model):>10,}")


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates.

    Returns (mean loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, optimizer, epochs, device,
        loss_fn=None, scheduler=None, amp=False, verbose=True):
    """Train for `epochs` and return a history dict of losses and accuracies."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if amp and device.type == "cuda" else None
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        vl, va = run_epoch(model, val_loader, loss_fn, device)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)):
            hist[k].append(v)
        if verbose:
            print(f"epoch {epoch:2d}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | "
                  f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.1f}s")
    return hist


def plot_history(histories, title="Training curves"):
    """Plot loss and accuracy curves. `histories` is one history or {name: history}."""
    if "train_loss" in histories:
        histories = {"": histories}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (name, h), color in zip(histories.items(), plt.cm.tab10.colors):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        label = f"{name} " if name else ""
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(ep, h[f"train_{key}"], "--", marker="o", ms=3, color=color, label=f"{label}train")
            ax.plot(ep, h[f"val_{key}"], "-", marker="o", ms=4, color=color, lw=2, label=f"{label}val")
    for ax, ylabel in zip(axes, ("Cross-entropy loss", "Accuracy")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict(model, loader, device):
    """Return (logits, labels) for a whole loader as NumPy arrays."""
    model.eval()
    outs, labels = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(outs).numpy(), torch.cat(labels).numpy()


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_confusion(cm, class_names, title="Confusion matrix", ax=None):
    """Row-normalised confusion matrix (each row sums to 1)."""
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    if own:
        plt.tight_layout()
        plt.show()


def load_pretrained(builder, weights):
    """Build a torchvision model with pretrained weights.

    Returns (model, pretrained_ok). Without internet access the model is built
    with random weights and a warning, so the rest of the notebook still runs.
    """
    try:
        return builder(weights=weights), True
    except Exception as exc:  # noqa: BLE001 - any download failure
        print(f"WARNING: could not download pretrained weights ({type(exc).__name__}). "
              "Using random weights: results below will NOT be meaningful.")
        try:  # detection/segmentation builders also fetch a backbone by default
            return builder(weights=None, weights_backbone=None), False
        except TypeError:
            return builder(weights=None), False


def denormalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo ImageNet normalisation of a CHW tensor and return an HWC array in [0, 1]."""
    t = t.detach().cpu().float() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return t.clamp(0, 1).permute(1, 2, 0).numpy()
