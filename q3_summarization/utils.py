"""Small helpers shared by the training, evaluation and visualisation scripts."""
import random
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, TransformerSummarizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device, enabled=True):
    """bf16 mixed precision on GPUs that support it, plain fp32 otherwise."""
    use = enabled and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use)


def load_model(run_dir, device, which="best"):
    """Load `best.pt` (default) or `last.pt` from a training run."""
    ckpt = torch.load(Path(run_dir) / f"{which}.pt", map_location=device, weights_only=False)
    model = TransformerSummarizer(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def plot_loss_curves(history, train_curve, path):
    """Training loss (fine-grained + averaged per evaluation) and validation loss vs. step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_curve:
        steps, losses = zip(*train_curve)
        ax.plot(steps, losses, color="tab:blue", alpha=0.25, linewidth=1, label="train (every log step)")
    if history:
        steps = [h["step"] for h in history]
        ax.plot(steps, [h["train_loss"] for h in history], "o-", color="tab:blue", label="train (mean since last eval)")
        ax.plot(steps, [h["val_loss"] for h in history], "o-", color="tab:orange", label="validation")
        best = min(history, key=lambda h: h["val_loss"])
        ax.axvline(best["step"], color="gray", linestyle="--", linewidth=1,
                   label=f"best checkpoint (step {best['step']}, val {best['val_loss']:.3f})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (per target token)")
    ax.set_title("Transformer summariser: training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
