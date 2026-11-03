"""
Collect the figures used in the report into report/figures/.

- token-length histograms of the training split, with the truncation limits
- the learning-rate schedules of the training runs (same function as train.py)
- validation loss per epoch of every run, on one plot
- copies of each run's loss curves (train.py) and attention heatmaps (visualize_attention.py),
  prefixed with the run name, e.g. figures/base_loss_curves.png

Run from q3_summarization/:
    python report/make_figures.py --runs base lr05_b64
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SRC_LEN, TGT_LEN  # noqa: E402
from train import noam_lambda  # noqa: E402

RUN_LABELS = {"base": "run 1: paper schedule, batch 32",
              "lr05_b64": "run 2: 0.5 x schedule, batch 64"}


def length_histograms(cache_dir, out):
    src_len = np.load(Path(cache_dir) / "train_src_len.npy")
    tgt_len = np.load(Path(cache_dir) / "train_tgt_len.npy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # whole-token bin widths (25 and 2) so the histogram has no aliasing spikes
    for ax, lengths, limit, clip, width, name in [(axes[0], src_len, SRC_LEN, 2500, 25, "Articles"),
                                                  (axes[1], tgt_len, TGT_LEN - 1, 250, 2, "Summaries")]:
        ax.hist(np.minimum(lengths, clip), bins=np.arange(0, clip + width + 1, width), color="tab:blue", alpha=0.8)
        ax.axvline(limit, color="tab:red", linestyle="--",
                   label=f"truncation at {limit} tokens ({(lengths > limit).mean():.1%} longer)")
        ax.axvline(np.median(lengths), color="black", linewidth=1, label=f"median {np.median(lengths):.0f}")
        ax.set_title(f"{name} (training split, {len(lengths):,} pairs)")
        ax.set_xlabel(f"WordPiece tokens (values above {clip:,} shown at {clip:,})")
        ax.set_ylabel("number of examples")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def lr_schedules(run_dirs, out, epochs=20):
    """Learning rate against epochs (not steps), so runs with gradient accumulation line up."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        a = cfg["args"]
        steps_per_epoch = cfg["train_examples"] // a["batch_size"] // a["accum"]
        schedule = noam_lambda(a["d_model"], a["warmup"])
        steps = np.arange(1, epochs * steps_per_epoch + 1)
        lr = [a["lr_factor"] * schedule(s) for s in steps]
        peak = a["lr_factor"] * schedule(a["warmup"])
        ax.plot(steps / steps_per_epoch, lr,
                label=f"{RUN_LABELS.get(run_dir.name, run_dir.name)} (peak {peak:.2e} at step {a['warmup']:,})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("lr = factor x d_model^-0.5 x min(step^-0.5, step x warmup^-1.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def validation_comparison(run_dirs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for run_dir in run_dirs:
        hist = json.loads((run_dir / "history.json").read_text())["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_loss"] for h in hist], "o-", label=f"{RUN_LABELS.get(run_dir.name, run_dir.name)}: validation")
        ax.plot(epochs, [h["train_loss"] for h in hist], "x--", alpha=0.6, color=ax.lines[-1].get_color(),
                label=f"{RUN_LABELS.get(run_dir.name, run_dir.name)}: training (epoch mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss (per target token)")
    ax.set_title("Training and validation loss of both runs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["base", "lr05_b64"], help="run names under runs/")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--cache_dir", default="data_cache")
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    run_dirs = [Path(args.runs_dir) / r for r in args.runs if (Path(args.runs_dir) / r / "history.json").exists()]

    length_histograms(args.cache_dir, out_dir / "length_histograms.png")
    lr_schedules(run_dirs, out_dir / "lr_schedule.png")
    validation_comparison(run_dirs, out_dir / "loss_comparison.png")
    for run_dir in run_dirs:
        for f in [run_dir / "loss_curves.png", *sorted((run_dir / "attention").glob("*.png"))]:
            if f.exists():
                shutil.copy(f, out_dir / f"{run_dir.name}_{f.name}")
    print(f"Figures in {out_dir}: {', '.join(sorted(f.name for f in out_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
