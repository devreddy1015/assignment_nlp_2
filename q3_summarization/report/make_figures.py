"""
Collect the figures used in the report into report/figures/.

- token-length histograms of the training split, with the truncation limits
- the learning-rate schedule used in training (same function as train.py)
- copies of the loss curves (train.py) and attention heatmaps (visualize_attention.py), if they
  exist yet

Run from q3_summarization/:
    python report/make_figures.py --run_dir runs/base
"""
import argparse
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


def lr_schedule(out, d_model=256, warmup=4000, steps=60000, steps_per_epoch=8972):
    schedule = noam_lambda(d_model, warmup)
    x = np.arange(1, steps + 1)
    y = [schedule(s) for s in x]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, color="tab:blue")
    ax.axvline(warmup, color="gray", linestyle="--", linewidth=1,
               label=f"end of warm-up (step {warmup:,}, lr {schedule(warmup):.2e})")
    for e in range(1, steps // steps_per_epoch + 1):
        ax.axvline(e * steps_per_epoch, color="lightgray", linewidth=0.8)
    ax.set_xlabel(f"optimizer step (light lines = epoch boundaries, {steps_per_epoch:,} steps each)")
    ax.set_ylabel("learning rate")
    ax.set_title(f"lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), d_model={d_model}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="runs/base")
    p.add_argument("--cache_dir", default="data_cache")
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    length_histograms(args.cache_dir, out_dir / "length_histograms.png")
    lr_schedule(out_dir / "lr_schedule.png")

    run_dir = Path(args.run_dir)
    copies = [run_dir / "loss_curves.png", *sorted((run_dir / "attention").glob("*.png"))]
    for f in copies:
        if f.exists():
            shutil.copy(f, out_dir / f.name)
    print(f"Figures in {out_dir}: {', '.join(sorted(f.name for f in out_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
