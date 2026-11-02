"""
Heatmap of the encoder-decoder (cross) attention while the model generates a summary.

- the attention is taken from the LAST decoder layer and recorded during greedy generation
  (not during teacher-forced training): row t is what the model attended to when it produced
  generated token t
- heads are averaged by default (--per_head also draws every head separately)
- rows = generated summary tokens, columns = source article tokens; because a 512-token article
  is unreadable, only a window of --window source tokens is shown: the window that receives the
  most attention

Writes runs/<run>/attention/attention_<split><index>.png (+ _heads.png with --per_head) and the
raw weights as .npz.

Example:
    python visualize_attention.py --run_dir runs/base --index 0 --per_head
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import SummarizationDataset, ids_to_text, load_cnn_dailymail, load_meta, load_tokenizer
from utils import get_device, load_model


def best_window(attn, width):
    """Start column of the `width`-wide source window that holds the most attention mass."""
    column_mass = attn.sum(axis=0)
    if len(column_mass) <= width:
        return 0
    window_mass = np.convolve(column_mass, np.ones(width), mode="valid")
    return int(window_mass.argmax())


def draw(ax, attn, x_labels, y_labels, title):
    im = ax.imshow(attn, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("source article tokens")
    ax.set_ylabel("generated summary tokens")
    ax.set_title(title, fontsize=9)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cache_dir", default="data_cache")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--index", type=int, default=0, help="example number within the split")
    p.add_argument("--window", type=int, default=60, help="number of source tokens shown")
    p.add_argument("--max_rows", type=int, default=40, help="number of generated tokens shown")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--per_head", action="store_true", help="also plot every head separately")
    args = p.parse_args()

    device = get_device()
    meta = load_meta(args.cache_dir)
    tok = load_tokenizer(Path(args.cache_dir) / "tokenizer")
    model, _ = load_model(args.run_dir, device, "best")

    ds = SummarizationDataset(args.cache_dir, args.split)
    src_row, _ = ds[args.index]
    src_ids = [int(i) for i in src_row if i != meta["pad_id"]]
    src = torch.tensor([src_ids], device=device)

    # fp32, no dropout (eval mode): weights recorded at every generation step
    gen, attn = model.greedy_decode(src, meta["bos_id"], meta["eos_id"], max_len=args.max_len,
                                    return_attention=True)
    gen_ids = gen[0].tolist()
    length = gen_ids.index(meta["eos_id"]) + 1 if meta["eos_id"] in gen_ids else len(gen_ids)
    gen_ids = gen_ids[:length]
    attn = attn[0, :, :length, :].numpy()            # (H, L, S)
    n_heads = attn.shape[0]

    rows = min(args.max_rows, length)
    mean_attn = attn.mean(axis=0)[:rows]              # average over heads: (rows, S)
    start = best_window(mean_attn, args.window)
    end = min(start + args.window, len(src_ids))
    mass = mean_attn[:, start:end].sum() / mean_attn.sum()

    src_tokens = tok.convert_ids_to_tokens(src_ids[start:end])
    gen_tokens = tok.convert_ids_to_tokens(gen_ids[:rows])

    out_dir = Path(args.run_dir) / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"attention_{args.split}{args.index}"

    fig, ax = plt.subplots(figsize=(max(8, 0.17 * (end - start) + 2), max(5, 0.2 * rows + 2)))
    im = draw(ax, mean_attn[:, start:end], src_tokens, gen_tokens,
              f"Last decoder layer cross-attention, mean of {n_heads} heads, recorded during greedy generation\n"
              f"{args.split} example {args.index}: source tokens {start}-{end - 1} of {len(src_ids)} "
              f"({mass:.0%} of these rows' attention falls inside this window)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)

    if args.per_head:
        cols = 2
        fig, axes = plt.subplots((n_heads + cols - 1) // cols, cols,
                                 figsize=(2 * max(7, 0.15 * (end - start) + 2), (n_heads + 1) // 2 * max(4.5, 0.18 * rows + 2)))
        for h, ax in enumerate(np.atleast_1d(axes).ravel()):
            if h < n_heads:
                draw(ax, attn[h, :rows, start:end], src_tokens, gen_tokens, f"head {h + 1}")
            else:
                ax.axis("off")
        fig.suptitle(f"Last decoder layer cross-attention per head ({args.split} example {args.index})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_heads.png", dpi=120)
        plt.close(fig)

    np.savez_compressed(out_dir / f"{stem}.npz", attention=attn, source_ids=np.array(src_ids),
                        generated_ids=np.array(gen_ids))
    raw = load_cnn_dailymail(args.split)[int(ds.hf_index[args.index])]
    info = {"index": args.index, "window": [start, end], "attention_mass_in_window": float(mass),
            "generated": ids_to_text(gen_ids, tok, meta), "reference": raw["highlights"]}
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Generated:\n{info['generated']}\n\nReference:\n{info['reference']}\n")
    print(f"Saved {out_dir / (stem + '.png')}" + (f" and {stem}_heads.png" if args.per_head else ""))


if __name__ == "__main__":
    main()
