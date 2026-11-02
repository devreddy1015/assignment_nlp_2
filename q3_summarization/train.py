"""
Train the from-scratch Transformer summariser on CNN / Daily Mail.

- cross-entropy loss that ignores padding positions (ignore_index = pad id)
- Adam with beta = (0.9, 0.98), eps = 1e-9 and the paper's warm-up then decay schedule
      lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
- teacher forcing: the gold summary shifted right is the decoder input, so the whole summary is
  trained in one parallel pass; the causal mask stops each position from seeing later tokens
- validation loss after every epoch (or every --eval_every steps); early stopping after
  --patience evaluations without improvement; the best checkpoint is kept as best.pt

Outputs in runs/<run_name>/: best.pt, last.pt (for --resume), config.json, history.json,
loss_curves.png, train_summary.json

Examples:
    python train.py --run_name smoke --train_size 2000 --val_size 500 --epochs 1
    python train.py --run_name base --train_size 100000
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from data import SummarizationDataset, load_meta, make_collate
from model import ModelConfig, TransformerSummarizer, count_parameters
from utils import autocast_ctx, get_device, plot_loss_curves, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default="base")
    p.add_argument("--cache_dir", default="data_cache")
    p.add_argument("--runs_dir", default="runs")
    # data
    p.add_argument("--train_size", type=int, default=None, help="random training subset (default: all)")
    p.add_argument("--val_size", type=int, default=None, help="validation subset (default: all)")
    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_enc_layers", type=int, default=4)
    p.add_argument("--n_dec_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--lr_factor", type=float, default=1.0, help="multiplier on the paper's schedule")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    p.add_argument("--patience", type=int, default=3, help="evaluations without improvement before stopping")
    p.add_argument("--eval_every", type=int, default=0, help="validate every N steps (0 = once per epoch)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--no_amp", action="store_true", help="disable bf16 mixed precision")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="continue from runs/<run_name>/last.pt")
    return p.parse_args()


def noam_lambda(d_model, warmup):
    """lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5): linear warm-up, then 1/sqrt(step) decay."""
    def schedule(step):
        step = max(step, 1)
        return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
    return schedule


@torch.no_grad()
def validation_loss(model, loader, pad_id, device, amp):
    """Mean cross-entropy per (non-padding) target token over the whole loader."""
    model.eval()
    total, tokens = 0.0, 0
    for src, dec_in, tgt in loader:
        src, dec_in, tgt = src.to(device), dec_in.to(device), tgt.to(device)
        with autocast_ctx(device, amp):
            logits = model(src, dec_in)
        total += nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=pad_id, reduction="sum").item()
        tokens += (tgt != pad_id).sum().item()
    model.train()
    return total / tokens


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp = not args.no_amp
    meta = load_meta(args.cache_dir)
    pad_id = meta["pad_id"]
    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    train_ds = SummarizationDataset(args.cache_dir, "train", size=args.train_size, random_subset=True, seed=args.seed)
    val_ds = SummarizationDataset(args.cache_dir, "validation", size=args.val_size)
    collate = make_collate(pad_id)
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds, generator=shuffle_gen),
                              collate_fn=collate, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    # ---------------------------------------------------------------- model + optimiser
    cfg = ModelConfig(vocab_size=meta["vocab_size"], d_model=args.d_model, n_heads=args.n_heads,
                      n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers, d_ff=args.d_ff,
                      dropout=args.dropout, max_len=max(meta["src_len"], meta["tgt_len"] + 1), pad_id=pad_id)
    model = TransformerSummarizer(cfg).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_factor, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda(args.d_model, args.warmup))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)

    state = {"step": 0, "epoch": 0, "best_val": math.inf, "bad_evals": 0, "history": [], "train_curve": []}
    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        print(f"Resumed from step {state['step']} (epoch {state['epoch']})")

    with open(run_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "model_config": asdict(cfg), "parameters": n_params,
                   "train_examples": len(train_ds), "val_examples": len(val_ds), "device": str(device),
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}, f, indent=2)
    steps_per_epoch = len(train_loader) // args.accum
    print(f"Device: {device} | parameters: {n_params:,} | train: {len(train_ds):,} | val: {len(val_ds):,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")

    def save_best(val_loss):
        torch.save({"model": model.state_dict(), "model_config": asdict(cfg), "step": state["step"],
                    "val_loss": val_loss, "meta": meta}, run_dir / "best.pt")

    def save_last():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state, "model_config": asdict(cfg)},
                   run_dir / "last.pt")

    def run_validation(train_loss):
        """Evaluate, update early-stopping state and checkpoints. Returns True to stop."""
        val_loss = validation_loss(model, val_loader, pad_id, device, amp)
        improved = val_loss < state["best_val"]
        if improved:
            state["best_val"], state["bad_evals"] = val_loss, 0
            save_best(val_loss)
        else:
            state["bad_evals"] += 1
        state["history"].append({
            "step": state["step"], "epoch": round(state["step"] / steps_per_epoch, 2),
            "train_loss": train_loss, "val_loss": val_loss, "val_ppl": math.exp(val_loss),
            "lr": scheduler.get_last_lr()[0], "elapsed_min": (time.time() - t0) / 60 + elapsed_before,
        })
        print(f"[eval] step {state['step']:,} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"(ppl {math.exp(val_loss):.1f}) | {'saved best' if improved else 'no improvement'} "
              f"({state['bad_evals']}/{args.patience})", flush=True)
        save_last()
        with open(run_dir / "history.json", "w") as f:
            json.dump({"history": state["history"], "train_curve": state["train_curve"]}, f)
        plot_loss_curves(state["history"], state["train_curve"], run_dir / "loss_curves.png")
        return state["bad_evals"] >= args.patience

    # ---------------------------------------------------------------- training loop
    t0 = time.time()
    elapsed_before = state["history"][-1]["elapsed_min"] if state["history"] else 0.0
    model.train()
    early_stop = hit_max_steps = False
    while state["epoch"] < args.epochs and not (early_stop or hit_max_steps):
        shuffle_gen.manual_seed(args.seed + state["epoch"])   # reproducible order per epoch
        interval_loss = torch.zeros((), device=device)        # token-weighted sums since last eval
        interval_tokens = torch.zeros((), device=device)
        log_loss, log_tokens = 0.0, 0
        t_log = time.time()
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        for src, dec_in, tgt in train_loader:
            src = src.to(device, non_blocking=True)
            dec_in = dec_in.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with autocast_ctx(device, amp):
                logits = model(src, dec_in)   # teacher forcing: one parallel pass over the summary
            loss = criterion(logits.float().view(-1, logits.size(-1)), tgt.reshape(-1))
            (loss / args.accum).backward()

            n_tok = (tgt != pad_id).sum()
            interval_loss += loss.detach() * n_tok
            interval_tokens += n_tok
            micro += 1
            if micro % args.accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1

            if state["step"] % args.log_every == 0:
                avg = (interval_loss / interval_tokens).item()
                tok_s = (interval_tokens.item() - log_tokens) / (time.time() - t_log)
                # loss averaged over the last log window
                window = (interval_loss.item() - log_loss) / max(interval_tokens.item() - log_tokens, 1)
                log_loss, log_tokens, t_log = interval_loss.item(), interval_tokens.item(), time.time()
                state["train_curve"].append((state["step"], window))
                print(f"epoch {state['epoch'] + 1} | step {state['step']:,} | loss {window:.4f} "
                      f"(interval mean {avg:.4f}) | lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"{tok_s:,.0f} target tok/s | {(time.time() - t0) / 60:.1f} min", flush=True)

            if args.eval_every and state["step"] % args.eval_every == 0:
                early_stop = run_validation((interval_loss / interval_tokens).item())
                interval_loss.zero_(); interval_tokens.zero_()
                log_loss, log_tokens, t_log = 0.0, 0, time.time()
            if args.max_steps and state["step"] >= args.max_steps:
                hit_max_steps = True
            if early_stop or hit_max_steps:
                break

        if not (early_stop or hit_max_steps):
            state["epoch"] += 1   # completed epochs; a resume starts the next one
        if interval_tokens.item() > 0 and not early_stop:
            early_stop = run_validation((interval_loss / interval_tokens).item())
        else:
            save_last()

    summary = {"best_val_loss": state["best_val"], "best_val_ppl": math.exp(state["best_val"]),
               "steps": state["step"], "epochs_completed": state["epoch"],
               "early_stopped": state["bad_evals"] >= args.patience,
               "train_minutes": state["history"][-1]["elapsed_min"] if state["history"] else 0}
    with open(run_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best validation loss {state['best_val']:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
