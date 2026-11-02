"""
Generate summaries with greedy decoding and score them with ROUGE.

- no teacher forcing: decoding starts from the start token and feeds every predicted token back
  in, until the end token or --max_len tokens
- each article is encoded once and the encoder output is reused at every decoding step
- token ids are converted back to text; [NL] becomes "\n" so ROUGE-Lsum sees one sentence per line
- ROUGE F1 with the rouge_score package (use_stemmer=True) against the original highlights:
  ROUGE-1, ROUGE-2, ROUGE-L (summary = one sequence) and ROUGE-Lsum (sentence level)
- uses best.pt (the best validation checkpoint), never the last one

Writes to runs/<run>/eval_<split>[_first<N>]/: rouge.json, predictions.jsonl

Examples:
    python evaluate.py --run_dir runs/base                 # full test set (11,490 articles)
    python evaluate.py --run_dir runs/base --limit 200     # quick check on the first 200
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SummarizationDataset, ids_to_text, load_cnn_dailymail, load_meta, load_tokenizer, make_collate
from utils import autocast_ctx, get_device, load_model

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}
LEAD3_TEST = {"rouge1": 40.06, "rouge2": 17.48, "rougeL": 25.02, "rougeLsum": 36.36}  # from the assignment


def has_repeated_trigram(text):
    words = text.split()
    trigrams = list(zip(words, words[1:], words[2:]))
    return len(trigrams) != len(set(trigrams))


def generate_summaries(model, loader, tok, meta, device, max_len, amp):
    predictions, hit_max_len = [], []
    for src, _, _ in tqdm(loader, desc="greedy decoding"):
        with autocast_ctx(device, amp):
            out = model.greedy_decode(src.to(device), meta["bos_id"], meta["eos_id"], max_len=max_len)
        for row in out.cpu().tolist():
            predictions.append(ids_to_text(row, tok, meta))
            hit_max_len.append(meta["eos_id"] not in row)
    return predictions, hit_max_len


def score_rouge(references, predictions):
    """Per-example F1 for every ROUGE type, plus mean precision/recall/F1 (x100)."""
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    per_example = [scorer.score(ref, pred) for ref, pred in
                   tqdm(zip(references, predictions), total=len(references), desc="ROUGE")]
    summary = {t: {"precision": 100 * float(np.mean([s[t].precision for s in per_example])),
                   "recall": 100 * float(np.mean([s[t].recall for s in per_example])),
                   "f1": 100 * float(np.mean([s[t].fmeasure for s in per_example]))}
               for t in ROUGE_TYPES}
    return summary, per_example


def print_table(rows):
    print(f"\n{'System':<32}" + "".join(f"{LABELS[t]:>12}" for t in ROUGE_TYPES))
    for name, scores in rows:
        print(f"{name:<32}" + "".join(f"{scores[t]:>12.2f}" for t in ROUGE_TYPES))
    print("(F1 x 100, rouge_score with stemming; ROUGE-L = whole summary as one sequence, "
          "ROUGE-Lsum = sentence level, sentences split on newlines)\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cache_dir", default="data_cache")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--limit", type=int, default=None, help="only the first N articles (default: all)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=128, help="maximum number of generated tokens")
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    device = get_device()
    meta = load_meta(args.cache_dir)
    tok = load_tokenizer(Path(args.cache_dir) / "tokenizer")
    model, ckpt = load_model(args.run_dir, device, "best")
    print(f"Loaded best checkpoint from step {ckpt['step']:,} (val loss {ckpt['val_loss']:.4f})")

    ds = SummarizationDataset(args.cache_dir, args.split, size=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(meta["pad_id"]))
    raw = load_cnn_dailymail(args.split)
    hf_rows = ds.hf_index[ds.rows].tolist()
    subset = raw.select(hf_rows)
    references = subset["highlights"]   # original text with "\n" between sentences
    example_ids = subset["id"]

    t0 = time.time()
    predictions, hit_max_len = generate_summaries(model, loader, tok, meta, device, args.max_len, not args.no_amp)
    gen_minutes = (time.time() - t0) / 60
    rouge, per_example = score_rouge(references, predictions)

    stats = {
        "examples": len(predictions),
        "decoding": f"greedy, max {args.max_len} tokens",
        "generation_minutes": round(gen_minutes, 1),
        "avg_words_generated": float(np.mean([len(p.split()) for p in predictions])),
        "avg_words_reference": float(np.mean([len(r.split()) for r in references])),
        "avg_sentences_generated": float(np.mean([len(p.split("\n")) for p in predictions])),
        "pct_hit_max_len_without_end_token": 100 * float(np.mean(hit_max_len)),
        "pct_with_repeated_trigram": 100 * float(np.mean([has_repeated_trigram(p) for p in predictions])),
        "pct_empty": 100 * float(np.mean([not p.strip() for p in predictions])),
    }
    out_dir = Path(args.run_dir) / (f"eval_{args.split}" + (f"_first{args.limit}" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rouge.json", "w") as f:
        json.dump({"checkpoint_step": ckpt["step"], "rouge": rouge, "stats": stats}, f, indent=2)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for i, ex_id, ref, pred, s, hit in zip(hf_rows, example_ids, references, predictions, per_example, hit_max_len):
            f.write(json.dumps({"index": i, "id": ex_id, "reference": ref, "prediction": pred,
                                **{t: s[t].fmeasure for t in ROUGE_TYPES}, "hit_max_len": hit}) + "\n")

    rows = [("Transformer (greedy)", {t: rouge[t]["f1"] for t in ROUGE_TYPES})]
    if args.split == "test" and not args.limit:
        rows.insert(0, ("Lead-3 baseline (assignment)", LEAD3_TEST))
    print_table(rows)
    print(json.dumps(stats, indent=2))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
