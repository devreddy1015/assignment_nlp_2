"""
Download CNN / Daily Mail 3.0.0, tokenize every split once and cache the result.

Writes to --cache_dir (default data_cache/):
    {split}_src.npy      (N, SRC_LEN)     uint16 article ids, padded
    {split}_tgt.npy      (N, TGT_LEN + 1) uint16 [start] summary [end], padded
    {split}_index.npy    (N,)             row of each example in the Hugging Face split
    {split}_src_len.npy / {split}_tgt_len.npy   untruncated token counts (for the report)
    meta.json            special-token ids, lengths and dataset statistics
    tokenizer/           the tokenizer actually used (bert-base-uncased + [NL])

Empty articles/summaries are dropped from train and validation only; the test split is kept
complete (11,490 pairs) so ROUGE is reported on the full test set.

Example:
    python prepare_data.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers.utils import logging as hf_logging

from data import (NL_TOKEN, SRC_LEN, TGT_LEN, TOKENIZER_NAME, clean_summary, clean_text,
                  load_cnn_dailymail, load_tokenizer)


def length_stats(lengths, limit):
    return {
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": int(lengths.max()),
        "pct_truncated": round(100.0 * float((lengths > limit).mean()), 2),
    }


def prepare_split(ds, tok, meta, split, out_dir, drop_empty, chunk=2000):
    n = len(ds)
    src = np.zeros((n, SRC_LEN), dtype=np.uint16)      # pad id is 0, so zeros = padding
    tgt = np.zeros((n, TGT_LEN + 1), dtype=np.uint16)
    src_len = np.zeros(n, dtype=np.int32)
    tgt_len = np.zeros(n, dtype=np.int32)
    keep = np.ones(n, dtype=bool)

    for start in range(0, n, chunk):
        rows = ds[start:start + chunk]
        articles = [clean_text(a) for a in rows["article"]]
        summaries = [clean_summary(h) for h in rows["highlights"]]
        art_ids = tok(articles, add_special_tokens=False)["input_ids"]
        sum_ids = tok(summaries, add_special_tokens=False)["input_ids"]
        for j, (a, s) in enumerate(zip(art_ids, sum_ids)):
            i = start + j
            src_len[i], tgt_len[i] = len(a), len(s)
            if drop_empty and (len(a) == 0 or len(s) == 0):
                keep[i] = False
                continue
            a = a[:SRC_LEN]
            s = [meta["bos_id"]] + s[:TGT_LEN - 1] + [meta["eos_id"]]
            src[i, :len(a)] = a
            tgt[i, :len(s)] = s
        print(f"  {split}: {min(start + chunk, n):,}/{n:,}", end="\r", flush=True)
    print()

    idx = np.nonzero(keep)[0]
    np.save(out_dir / f"{split}_src.npy", src[idx])
    np.save(out_dir / f"{split}_tgt.npy", tgt[idx])
    np.save(out_dir / f"{split}_index.npy", idx)
    np.save(out_dir / f"{split}_src_len.npy", src_len[idx])
    np.save(out_dir / f"{split}_tgt_len.npy", tgt_len[idx])
    return {
        "examples": int(len(idx)),
        "dropped_empty": int(n - len(idx)),
        "article_tokens": length_stats(src_len[idx], SRC_LEN),
        "summary_tokens": length_stats(tgt_len[idx], TGT_LEN - 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default="data_cache")
    args = parser.parse_args()

    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_logging.set_verbosity_error()  # silence "sequence longer than 512" warnings; we truncate ourselves

    tok = load_tokenizer()
    tok.save_pretrained(out_dir / "tokenizer")
    meta = {
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": len(tok),
        "pad_id": tok.pad_token_id,
        "bos_id": tok.cls_token_id,   # [CLS] is used as the summary start token
        "eos_id": tok.sep_token_id,   # [SEP] is used as the summary end token
        "nl_id": tok.convert_tokens_to_ids(NL_TOKEN),
        "src_len": SRC_LEN,
        "tgt_len": TGT_LEN,
        "splits": {},
    }
    assert meta["pad_id"] == 0 and meta["vocab_size"] < 2 ** 16

    ds = load_cnn_dailymail()
    for split in ["train", "validation", "test"]:
        print(f"Tokenizing {split} ({len(ds[split]):,} pairs)")
        meta["splits"][split] = prepare_split(ds[split], tok, meta, split, out_dir, drop_empty=split != "test")
        print(json.dumps(meta["splits"][split], indent=2))

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
