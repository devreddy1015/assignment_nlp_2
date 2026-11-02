"""
Data loading and preprocessing for CNN / Daily Mail (version 3.0.0).

Preprocessing pipeline (run once with `python prepare_data.py`):
  1. lowercase the text and collapse whitespace (the tokenizer is uncased, so the two match)
  2. the "\n" between highlight sentences becomes a special [NL] token, so sentence breaks
     survive tokenization and can be turned back into "\n" for ROUGE-Lsum
  3. sub-word tokenization with the pre-trained bert-base-uncased WordPiece tokenizer;
     one shared vocabulary for articles and summaries
  4. truncate articles to SRC_LEN tokens and summaries to TGT_LEN - 1 tokens
  5. every summary gets a start token ([CLS]) and an end token ([SEP])
  6. pad to a fixed length and store as uint16 .npy arrays (small and fast to load)

At batch time `make_collate` trims the padding and builds
  decoder input = summary shifted right (starts with the start token)
  target        = summary shifted left  (ends with the end token)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKENIZER_NAME = "bert-base-uncased"
NL_TOKEN = "[NL]"       # sentence separator inside summaries
SRC_LEN = 512           # article truncation length (tokens)
TGT_LEN = 128           # summary length seen by the decoder (start/end token included)

# The assignment's load_dataset("cnn_dailymail", "3.0.0"). Recent huggingface_hub versions only
# accept namespaced ids, and the canonical repo of the same dataset is abisee/cnn_dailymail.
DATASET_NAME, DATASET_VERSION = "abisee/cnn_dailymail", "3.0.0"


def load_cnn_dailymail(split=None):
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, DATASET_VERSION, split=split)


def load_tokenizer(path=TOKENIZER_NAME):
    """The pre-trained WordPiece tokenizer plus the [NL] sentence-separator token."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if NL_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [NL_TOKEN]})
    return tok


def clean_text(text):
    """Lowercase and collapse all whitespace (including newlines) to single spaces."""
    return " ".join(text.lower().split())


def clean_summary(highlights):
    """Keep the highlight sentence boundaries as [NL] tokens."""
    sentences = [clean_text(s) for s in highlights.split("\n")]
    return f" {NL_TOKEN} ".join(s for s in sentences if s)


def ids_to_text(ids, tok, meta):
    """Token ids -> summary text. Stops at the end token, drops padding and turns [NL] back
    into "\n" so that ROUGE-Lsum sees one sentence per line."""
    sentences, current = [], []
    for i in ids:
        if i == meta["eos_id"]:
            break
        if i == meta["pad_id"]:
            continue
        if i == meta["nl_id"]:
            sentences.append(current)
            current = []
        else:
            current.append(i)
    sentences.append(current)
    texts = [tok.decode(s, skip_special_tokens=True).strip() for s in sentences]
    return "\n".join(t for t in texts if t)


def load_meta(cache_dir):
    with open(Path(cache_dir) / "meta.json") as f:
        return json.load(f)


class SummarizationDataset(Dataset):
    """Pre-tokenized split stored as memory-mapped .npy arrays.

    size: use only this many examples. With random_subset=True they are drawn at random with
    `seed` (for training subsets); otherwise the first `size` examples are used.
    """

    def __init__(self, cache_dir, split, size=None, random_subset=False, seed=0):
        cache_dir = Path(cache_dir)
        self.src = np.load(cache_dir / f"{split}_src.npy", mmap_mode="r")
        self.tgt = np.load(cache_dir / f"{split}_tgt.npy", mmap_mode="r")
        self.hf_index = np.load(cache_dir / f"{split}_index.npy")  # row in the HF split
        n = len(self.src)
        if size is None or size >= n:
            self.rows = np.arange(n)
        elif random_subset:
            self.rows = np.sort(np.random.default_rng(seed).choice(n, size, replace=False))
        else:
            self.rows = np.arange(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.src[r], self.tgt[r]


def _trim(x, pad_id):
    """Drop the padding columns that every row of the batch has (padding is on the right)."""
    length = max(int((x != pad_id).sum(dim=1).max()), 1)
    return x[:, :length]


def make_collate(pad_id):
    def collate(batch):
        src = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.int64))
        tgt = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.int64))
        src, tgt = _trim(src, pad_id), _trim(tgt, pad_id)
        # tgt = [start, w1, ..., wn, end]
        dec_in = tgt[:, :-1]   # [start, w1, ..., wn]  (shifted right)
        target = tgt[:, 1:]    # [w1, ..., wn, end]    (shifted left)
        return src, dec_in, target

    return collate
