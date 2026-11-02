"""
Lead-3 baseline: the first three sentences of the article, with no model at all.

Used to check that our ROUGE setup reproduces the assignment's reference numbers
(40.06 / 17.48 / 25.02 / 36.36 on the full test set) before comparing the Transformer to it.
Sentences are split with NLTK's punkt tokenizer and joined with "\n" for ROUGE-Lsum.

Example:
    python lead3_baseline.py
"""
import argparse
import json
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from data import load_cnn_dailymail
from evaluate import LEAD3_TEST, ROUGE_TYPES, print_table, score_rouge


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="outputs/lead3_test.json")
    args = p.parse_args()

    nltk.download("punkt_tab", quiet=True)
    raw = load_cnn_dailymail(args.split)
    predictions = ["\n".join(sent_tokenize(a)[:3]) for a in raw["article"]]
    rouge, _ = score_rouge(raw["highlights"], predictions)

    print_table([("Lead-3 (assignment)", LEAD3_TEST),
                 ("Lead-3 (our ROUGE setup)", {t: rouge[t]["f1"] for t in ROUGE_TYPES})])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "examples": len(predictions), "rouge": rouge}, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
