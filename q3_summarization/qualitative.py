"""
Pick test examples for the qualitative analysis and write them to a Markdown file.

Reads the predictions written by evaluate.py and selects typical, best and worst examples plus
failure cases (repeated phrases, numbers that are not in the article). Each example gets
automatic flags to help the analysis:
  - repeated word trigrams in the generated summary
  - numbers in the generated summary that do not appear in the article (wrong numbers)
  - words in the generated summary that never occur in the article (possible invented facts)
The flags are hints only; read the article before calling something an error.

Example:
    python qualitative.py --eval_dir runs/base/eval_test
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from data import load_cnn_dailymail

# digit runs only: WordPiece decoding can turn "1,000" into "1, 000", so compare the pieces
NUMBER = re.compile(r"\d+")
WORD = re.compile(r"[a-z]+")


def repeated_trigrams(text):
    words = text.lower().split()
    counts = Counter(zip(words, words[1:], words[2:]))
    return [" ".join(t) for t, c in counts.items() if c > 1]


def numbers_not_in_article(pred, article):
    in_article = set(NUMBER.findall(article))
    return sorted({n for n in NUMBER.findall(pred) if n not in in_article})


def novel_words(pred, article):
    in_article = set(WORD.findall(article.lower()))
    return sorted({w for w in WORD.findall(pred.lower()) if len(w) > 3 and w not in in_article})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True, help="folder with predictions.jsonl from evaluate.py")
    p.add_argument("--split", default="test")
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--article_words", type=int, default=150, help="article words shown per example")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    preds = [json.loads(line) for line in open(eval_dir / "predictions.jsonl")]
    raw = load_cnn_dailymail(args.split)
    articles = raw.select([r["index"] for r in preds])["article"]
    for r, article in zip(preds, articles):
        r["article"] = article
        r["flags"] = {"repeated_trigrams": repeated_trigrams(r["prediction"]),
                      "numbers_not_in_article": numbers_not_in_article(r["prediction"], article),
                      "words_not_in_article": novel_words(r["prediction"], article)}

    rng = random.Random(args.seed)
    by_r1 = sorted(range(len(preds)), key=lambda i: preds[i]["rouge1"])
    picks = []   # (position in preds, reason)

    def add(i, reason):
        if all(i != j for j, _ in picks):
            picks.append((i, reason))

    for i in rng.sample(range(len(preds)), min(args.n_random, len(preds))):
        add(i, "random sample")
    add(by_r1[len(by_r1) // 2], "median ROUGE-1 (typical output)")
    add(by_r1[-1], "highest ROUGE-1")
    add(by_r1[0], "lowest ROUGE-1")
    repeats = sorted(range(len(preds)), key=lambda i: -len(preds[i]["flags"]["repeated_trigrams"]))
    for i in repeats[:2]:
        if preds[i]["flags"]["repeated_trigrams"]:
            add(i, "failure: repeated phrases")
    for i in [i for i in range(len(preds)) if preds[i]["flags"]["numbers_not_in_article"]][:2]:
        add(i, "failure: number not in the article")

    lines = [f"# Qualitative examples ({args.split} set)\n",
             f"{len(preds):,} generated summaries; {len(picks)} examples selected. "
             "Flags are automatic hints, not verdicts.\n"]
    for k, (i, reason) in enumerate(picks, 1):
        r = preds[i]
        words = r["article"].split()
        snippet = " ".join(words[:args.article_words]) + (" ..." if len(words) > args.article_words else "")
        f = r["flags"]
        ref_lines = [f"- {s}" for s in r["reference"].split("\n") if s.strip()]
        gen_lines = [f"- {s}" for s in r["prediction"].split("\n") if s.strip()] or ["- (empty)"]
        lines += [
            f"## Example {k}: {args.split} #{r['index']} ({reason})\n",
            f"ROUGE-1 {100 * r['rouge1']:.1f} | ROUGE-2 {100 * r['rouge2']:.1f} | "
            f"ROUGE-L {100 * r['rougeL']:.1f} | ROUGE-Lsum {100 * r['rougeLsum']:.1f}"
            + (" | hit the length limit" if r.get("hit_max_len") else "") + "\n",
            f"**Article (first {args.article_words} words):** {snippet}\n",
            "**Reference summary:**\n", *ref_lines, "",
            "**Generated summary:**\n", *gen_lines, "",
            "**Automatic flags:**\n",
            f"- repeated trigrams: {', '.join(f['repeated_trigrams']) or 'none'}",
            f"- numbers not in the article: {', '.join(f['numbers_not_in_article']) or 'none'}",
            f"- words not in the article: {', '.join(f['words_not_in_article'][:15]) or 'none'}",
            "",
        ]
    out = eval_dir / "qualitative_examples.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {len(picks)} examples to {out}")


if __name__ == "__main__":
    main()
