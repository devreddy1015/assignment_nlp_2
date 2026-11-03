"""
Statistics of the generated test summaries used in Sections 5-7 of the report.

- how much of a summary is copied from the article (novel n-grams, verbatim sentences) and from
  where in the article the copied text comes, compared with the reference summaries
- error indicators: repeated trigrams and sentences, numbers and words not in the article
- per-article comparison with Lead-3, with a paired bootstrap of the F1 difference
- a diagnostic only (not used for the reported scores): ROUGE after deleting repeated sentences
- sharpness of the last decoder layer's cross-attention in the saved heatmap data

All counts use the same word tokenization for generated summaries, references and articles
(lowercase runs of letters and digits), so "patient ' s" and "patient's" are the same words.

Run from q3_summarization/:
    python report/analyze_outputs.py --runs base lr05_b64
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from nltk.tokenize import sent_tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_cnn_dailymail  # noqa: E402
from evaluate import ROUGE_TYPES, score_rouge  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
NUMBER = re.compile(r"\d+")
NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten "
                                               "eleven twelve".split())}
STUDENT_NEWS = "this page includes the show transcript"


def words(text):
    return TOKEN.findall(text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def summary_stats(summary, article, art_tokens, art_ngrams, lead3_len):
    """Copy and error statistics of one summary (generated or reference) against its article."""
    tokens = words(summary)
    out = {}
    for n in (1, 2, 4):
        grams = ngrams(tokens, n)
        out[f"novel_{n}"] = np.mean([g not in art_ngrams[n] for g in grams]) if grams else np.nan
    sentences = [words(s) for s in summary.split("\n") if words(s)]
    art_text = " " + " ".join(art_tokens) + " "
    out["copied_sentences"] = np.mean([f" {' '.join(s)} " in art_text for s in sentences]) if sentences else np.nan
    # where the copied 4-grams first occur in the article: inside the Lead-3 sentences or later
    first = [art_ngrams["first4"][g] for g in ngrams(tokens, 4) if g in art_ngrams["first4"]]
    out["copied4_in_lead3"] = np.mean([p < lead3_len for p in first]) if first else np.nan
    out["copied4_rel_pos"] = np.mean([p / max(len(art_tokens), 1) for p in first]) if first else np.nan
    trigrams = ngrams(tokens, 3)
    out["repeated_trigram"] = len(trigrams) != len(set(trigrams))
    out["repeated_sentence"] = len(sentences) != len({tuple(s) for s in sentences})
    art_numbers = set(NUMBER.findall(article)) | {NUMBER_WORDS[w] for w in art_tokens if w in NUMBER_WORDS}
    out["number_not_in_article"] = any(n not in art_numbers for n in NUMBER.findall(summary))
    novel = {w for w in tokens if len(w) > 3 and not w.isdigit() and w not in art_ngrams["vocab"]}
    out["novel_words"] = len(novel)
    out["words"] = len(tokens)
    return out


def article_index(article):
    tokens = words(article)
    first4 = {}
    for i, g in enumerate(ngrams(tokens, 4)):
        first4.setdefault(g, i)
    lead3_len = len(words(" ".join(sent_tokenize(article)[:3])))
    return tokens, {1: set(ngrams(tokens, 1)), 2: set(ngrams(tokens, 2)), 4: set(first4),
                    "first4": first4, "vocab": set(tokens)}, lead3_len


def drop_repeated_sentences(summary):
    seen, kept = set(), []
    for sentence in summary.split("\n"):
        key = tuple(words(sentence))
        if key and key not in seen:
            seen.add(key)
            kept.append(sentence)
    return "\n".join(kept)


def paired_bootstrap(diff, samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(samples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def attention_stats(att_dir):
    """Normalised entropy, peak weight relative to uniform, and how often the most-attended article
    token is the token being generated (when that token occurs in the article)."""
    rows = []
    for f in sorted(att_dir.glob("attention_test*.npz")):
        z = np.load(f)
        attn, src, gen = z["attention"].astype(np.float64), z["source_ids"], z["generated_ids"]
        length = int((src != 0).sum())
        mean = attn.mean(axis=0)[:, :length]
        entropy = -(mean * np.log(mean + 1e-12)).sum(axis=1) / np.log(length)
        peak = mean.max(axis=1) * length
        in_article = np.isin(gen, src[:length]) & (gen >= 1996)  # ids below 1996: special tokens, punctuation, single characters
        hits = src[mean.argmax(axis=1)] == gen
        rows.append({"file": f.stem, "article_tokens": length, "entropy": float(entropy.mean()),
                     "peak_x_uniform": float(peak.mean()),
                     "argmax_is_same_token": float(hits[in_article].mean()) if in_article.any() else None,
                     "tokens_in_article": int(in_article.sum()), "generated_tokens": len(gen)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["base", "lr05_b64"], help="run names under runs/")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="report/output_analysis.json")
    args = p.parse_args()

    raw = load_cnn_dailymail("test")
    articles, references = raw["article"], raw["highlights"]
    print("indexing articles ...")
    index = [article_index(a) for a in articles]
    lead3 = ["\n".join(sent_tokenize(a)[:3]) for a in articles]

    systems = {"reference": references, "lead3": lead3}
    per_example_rouge = {}
    for run in args.runs:
        preds = [json.loads(l) for l in open(Path(args.runs_dir) / run / "eval_test" / "predictions.jsonl")]
        systems[run] = [r["prediction"] for r in preds]
        per_example_rouge[run] = {t: np.array([r[t] for r in preds]) for t in ROUGE_TYPES}
    print("scoring Lead-3 per article ...")
    _, lead3_scores = score_rouge(references, lead3)
    per_example_rouge["lead3"] = {t: np.array([s[t].fmeasure for s in lead3_scores]) for t in ROUGE_TYPES}

    result = {"summaries": {}, "vs_lead3": {}, "without_repeated_sentences": {}, "attention": {}}
    for name, summaries in systems.items():
        stats = [summary_stats(s, a, *idx) for s, a, idx in zip(summaries, articles, index)]
        agg = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
        agg = {k: 100 * v if k not in ("novel_words", "words") else v for k, v in agg.items()}
        agg["pct_with_novel_word"] = 100 * float(np.mean([s["novel_words"] > 0 for s in stats]))
        agg["pct_shorter_than_30_words"] = 100 * float(np.mean([s["words"] < 30 for s in stats]))
        agg["pct_longer_than_100_words"] = 100 * float(np.mean([s["words"] > 100 for s in stats]))
        agg["student_news_template"] = int(sum(STUDENT_NEWS in s.lower() for s in summaries))
        result["summaries"][name] = agg

    for run in args.runs:
        comp = {}
        for t in ROUGE_TYPES:
            diff = per_example_rouge[run][t] - per_example_rouge["lead3"][t]
            lo, hi = paired_bootstrap(diff)
            comp[t] = {"mean_diff": 100 * float(diff.mean()), "ci95": [100 * lo, 100 * hi],
                       "pct_articles_better": 100 * float((diff > 0).mean()),
                       "pct_articles_worse": 100 * float((diff < 0).mean())}
        result["vs_lead3"][run] = comp
        deduped = [drop_repeated_sentences(s) for s in systems[run]]
        rouge, _ = score_rouge(references, deduped)
        result["without_repeated_sentences"][run] = {
            "summaries_changed": int(sum(d != s for d, s in zip(deduped, systems[run]))),
            "rouge_f1": {t: rouge[t]["f1"] for t in ROUGE_TYPES}}
        att_dir = Path(args.runs_dir) / run / "attention"
        if att_dir.exists():
            result["attention"][run] = attention_stats(att_dir)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
