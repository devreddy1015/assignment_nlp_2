# Q3: Abstractive Text Summarization with a From-Scratch Transformer

DSL 504 Natural Language Processing, Group Assignment 2, Question 3.

A Transformer encoder-decoder ("Attention Is All You Need") written from basic PyTorch layers,
trained on CNN / Daily Mail 3.0.0 to turn a news article into a multi-sentence summary, and
evaluated with ROUGE on the full test set (11,490 articles).

No `nn.Transformer*`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging Face
model classes or pre-trained weights are used. The only pre-trained component is the tokenizer,
which the assignment allows.

**Results** (full test set, F1 x 100, greedy decoding; the report is `report/Q3_report.pdf`):

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Lead-3 baseline | 40.06 | 17.48 | 25.02 | 36.36 |
| Run 1: paper's learning rate, batch 32 (stalled, best = epoch 2) | 13.65 | 1.62 | 10.97 | 12.62 |
| **Run 2, final model: half the learning rate, batch 64 (best = epoch 20)** | **36.72** | **15.32** | **25.67** | **34.33** |

## Files

| File | What it does |
|---|---|
| `model.py` | Multi-head attention, feed-forward network, sinusoidal positional encoding, encoder/decoder layers, the three masks, the full model and greedy decoding |
| `data.py` | Text cleaning, tokenizer, `[NL]` sentence separator, dataset and batching (shifted decoder input / target) |
| `prepare_data.py` | Downloads the dataset, tokenizes all splits once, caches them as `.npy` and prints length statistics |
| `train.py` | Training: cross-entropy ignoring padding, Adam + warm-up/decay schedule, teacher forcing, early stopping, loss curves |
| `evaluate.py` | Greedy generation on the test set and ROUGE-1/2/L/Lsum, using the best checkpoint |
| `qualitative.py` | Picks typical, best, worst and failure examples into a Markdown file with automatic flags |
| `visualize_attention.py` | Cross-attention heatmap from the last decoder layer, recorded during generation |
| `lead3_baseline.py` | Lead-3 baseline, to check that the ROUGE setup matches the assignment's numbers |
| `sanity_checks.py` | Tests for the masks and the attention, plus a copy task that checks the whole model end to end |
| `utils.py` | Seeding, device / mixed-precision helpers, checkpoint loading, loss-curve plot |
| `make_submission.py` | Builds the submission zip without the dataset, the virtual environment or checkpoints |
| `report/analyze_outputs.py` | Statistics of the generated summaries for the report: copying, repetition, error indicators, per-article comparison with Lead-3, attention sharpness |
| `report/make_figures.py` | Collects the report figures into `report/figures/` |
| `report/build_pdf.py` | Converts `report/Q3_report.md` into HTML and PDF |

## 1. Requirements

- **Python 3.12** (3.10 or newer should also work).
- **An NVIDIA GPU is strongly recommended.** Tested on an RTX 4060 Laptop GPU (8 GB) with
  PyTorch 2.14.1 + CUDA 13.0 on Linux. Training at batch size 32 takes about 5.5 GB of GPU
  memory, so a GPU with 8 GB or more is comfortable; with less, see Troubleshooting.
  On a CPU it runs, but very slowly, so use a small `--train_size`.
- **Disk:** about 2 GB for the dataset download and the tokenized cache, plus space for PyTorch.
- **Internet** the first time, to download the dataset and the tokenizer from Hugging Face.

Libraries (`requirements.txt`):

| Library | Used for |
|---|---|
| `torch` >= 2.4 | the model, training and decoding |
| `datasets` >= 3.0 | loading CNN / Daily Mail |
| `transformers` >= 4.44 | the `bert-base-uncased` tokenizer only (no model classes) |
| `rouge_score` >= 0.1.2 | ROUGE-1/2/L/Lsum |
| `nltk` >= 3.8 | sentence splitting for the Lead-3 baseline |
| `numpy`, `matplotlib`, `tqdm` | arrays, plots, progress bars |
| `markdown` | only for `report/build_pdf.py` (the PDF also needs Chrome, Chromium, Brave or Edge) |

## 2. Setup

```bash
cd q3_summarization
uv venv --python 3.12 .venv            # or: python3 -m venv .venv
source .venv/bin/activate              # fish: source .venv/bin/activate.fish
                                       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt     # or: pip install -r requirements.txt
```

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False` on a machine with an NVIDIA GPU, install the CUDA build of PyTorch with the
command from https://pytorch.org/get-started/locally/.

All commands below are run from the `q3_summarization/` folder with the environment activated.
They work the same in bash and fish.

## 3. Quick checks (optional)

**Model tests, about 1 minute, no dataset needed:**

```bash
python sanity_checks.py
```

Expected output:

```
ok  no built-in Transformer/attention modules
ok  causal mask: earlier positions never see later tokens
ok  padding masks: extra padding in article or summary changes nothing
ok  attention rows sum to 1 and padding gets zero attention
ok  copy task: 96% of sequences copied exactly by greedy decoding (loss 0.0052)
All sanity checks passed.
```

**Smoke test of the whole pipeline on a tiny subset, a few minutes** (run `prepare_data.py`
first). The summaries will be poor; this only checks that every script runs.

```bash
python train.py --run_name smoke --train_size 2000 --val_size 500 --epochs 1
python evaluate.py --run_dir runs/smoke --limit 50
```

## 4. Full run, step by step

Times are for the RTX 4060 Laptop GPU.

| # | Command | Time | Writes |
|---|---|---|---|
| 1 | `python prepare_data.py` | about 2 min, plus the download the first time | `data_cache/` |
| 2 | `python train.py --run_name lr05_b64 --lr_factor 0.5 --accum 2` | about 26 min per epoch on all 287,113 articles; our run used all 20 epochs (8.7 hours) | `runs/lr05_b64/` |
| 3 | `python evaluate.py --run_dir runs/lr05_b64` | about 6 min | `runs/lr05_b64/eval_test/` |
| 4 | `python qualitative.py --eval_dir runs/lr05_b64/eval_test` | seconds | `runs/lr05_b64/eval_test/qualitative_examples.md` |
| 5 | `python visualize_attention.py --run_dir runs/lr05_b64 --index 0 --per_head` | under a minute | `runs/lr05_b64/attention/` |
| 6 | `python lead3_baseline.py` | under a minute | `outputs/lead3_test.json` |
| 7 | `python report/analyze_outputs.py --runs base lr05_b64` | about 3 min | `report/output_analysis.json` |
| 8 | `python report/make_figures.py --runs base lr05_b64` and `python report/build_pdf.py` | seconds | `report/figures/`, `report/Q3_report.pdf` |

Steps 7 and 8 need both runs (`runs/base` from `python train.py --run_name base`, evaluated with
steps 3-5). The report's Figure 6 shows all generated tokens of test article 2:
`python visualize_attention.py --run_dir runs/lr05_b64 --index 2 --max_rows 70 --window 70 --per_head`.

**Why the training options in step 2.** These are the settings of our final model (run 2): half
the paper's learning rate (`--lr_factor 0.5`) and an effective batch of 64 (`--accum 2`). With the
defaults (`python train.py --run_name base`, the paper's schedule and batch 32), training stalled at
a validation loss of about 4.5 and the model ignored most of the article. The report explains this in
Section 4.2. Both runs are in `runs/`.

What each step does:

1. **Prepare data.** Downloads CNN / Daily Mail 3.0.0, lowercases, tokenizes, truncates
   (articles 512 tokens, summaries 128 decoder positions), adds start/end tokens and caches
   everything as `.npy` arrays. Prints length statistics for every split.
2. **Train.** Teacher-forced training with the paper's warm-up learning-rate schedule, multiplied by
   `--lr_factor`. Validation loss is computed after every epoch; the best model is saved as
   `best.pt`, and training stops after 3 evaluations without improvement. The loss curves are redrawn after every evaluation.
3. **Evaluate.** Loads `best.pt` (never the last checkpoint), generates a summary for every test
   article with greedy decoding (no teacher forcing), and scores it with ROUGE.
4. **Qualitative examples.** Selects random, typical, best, worst and failure examples (repeated
   phrases, numbers not in the article) with the article, the reference and the generated summary.
5. **Attention heatmap.** Generates a summary for one test article and plots the last decoder
   layer's cross-attention recorded during generation. Try a few `--index` values and keep the
   clearest one.
6. **Lead-3 baseline.** The first three sentences of each article, scored the same way. It should
   be close to the assignment's 40.06 / 17.48 / 25.02 / 36.36.
7. **Output analysis.** Counts copied and novel n-grams, repeated trigrams and sentences, numbers and
   words that are not in the article, compares every article's ROUGE with Lead-3 (with a bootstrap
   interval), and measures how sharp the saved attention maps are.
8. **Report.** Copies the loss curves and heatmaps of both runs into `report/figures/`, draws the
   length histograms, learning-rate schedules and loss comparison, and builds the PDF.

**Everything in one command:**

```bash
python prepare_data.py && python train.py --run_name lr05_b64 --lr_factor 0.5 --accum 2 && python evaluate.py --run_dir runs/lr05_b64 && python qualitative.py --eval_dir runs/lr05_b64/eval_test && python visualize_attention.py --run_dir runs/lr05_b64 --index 0 --per_head && python lead3_baseline.py
```

**Long runs.** To keep training running after you close the terminal, and to save the log:

```bash
nohup python train.py --run_name lr05_b64 --lr_factor 0.5 --accum 2 > logs/train_lr05_b64.log 2>&1 &
tail -f logs/train_lr05_b64.log      # Ctrl+C stops watching, not training
```

**If training stops** (crash, reboot, Ctrl+C), continue from the last validation point with the
same options plus `--resume`:

```bash
python train.py --run_name lr05_b64 --lr_factor 0.5 --accum 2 --resume
```

The interrupted epoch starts again, reshuffled.

**What the training log shows.** Every 100 steps:

```
epoch 1 | step 500 | loss 6.9195 (interval mean 8.0906) | lr 1.24e-04 | 12,893 target tok/s | 1.4 min
```

and after every epoch a validation line such as
`[eval] step 8,972 | train ... | val ... (ppl ...) | saved best (0/3)`. The `(0/3)` counts
evaluations without improvement; training stops at `3/3`.

## 5. Options

Every script has `--help`. The most useful options:

**`train.py`**

| Option | Default | Meaning |
|---|---|---|
| `--run_name` | `base` | output folder `runs/<run_name>/` |
| `--train_size` | all (287,113) | random training subset, e.g. `100000` (about 9 min per epoch) |
| `--val_size` | all (13,368) | validation subset (the first N examples) |
| `--epochs` | 20 | maximum number of epochs |
| `--patience` | 3 | evaluations without improvement before early stopping |
| `--eval_every` | 0 | validate every N steps instead of once per epoch |
| `--batch_size` / `--accum` | 32 / 1 | batch size and gradient accumulation steps |
| `--d_model` / `--n_heads` | 256 / 4 | model width and number of attention heads |
| `--n_enc_layers` / `--n_dec_layers` | 4 / 4 | encoder and decoder depth |
| `--d_ff` / `--dropout` | 1024 / 0.1 | feed-forward size and dropout |
| `--warmup` / `--lr_factor` | 4000 / 1.0 | warm-up steps and a multiplier on the paper's schedule |
| `--label_smoothing` | 0.0 | label smoothing in the training loss |
| `--clip` | 1.0 | gradient-norm clipping |
| `--max_steps` | 0 | stop after N optimizer steps (0 = no limit) |
| `--no_amp` | off | turn off bf16 mixed precision |
| `--seed` | 42 | random seed |
| `--resume` | off | continue from `runs/<run_name>/last.pt` |

**`evaluate.py`**: `--run_dir` (required), `--split test|validation`, `--limit N` (first N articles
only, for quick checks), `--batch_size 64`, `--max_len 128` (maximum generated tokens).

**`qualitative.py`**: `--eval_dir` (required), `--n_random 3`, `--article_words 150`, `--seed 0`.

**`visualize_attention.py`**: `--run_dir` (required), `--index 0` (test example), `--window 60`
(source tokens shown), `--max_rows 40` (generated tokens shown), `--per_head` (also one plot per
head).

## 6. Outputs

```
data_cache/                    tokenized dataset (not submitted)
  meta.json                    special-token ids, truncation limits, length statistics
runs/lr05_b64/                  final model (run 2); runs/base/ (run 1) has the same layout
  config.json                  all options, model configuration, parameter count, training-set size, GPU
  history.json                 training and validation loss at every evaluation
  loss_curves.png              training and validation loss curves
  train_summary.json           best validation loss, steps, epochs, training time
  best.pt, last.pt             checkpoints (not submitted)
  eval_test/
    rouge.json                 ROUGE-1/2/L/Lsum (precision, recall, F1) and output statistics
    predictions.jsonl          every generated summary with its reference and ROUGE scores
    qualitative_examples.md    selected examples for the qualitative analysis
  attention/
    attention_test0.png        cross-attention heatmap, mean over heads
    attention_test0_heads.png  one heatmap per head
    attention_test0.json/.npz  generated and reference text, raw attention weights
outputs/lead3_test.json        Lead-3 ROUGE on the test set
logs/                          console output of each step
```

Where each output goes in the report:

| Report item | File |
|---|---|
| Tokenizer, truncation limits, share of truncated examples | `data_cache/meta.json` |
| Model configuration, parameter count, training-set size | `runs/lr05_b64/config.json` |
| Training and validation loss curves | `runs/lr05_b64/loss_curves.png` |
| Best validation loss, epochs, training time | `runs/lr05_b64/train_summary.json` |
| ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum on the test set | `runs/lr05_b64/eval_test/rouge.json` |
| Lead-3 comparison | `outputs/lead3_test.json` |
| Qualitative examples and failure cases | `runs/lr05_b64/eval_test/qualitative_examples.md` |
| Attention visualization | `runs/lr05_b64/attention/*.png` |

## 7. Design choices

- **Dataset:** CNN / Daily Mail 3.0.0, loaded as `abisee/cnn_dailymail`. This is the same dataset
  as the assignment's `load_dataset("cnn_dailymail", "3.0.0")`; recent `huggingface_hub` versions
  only accept the namespaced name. Train / validation / test = 287,113 / 13,368 / 11,490. The test
  set is used only for the final ROUGE scores.
- **Tokenizer:** pre-trained `bert-base-uncased` WordPiece, 30,522 sub-words plus one added
  `[NL]` token (30,523 in total). It is uncased, so the text is lowercased to match. Articles and
  summaries share one vocabulary.
- **Start / end tokens:** `[CLS]` starts every summary and `[SEP]` ends it.
  Decoder input = `[CLS] summary` (shifted right), target = `summary [SEP]` (shifted left).
- **Sentence breaks:** the `\n` between highlight sentences becomes the added `[NL]` token, because
  the WordPiece tokenizer would otherwise drop it. Generated `[NL]` becomes `\n` again, which
  ROUGE-Lsum needs.
- **Truncation:** articles to 512 tokens, summaries to 128 decoder positions (127 summary tokens
  plus the start or end token).
- **Masks:** a source padding mask (encoder self-attention and encoder-decoder attention), and a
  causal mask combined with a target padding mask (decoder self-attention).
- **Model (default):** d_model 256, 4 heads, 4 encoder + 4 decoder layers, d_ff 1024, dropout 0.1,
  post-norm `LayerNorm(x + Dropout(Sublayer(x)))`, embeddings scaled by sqrt(d_model). The
  embeddings and the output projection are tied. **15,186,688 trainable parameters.**
- **Training:** cross-entropy with `ignore_index=pad`, Adam (beta 0.9 / 0.98, eps 1e-9), the
  paper's schedule `lr = factor * d_model^-0.5 * min(step^-0.5, step * 4000^-1.5)`, gradient
  clipping 1.0, bf16 mixed precision, early stopping on validation loss (patience 3). The best
  checkpoint is restored for testing. Run 1 (`runs/base`): factor 1.0, batch 32; it stalled. Run 2
  (`runs/lr05_b64`, the final model): factor 0.5, batch 64 (32 x 2 accumulation steps).
- **Decoding:** greedy, at most 128 tokens, no teacher forcing. Each article is encoded once and
  the encoder output is reused at every step.
- **ROUGE:** `rouge_score` with `use_stemmer=True`, F1 against the original highlights text, on all
  11,490 test articles. ROUGE-L treats the summary as one sequence; ROUGE-Lsum works sentence by
  sentence (sentences separated by `\n`). Both are reported and labelled.

**Data statistics** (WordPiece tokens after lowercasing; summary counts include the `[NL]`
tokens), from `data_cache/meta.json`:

| Split | Pairs | Article median | Article 90th pct | Articles > 512 | Summary median | Summary 90th pct | Summaries > 127 |
|---|---|---|---|---|---|---|---|
| train | 287,113 | 798 | 1,479 | 79.8% | 61 | 97 | 2.7% |
| validation | 13,368 | 763 | 1,474 | 75.7% | 67 | 109 | 4.4% |
| test | 11,490 | 778 | 1,502 | 76.6% | 64 | 102 | 3.4% |

On the first 2,000 test articles, the sample the assignment measured, we get article median 639,
90th percentile 1,301, max 2,965, 62% longer than 512 tokens, and summary median 49. The
assignment's table gives 641 / 1,304 / 2,967 / 62.5% / 49, so the preprocessing matches.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Connection reset` or other download errors | Run the command again. Setting `HF_TOKEN` gives higher Hugging Face rate limits. |
| `FileNotFoundError: data_cache/meta.json` | Run `python prepare_data.py` first. |
| `FileNotFoundError: runs/<run_name>/best.pt` | Training has not finished its first validation yet. |
| CUDA out of memory in training | `--batch_size 16 --accum 2` (same effective batch size). |
| CUDA out of memory in evaluation | `python evaluate.py --run_dir runs/lr05_b64 --batch_size 32` |
| No GPU, or a GPU without bf16 | It falls back to fp32 automatically. On a CPU, use a small `--train_size`. |
| Training was interrupted | Run the same `train.py` command again with `--resume` added |
| Lead-3 differs slightly from 40.06 / 17.48 / 25.02 / 36.36 | Small differences can come from the NLTK sentence-splitter version. |

## 9. Reproducibility

The seed (default 42) fixes the training subset, the shuffling order of every epoch and the weight
initialisation. GPU kernels are not fully deterministic, so two runs can still differ slightly.
`runs/<run_name>/config.json` records every option, the GPU and the training-set size.

## 10. Submission

The report is in `report/` (`Q3_report.md`, `Q3_report.pdf`, `figures/`). To build the zip, run:

```bash
python make_submission.py        # writes ../q3_submission.zip
```

The zip contains the source code, this README, the report, the logs and the results (loss curves,
ROUGE scores, predictions, attention plots). It leaves out `.venv/`, `data_cache/` (the dataset),
`__pycache__/` and the `*.pt` checkpoints, as the assignment asks.
