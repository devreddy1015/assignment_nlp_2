# Abstractive Text Summarization with a Transformer Built from Scratch

**DSL 504 Natural Language Processing: Group Assignment 2, Question 3**

**Group members:** ⟦PENDING: names and roll numbers⟧

---

## Abstract

We implement the Transformer encoder-decoder of Vaswani et al. (2017) from basic PyTorch layers and
train it to write multi-sentence summaries of news articles from the CNN / Daily Mail dataset. The
attention, feed-forward, positional-encoding, encoder-layer and decoder-layer code is our own; only
the tokenizer is pre-trained. The model has 15.2 million parameters (d_model 256, 4 heads, 4 encoder
and 4 decoder layers) and is trained on all 287,113 training pairs with teacher forcing, the paper's
warm-up learning-rate schedule and early stopping on the validation loss. Summaries are generated
with greedy decoding and scored with ROUGE on the full test set of 11,490 articles.
⟦PENDING: one sentence with the four ROUGE scores and the gap to Lead-3⟧

### Key facts at a glance

| Item | Value |
|---|---|
| Tokenizer | Pre-trained `bert-base-uncased` WordPiece, 30,522 sub-words + 1 added `[NL]` token = 30,523, shared by articles and summaries |
| Truncation | Articles: first 512 tokens. Summaries: first 127 tokens, plus a start or end token (128 decoder positions) |
| Model | d_model 256, 4 heads (d_k 64), 4 encoder + 4 decoder layers, d_ff 1024, dropout 0.1, tied embeddings |
| Parameters | 15,186,688 |
| Training-set size | 287,113 pairs (the full training split) |
| Decoding | Greedy, at most 128 tokens, no teacher forcing |
| ROUGE on the test set (F1, 11,490 articles) | ⟦PENDING: ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum⟧ (Table 7) |

---

## 1. Introduction

Abstractive summarisation means writing a summary in new words, rather than copying sentences out
of the article (extractive summarisation). It is a sequence-to-sequence problem: the input is a news
article of several hundred words and the output is a short summary of three or four sentences.

We use the Transformer encoder-decoder from "Attention Is All You Need" (Vaswani et al., 2017). The
encoder reads the whole article and builds a contextual representation of every token. The decoder
writes the summary one token at a time; at every step it looks back at the summary written so far
(masked self-attention) and at the article (encoder-decoder attention). We implement every
component ourselves, train the model on CNN / Daily Mail, evaluate it with ROUGE, compare it with
the Lead-3 baseline, study its outputs and visualise where it attends while it writes.

## 2. Dataset and preprocessing

### 2.1 Dataset

We use CNN / Daily Mail version 3.0.0 (Hermann et al., 2015; Nallapati et al., 2016). Each example is
a news article and its "highlights", three or four sentences written by the journalists, separated by
newline characters. The highlights are the reference summaries.

**Table 1. Dataset splits and how we use them.**

| Split | Pairs | Used for |
|---|---|---|
| Training | 287,113 | Training the model (all of it) |
| Validation | 13,368 | Validation loss after every epoch, early stopping, choosing the checkpoint |
| Test | 11,490 | Final ROUGE scores only, never for training or tuning |

The dataset is loaded with `load_dataset("abisee/cnn_dailymail", "3.0.0")`. This is the same dataset
as the assignment's `load_dataset("cnn_dailymail", "3.0.0")`; recent versions of the Hugging Face
hub only accept the namespaced name. No split contains an empty article or summary, so no example
was dropped and the test set is complete.

### 2.2 Preprocessing pipeline

Preprocessing is done once by `prepare_data.py` (using `data.py`), and the result is cached as NumPy
arrays.

1. **Lowercasing and whitespace.** All text is lowercased and every run of whitespace is collapsed
   to a single space. Lowercasing matches the uncased tokenizer.
2. **Sentence separator.** The `\n` between highlight sentences is replaced by a special `[NL]` token.
   The BERT tokenizer treats a newline as ordinary whitespace, so without this the sentence
   boundaries would be lost. ROUGE-Lsum needs them: it compares the summaries sentence by sentence,
   with sentences separated by `\n` in both the reference and the output. The model learns to
   produce `[NL]` between sentences, and when token ids are turned back into text every `[NL]`
   becomes `\n` again.
3. **Sub-word tokenization.** We use the pre-trained `bert-base-uncased` WordPiece tokenizer
   (Devlin et al., 2019), which the assignment allows. Its vocabulary of 30,522 sub-words is extended
   with `[NL]` (id 30,522), giving 30,523 entries. Articles and summaries share this one vocabulary.
   Sub-words mean there are no out-of-vocabulary words: a rare word or name is split into known
   pieces, for example "123rd" becomes `123 ##rd`.
4. **Truncation.** Each article is cut to its first 512 tokens and each summary to its first 127
   tokens (Section 2.3).
5. **Start and end tokens.** Every summary gets the start token `[CLS]` (id 101) in front and the end
   token `[SEP]` (id 102) at the end. We reuse these existing special tokens instead of adding new
   ones.
6. **Padding.** Sequences are padded with `[PAD]` (id 0) to a fixed length (512 for articles, 129
   for summaries) and stored as `uint16` arrays. When a batch is built, the columns that are padding
   in every row of the batch are removed, so all sequences in a batch have the same length but no
   compute is wasted on padding beyond the longest sequence in the batch.

**Decoder input and target.** The summary `[CLS] w1 ... wn [SEP]` gives the decoder input (shifted
right, starts with the start token) and the prediction target (shifted left, ends with the end
token). Table 2 shows this for the first test example, whose summary has 40 tokens including the
start and end tokens.

**Table 2. Decoder input and target for test example 0.**

| Position | 1 | 2 | 3 | 4 | ... | 17 | 18 | ... | 39 |
|---|---|---|---|---|---|---|---|---|---|
| Decoder input | `[CLS]` | membership | gives | the | ... | . | `[NL]` | ... | last word |
| Target | membership | gives | the | icc | ... | `[NL]` | israel | ... | `[SEP]` |

The article of the same example begins `( cnn ) the palestinian authority officially became the 123
##rd member of the international criminal court on wednesday , ...`.

### 2.3 Length statistics and the truncation limits

**Table 3. Length in WordPiece tokens after lowercasing (summary lengths include the `[NL]` tokens).**

| Split | Article median | Article 90th pct | Article max | Articles > 512 | Summary median | Summary 90th pct | Summaries > 127 |
|---|---|---|---|---|---|---|---|
| Train | 798 | 1,479 | 6,321 | 79.8% | 61 | 97 | 2.7% |
| Validation | 763 | 1,474 | 2,415 | 75.7% | 67 | 109 | 4.4% |
| Test | 778 | 1,502 | 2,965 | 76.6% | 64 | 102 | 3.4% |

![Token-length histograms of the training split](figures/length_histograms.png)

**Figure 1.** Article and summary lengths in the training split, with the truncation limits.

The assignment's table was measured on 2,000 test articles. On the first 2,000 test articles we get
an article median of 639 tokens, a 90th percentile of 1,301, a maximum of 2,965, 62% longer than 512
tokens, and a summary median of 49 (47 without `[NL]`). The assignment reports 641, 1,304, 2,967,
62.5% and 49, so our tokenization matches. The full test split has longer articles (median 778)
than this sample, so Table 3 gives the statistics of every full split.

**Why 512 article tokens.** About 77-80% of the articles are longer than 512 tokens, so most are
truncated. This loses little, because news articles put the key facts first (the "inverted pyramid"
style). We measured it on the test set: 86.3% of the distinct words of a reference summary appear
somewhere in the full article, and 80.8% already appear in its first 512 tokens. So the truncated
article still contains about 94% of the reference words that the article contains at all. 512 tokens
keep a median of 437 words per article. A longer limit would also be expensive: self-attention cost
grows with the square of the length, and 512 is the upper end of the range the assignment suggests.

**Why 127 summary tokens.** Only 2.7% of the training summaries are longer than 127 tokens, so almost
every summary is kept whole. With the start or end token this gives 128 decoder positions, which
matches the maximum generation length.

## 3. Model architecture

All model code is in `model.py`. It uses only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.Dropout` and plain tensor operations. No `nn.Transformer`, `nn.TransformerEncoder(Layer)`,
`nn.TransformerDecoder(Layer)`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging
Face model class or pre-trained weight is used.

### 3.1 Overview

```
article ids (B, S)
  -> Embedding x sqrt(d_model) + positional encoding -> dropout
  -> 4 x EncoderLayer [ self-attention -> feed-forward ]
  -> memory (B, S, 256) ------------------------------------------+
                                                                  |
summary shifted right (B, T)                                      |
  -> Embedding x sqrt(d_model) + positional encoding -> dropout   |
  -> 4 x DecoderLayer [ masked self-attention                     |
                        -> encoder-decoder attention  <-----------+
                        -> feed-forward ]
  -> Linear (weights tied to the embedding) -> logits (B, T, 30,523)
```

B is the batch size, S the article length and T the summary length. The encoder and decoder share one
embedding table, because articles and summaries share one vocabulary.

### 3.2 Scaled dot-product attention and multi-head attention

The core formula is

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + mask ) V
```

`MultiHeadAttention` projects the queries, keys and values with three `nn.Linear(256, 256)` layers,
splits each into 4 heads of size d_k = 64, applies the formula to every head in parallel,
concatenates the heads and applies an output projection `nn.Linear(256, 256)`. The central lines:

```python
q = self._split_heads(self.w_q(query))                  # (B, H, Tq, Dk)
k = self._split_heads(self.w_k(key_value))              # (B, H, Tk, Dk)
v = self._split_heads(self.w_v(key_value))
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, Tq, Tk)
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
weights = torch.softmax(scores, dim=-1)                 # each row sums to 1
out = self.attn_dropout(weights) @ v                    # (B, H, Tq, Dk)
out = out.transpose(1, 2).contiguous().view(B, Tq, H * Dk)
return self.w_o(out), weights
```

Masking replaces a blocked score with the most negative finite number, which after the softmax
gives a weight of exactly zero, the same as adding minus infinity. A finite value is used so that a
row in which every position is blocked would give a uniform row instead of NaN. The same module
serves as self-attention (query and key_value are the same sequence) and as encoder-decoder
attention (the query comes from the decoder, key_value is the encoder output). It also returns the
attention weights, which we use for the visualisation in Section 7.

### 3.3 Position-wise feed-forward network

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

Two linear layers, 256 -> 1024 -> 256, with a ReLU in between (and dropout after the ReLU), applied
to every position separately.

### 3.4 Positional encoding

Attention by itself ignores word order, so a fixed sinusoidal encoding is added to the token
embeddings:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

The table is computed once for 512 positions and stored as a buffer, so it is not trained. As in the
paper, the embeddings are multiplied by sqrt(d_model) = 16 before the encoding is added, and dropout
is applied to the sum.

### 3.5 Encoder and decoder layers

Every sub-layer is wrapped as `LayerNorm(x + Dropout(Sublayer(x)))`, the post-norm form of the
original paper.

- **Encoder layer:** multi-head self-attention, then the feed-forward network.
- **Decoder layer:** masked multi-head self-attention over the summary so far, then encoder-decoder
  attention (queries from the decoder, keys and values from the encoder output), then the
  feed-forward network.

The model stacks 4 encoder layers and 4 decoder layers. Dropout (0.1) is applied to the embeddings,
to every sub-layer output, to the attention weights and inside the feed-forward network.

### 3.6 The three masks

Masks are boolean tensors in which True means "may attend".

**Table 4. Masks.**

| Mask | Shape | Used in | Blocks |
|---|---|---|---|
| Source padding mask | (B, 1, 1, S) | Encoder self-attention and encoder-decoder attention | Padding tokens of the article |
| Causal (look-ahead) mask | (1, 1, T, T) | Decoder self-attention | Later summary positions |
| Target padding mask | (B, 1, 1, T) | Decoder self-attention (combined with the causal mask by AND) | Padding tokens of the summary |

The causal mask is lower-triangular: position t may attend to positions 1 to t only. For T = 4:

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

### 3.7 Output layer and initialisation

A final linear layer maps each decoder output vector to 30,523 vocabulary logits. Its weight matrix
is shared with the embedding table (weight tying; Press and Wolf, 2017, also used in the Transformer
paper), which saves 7.8 million parameters. Weight matrices are initialised with Xavier-uniform
initialisation. The embedding table is initialised from a normal distribution with standard
deviation d_model^-0.5, so that after scaling by sqrt(d_model) the embeddings have roughly unit
variance, and the row of the padding token is set to zero.

### 3.8 Configuration and parameter count

**Table 5. Model configuration, compared with the paper's base model.**

| Hyperparameter | Ours | Transformer base (paper) |
|---|---|---|
| d_model | 256 | 512 |
| Attention heads (d_k) | 4 (64) | 8 (64) |
| Encoder / decoder layers | 4 / 4 | 6 / 6 |
| Feed-forward size d_ff | 1,024 | 2,048 |
| Dropout | 0.1 | 0.1 |
| Vocabulary | 30,523 (shared) | 37,000 (shared BPE) |
| Parameters | **15,186,688** | about 65 million |

Parameter breakdown:

| Part | Parameters |
|---|---|
| Shared embedding / output projection (30,523 x 256) | 7,813,888 (51.5%) |
| Encoder: 4 layers x 789,760 (attention 263,168 + feed-forward 525,568 + 2 LayerNorms 1,024) | 3,159,040 |
| Decoder: 4 layers x 1,053,440 (2 attentions 526,336 + feed-forward 525,568 + 3 LayerNorms 1,536) | 4,213,760 |
| **Total** | **15,186,688** |

Half of the parameters are in the embedding table, which is typical for a small model with a large
vocabulary. The configuration is within the range the assignment suggests (d_model 256-512, 3-6
layers per stack, 4-8 heads) and fits comfortably on an 8 GB GPU.

### 3.9 Greedy decoding

At test time there is no teacher forcing. `greedy_decode` encodes the article once and reuses the
encoder output at every step. It starts from `[CLS]`, runs the decoder on the tokens generated so
far, takes the most probable next token, appends it and repeats, until every summary in the batch
has produced `[SEP]` or 128 tokens have been generated. The padding and start tokens are never
chosen, and a sequence that has finished just receives padding while the rest of the batch
continues. With `return_attention=True`, the function also records the encoder-decoder attention of
the last decoder layer at every step, which is what Section 7 visualises. For simplicity the
decoder is re-run on the whole prefix at every step (no key/value cache); this is slower but gives
the same result.

### 3.10 Checking the implementation

`sanity_checks.py` tests the implementation before any real training:

| Check | Result |
|---|---|
| The model contains no built-in Transformer or attention modules | Passed |
| Causal mask: changing a later summary token never changes the output at earlier positions | Passed |
| Padding masks: adding padding to the article or the summary does not change the outputs | Passed |
| Attention weights sum to 1 and padding positions get zero attention | Passed |
| Copy task: a small model is trained to copy random sequences, then greedy decoding must reproduce them | 96% copied exactly (loss 0.005) |

The copy task exercises the whole pipeline end to end (embeddings, positional encoding, both attention
types, the masks, training and greedy decoding). A model with a broken causal mask or broken
cross-attention cannot pass it. The first two checks test directly for common pitfalls 2 and 3 of the
assignment.

## 4. Training

### 4.1 Setup

**Table 6. Training configuration.**

| Setting | Value |
|---|---|
| Training data | All 287,113 training pairs, reshuffled every epoch |
| Loss | Cross-entropy over the vocabulary, padding positions ignored (`ignore_index = 0`) |
| Teacher forcing | Decoder input = gold summary shifted right; the whole summary is trained in one parallel pass |
| Optimiser | Adam, beta = (0.9, 0.98), eps = 1e-9 |
| Learning-rate schedule | lr = d_model^-0.5 x min(step^-0.5, step x warmup^-1.5), warm-up 4,000 steps |
| Batch size | 32 articles (8,972 optimizer steps per epoch) |
| Gradient clipping | Global norm 1.0 |
| Precision | bf16 mixed precision (the loss is computed in fp32) |
| Validation | Loss on all 13,368 validation pairs after every epoch |
| Early stopping | Stop after 3 epochs without improvement; at most 20 epochs |
| Checkpoint for testing | The best validation-loss checkpoint (`best.pt`), never the last one |
| Seed | 42 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.14.1 + CUDA 13.0 |
| Speed | About 12,900 target tokens per second, about 24 minutes per epoch |

**Teacher forcing.** During training the decoder input is the gold summary shifted right, so the model
predicts every summary token in one parallel pass. The causal mask makes sure that position t only
sees the gold tokens before it, so the model cannot copy the answer.

**Loss.** The cross-entropy is averaged over the real target tokens only; padding positions are
ignored in both the training and the validation loss (common pitfall 4). The validation loss is the
average over all non-padding target tokens of the validation split.

**Learning-rate schedule.** The learning rate rises linearly for the first 4,000 steps to a peak of
9.88e-4 and then decays in proportion to 1/sqrt(step): 6.6e-4 at the end of epoch 1 and 4.7e-4 at
the end of epoch 2. The warm-up keeps the early updates small while Adam's statistics are still
unreliable, which matters for post-norm Transformers.

![Learning-rate schedule](figures/lr_schedule.png)

**Figure 2.** The learning-rate schedule used for training.

**Early stopping.** After every epoch the model is evaluated on the validation split. When the
validation loss improves, the model is saved as `best.pt`. Training stops after 3 epochs in a row
without improvement, and the evaluation script always loads `best.pt` (common pitfall 7).

### 4.2 Training run

⟦PENDING: number of epochs trained, epoch of the best checkpoint, best validation loss and
perplexity, whether early stopping triggered, total training time (from runs/base/train_summary.json
and history.json)⟧

## 5. Quantitative results

### 5.1 Loss curves

⟦PENDING: figure figures/loss_curves.png and a short description of the curves⟧

![Training and validation loss](figures/loss_curves.png)

**Figure 3.** Training loss (light: every 100 steps; dots: mean per epoch) and validation loss after
every epoch. The dashed line marks the best checkpoint, which is used for testing.

### 5.2 ROUGE on the test set

All scores are F1 x 100 on the full test set (11,490 articles), computed with the `rouge_score`
package with stemming (`use_stemmer=True`). The generated token ids are converted back to text before
scoring, and the references are the original highlights. **ROUGE-L** treats each summary as a single
sequence and finds the longest common subsequence. **ROUGE-Lsum** splits both summaries into
sentences at `\n` and computes the longest common subsequence sentence by sentence; it is the
standard variant for CNN / Daily Mail. The two give very different numbers, so we report both.

**Table 7. ROUGE F1 on the CNN / Daily Mail test set.**

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Lead-3 (assignment) | 40.06 | 17.48 | 25.02 | 36.36 |
| Lead-3 (our ROUGE setup) | ⟦PENDING⟧ | ⟦PENDING⟧ | ⟦PENDING⟧ | ⟦PENDING⟧ |
| **Our Transformer (greedy)** | ⟦PENDING⟧ | ⟦PENDING⟧ | ⟦PENDING⟧ | ⟦PENDING⟧ |

The Lead-3 row computed with our own scoring code checks that our ROUGE setup reproduces the
assignment's numbers.

⟦PENDING: precision / recall table from runs/base/eval_test/rouge.json⟧

### 5.3 Properties of the generated summaries

⟦PENDING: from rouge.json "stats": average length in words vs. the reference, average number of
sentences, % of summaries with a repeated word trigram, % that reached 128 tokens without an end
token, % empty⟧

### 5.4 The gap to Lead-3

⟦PENDING: state the measured gap; keep only the reasons below that the results support⟧

A Transformer trained from scratch on a limited budget usually scores below Lead-3. The main
reasons:

1. **Lead-3 is a very strong baseline on CNN / Daily Mail.** News is written in the inverted-pyramid
   style, and the highlights often restate the first sentences. Even dedicated neural models of the
   time did not beat it: the pointer-generator network with coverage (See et al., 2017) reached
   ROUGE-1 39.53 against 40.34 for their Lead-3.
2. **No pre-training.** Our model learns English, facts about the world and summarisation from
   287,113 articles alone. Models that beat Lead-3 clearly, such as BART (Lewis et al., 2020;
   ROUGE-1 44.16), are pre-trained on far more text and have hundreds of millions of parameters.
3. **No copy mechanism.** Names and numbers must be produced from the output vocabulary. WordPiece
   avoids unknown words, but a rare name split into several pieces has to be generated correctly
   piece by piece. Pointer-generator networks add an explicit copy distribution for exactly this.
4. **Model size and compute.** 15 million parameters and a training budget of hours on one laptop
   GPU, compared with 65 million parameters for the base Transformer.
5. **Greedy decoding and exposure bias.** The model is trained on gold prefixes but generates from its
   own predictions; a single wrong token cannot be undone, and greedy decoding tends to repeat
   phrases. Beam search with trigram blocking usually adds ROUGE points.
6. **Truncation.** About 6% of the reference words that occur in an article fall beyond the first
   512 tokens (Section 2.3), a small loss.

## 6. Qualitative analysis

### 6.1 How the examples were chosen

`qualitative.py` selects the examples automatically from the test predictions so that they are not
cherry-picked: three random articles, the article with the median ROUGE-1 (a typical output), the
best and the worst article by ROUGE-1, the two summaries with the most repeated word trigrams, and
two summaries that contain a number not found in the article. For every example it lists the start
of the article, the reference and the generated summary, and three automatic flags: repeated
trigrams, numbers not in the article and words not in the article. We checked every flag against the
article by hand before calling something an error.

### 6.2 Examples

⟦PENDING: at least five examples (article excerpt, reference, generated summary, ROUGE) from
runs/base/eval_test/qualitative_examples.md, including failure cases⟧

### 6.3 Failure analysis

⟦PENDING: categories with counts or examples: repeated phrases, wrong names, wrong numbers, facts not
in the article, summaries that copy the lead sentences, too short / too long⟧

## 7. Attention visualisation

### 7.1 Method

`visualize_attention.py` generates a summary for one test article with greedy decoding and records
the encoder-decoder (cross) attention of the **last decoder layer at every generation step**, that is,
while the model writes the summary by itself, not during teacher-forced training. Row t of the
heatmap is the attention the model paid to the article when it generated token t. The main figure
**averages the 4 heads**; a second figure shows each head separately. Rows are the generated summary
tokens and columns are article tokens, with the tokens as axis labels. A whole 512-token article is
unreadable, so the plot shows the 60-token window of the article that receives the most attention,
and the title states what share of the attention falls inside this window.

### 7.2 Results

⟦PENDING: heatmap figures/attention_test<N>.png (mean of heads) and figures/attention_test<N>_heads.png,
with an interpretation⟧

## 8. Discussion

⟦PENDING: summarise what worked and what did not, based on the results⟧

**Possible improvements**, roughly in order of expected benefit for the effort:

- Beam search with a length penalty and trigram blocking, to reduce repetition.
- A copy (pointer) mechanism, to get names and numbers right.
- Training longer or with a larger model (d_model 512, 6 + 6 layers) if more compute is available.
- Label smoothing (0.1, as in the paper), which often improves generation quality.
- Starting from a pre-trained encoder-decoder, which the assignment rules out but which is the
  standard way to beat Lead-3.

## 9. Conclusion

⟦PENDING⟧

## References

- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional
  Transformers for language understanding. NAACL.
- Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M. and Blunsom, P.
  (2015). Teaching machines to read and comprehend. NeurIPS.
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and
  Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language
  generation, translation, and comprehension. ACL.
- Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. Text Summarization
  Branches Out, ACL Workshop.
- Nallapati, R., Zhou, B., dos Santos, C., Gulçehre, Ç. and Xiang, B. (2016). Abstractive text
  summarization using sequence-to-sequence RNNs and beyond. CoNLL.
- Press, O. and Wolf, L. (2017). Using the output embedding to improve language models. EACL.
- See, A., Liu, P. J. and Manning, C. D. (2017). Get to the point: Summarization with
  pointer-generator networks. ACL.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and
  Polosukhin, I. (2017). Attention is all you need. NeurIPS.

---

## Appendix A. How we avoided the assignment's common pitfalls

| Pitfall | What we did |
|---|---|
| 1. Using `nn.Transformer`, `nn.MultiheadAttention` or a pre-trained model | All components written from basic layers (Section 3); `sanity_checks.py` checks that no built-in module is present |
| 2. Missing or wrong causal mask | Lower-triangular mask combined with the target padding mask; tested by changing future tokens (Section 3.10) |
| 3. Forgetting the padding mask | Source padding mask in encoder self-attention and cross-attention, target padding mask in decoder self-attention; tested by adding padding |
| 4. Counting padding in the loss | `ignore_index = 0` in training; the validation loss is averaged over non-padding tokens only |
| 5. ROUGE on teacher-forced outputs | ROUGE is computed on summaries generated by greedy decoding with no teacher forcing |
| 6. ROUGE-L without saying which | Both ROUGE-L and ROUGE-Lsum are reported and labelled |
| 7. Tuning on the test set, or testing the last checkpoint | The test set is used only by the final evaluation; early stopping uses the validation set; testing uses the best checkpoint |

## Appendix B. Code map

| File | Contents | Rubric item |
|---|---|---|
| `data.py`, `prepare_data.py` | Cleaning, tokenizer, `[NL]`, truncation, start/end tokens, padding, batching | 1. Dataset loading and preprocessing |
| `model.py` | Multi-head attention, feed-forward network, positional encoding | 2. Transformer core components |
| `model.py` | Encoder and decoder layers, masks, full model, greedy decoding | 3. Full encoder-decoder model |
| `train.py`, `utils.py` | Loss, Adam, learning-rate schedule, teacher forcing, early stopping, loss curves | 4. Training procedure |
| `evaluate.py`, `lead3_baseline.py` | Greedy generation, ROUGE-1/2/L/Lsum, Lead-3 | 5. Quantitative evaluation |
| `qualitative.py`, `visualize_attention.py` | Example selection with error flags, attention heatmaps | 6. Qualitative analysis and attention |
| `sanity_checks.py` | Tests for the masks, attention and the whole model | 2 and 3 |

## Appendix C. Reproducing the results

The `README.md` in the code folder lists the setup and every command. In short:

```bash
python sanity_checks.py
python prepare_data.py
python train.py --run_name base
python evaluate.py --run_dir runs/base
python qualitative.py --eval_dir runs/base/eval_test
python visualize_attention.py --run_dir runs/base --index 0 --per_head
python lead3_baseline.py
python report/make_figures.py --run_dir runs/base
```
