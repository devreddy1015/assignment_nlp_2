"""
Transformer encoder-decoder ("Attention Is All You Need") for abstractive summarisation,
written from scratch.

Only basic layers are used (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout) together with
plain tensor operations. The attention, feed-forward, positional-encoding, encoder-layer and
decoder-layer code is all implemented below; no built-in Transformer / attention modules and no
pre-trained weights are used.

Shape names used in the comments:
    B = batch size, S = source (article) length, T = target (summary) length,
    D = d_model, H = number of heads, Dk = D // H (size of one head), V = vocabulary size
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512          # longest sequence the positional-encoding table covers
    pad_id: int = 0
    tie_embeddings: bool = True  # share encoder/decoder embeddings and the output projection


# --------------------------------------------------------------------------- masks
# Masks are boolean: True = "may attend to this position", False = "blocked".

def make_src_mask(src, pad_id):
    """Source padding mask, shape (B, 1, 1, S).

    Used in encoder self-attention and in the decoder's encoder-decoder attention, so that no
    query ever attends to a padding token of the article. Broadcasts over heads and queries.
    """
    return (src != pad_id)[:, None, None, :]


def make_tgt_mask(tgt, pad_id):
    """Decoder self-attention mask, shape (B, 1, T, T): causal mask AND target padding mask.

    Position t may only attend to positions <= t (no peeking at later summary tokens) that are
    not padding.
    """
    T = tgt.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))  # (T, T)
    not_pad = (tgt != pad_id)[:, None, None, :]                                  # (B, 1, 1, T)
    return causal[None, None, :, :] & not_pad


# --------------------------------------------------------------------------- building blocks

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    Q, K and V are linear projections of the inputs, split into H heads that attend in
    parallel; the head outputs are concatenated and projected back to d_model.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, L, D) -> (B, H, L, Dk)
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key_value, mask=None):
        """
        query:     (B, Tq, D)  - the sequence that asks the questions
        key_value: (B, Tk, D)  - the sequence that is looked at (same as query for self-attention,
                                 the encoder output for encoder-decoder attention)
        mask:      bool, broadcastable to (B, H, Tq, Tk)
        returns:   output (B, Tq, D) and attention weights (B, H, Tq, Tk)
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key_value))
        v = self._split_heads(self.w_v(key_value))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)        # (B, H, Tq, Tk)
        if mask is not None:
            # Equivalent to adding -inf at blocked positions. The most negative finite value is
            # used instead of -inf so a fully blocked row gives a uniform row, not NaN.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)                         # rows sum to 1

        out = self.attn_dropout(weights) @ v                            # (B, H, Tq, Dk)
        B, _, Tq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.n_heads * self.d_k)  # concat heads
        return self.w_o(out), weights


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2, applied to every position separately."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings, added to the token embeddings.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)     # (max_len, 1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float)          # 0, 2, 4, ... = "2i"
        angle = pos / torch.pow(10000.0, two_i / d_model)               # (max_len, D/2)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle)   # even dimensions
        pe[:, 1::2] = torch.cos(angle)   # odd dimensions
        self.register_buffer("pe", pe, persistent=False)  # not a parameter: never trained

    def forward(self, x):
        # x: (B, L, D) -> add the encoding of positions 0..L-1
        return x + self.pe[: x.size(1)]


# --------------------------------------------------------------------------- layers
# Every sub-layer is wrapped as LayerNorm(x + Dropout(Sublayer(x))) (post-norm, as in the paper).

class EncoderLayer(nn.Module):
    """Multi-head self-attention -> position-wise feed-forward network."""

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, src_mask):
        attn_out, _ = self.self_attn(x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Masked multi-head self-attention -> encoder-decoder attention -> feed-forward network."""

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        # 1) masked self-attention over the summary generated so far
        attn_out, _ = self.self_attn(y, y, tgt_mask)
        y = self.norm1(y + self.dropout(attn_out))
        # 2) encoder-decoder attention: queries from the decoder, keys/values from the encoder
        cross_out, cross_weights = self.cross_attn(y, memory, src_mask)
        y = self.norm2(y + self.dropout(cross_out))
        # 3) feed-forward
        y = self.norm3(y + self.dropout(self.ffn(y)))
        return y, cross_weights


# --------------------------------------------------------------------------- full model

class TransformerSummarizer(nn.Module):
    """Encoder-decoder Transformer with one shared vocabulary for articles and summaries."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        # final linear layer: decoder output -> vocabulary logits
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.out_proj.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Embeddings are multiplied by sqrt(d_model) (as in the paper), so start them at
        # std d_model^-0.5 to give the scaled embeddings roughly unit variance.
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[self.cfg.pad_id].zero_()

    def embed_tokens(self, ids):
        x = self.embed(ids) * math.sqrt(self.cfg.d_model)
        return self.emb_dropout(self.pos_enc(x))

    def encode(self, src, src_mask):
        """Article ids (B, S) -> contextual representations ("memory") (B, S, D)."""
        x = self.embed_tokens(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_in, memory, src_mask, tgt_mask):
        """Returns decoder hidden states (B, T, D) and the LAST decoder layer's
        encoder-decoder attention weights (B, H, T, S)."""
        y = self.embed_tokens(tgt_in)
        cross_weights = None
        for layer in self.decoder_layers:
            y, cross_weights = layer(y, memory, tgt_mask, src_mask)
        return y, cross_weights

    def forward(self, src, tgt_in):
        """Teacher-forced forward pass. src: (B, S) article ids; tgt_in: (B, T) summary shifted
        right (starts with the start token). Returns logits (B, T, V) for the next tokens."""
        pad = self.cfg.pad_id
        src_mask = make_src_mask(src, pad)
        tgt_mask = make_tgt_mask(tgt_in, pad)
        memory = self.encode(src, src_mask)
        y, _ = self.decode(tgt_in, memory, src_mask, tgt_mask)
        return self.out_proj(y)

    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=128, return_attention=False):
        """Autoregressive greedy decoding (no teacher forcing).

        Starts from the start token and repeatedly appends the most probable next token until
        every sequence has produced the end token or max_len tokens. The article is encoded
        ONCE and the encoder output is reused at every step.

        Returns generated ids (B, L) without the start token (positions after the end token are
        padding). With return_attention=True it also returns the last decoder layer's
        encoder-decoder attention recorded at each generation step: (B, H, L, S).
        """
        pad = self.cfg.pad_id
        src_mask = make_src_mask(src, pad)
        memory = self.encode(src, src_mask)

        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        step_attention = []
        for _ in range(max_len):
            y, cross_weights = self.decode(ys, memory, src_mask, make_tgt_mask(ys, pad))
            logits = self.out_proj(y[:, -1])        # only the newest position is needed
            logits[:, pad] = float("-inf")          # never generate padding or a second start token
            logits[:, bos_id] = float("-inf")
            next_tok = logits.argmax(dim=-1)
            next_tok = next_tok.masked_fill(finished, pad)   # finished sequences just get padding
            ys = torch.cat([ys, next_tok[:, None]], dim=1)
            if return_attention:
                # attention of the token just generated over the source: (B, H, S)
                step_attention.append(cross_weights[:, :, -1, :].float().cpu())
            finished |= next_tok == eos_id
            if finished.all():
                break

        generated = ys[:, 1:]
        if return_attention:
            return generated, torch.stack(step_attention, dim=2)
        return generated


def count_parameters(model):
    """Number of trainable parameters (tied weights are counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
