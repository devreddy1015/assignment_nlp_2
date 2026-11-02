"""
Quick correctness checks for the from-scratch Transformer (no dataset needed).

1. no built-in Transformer / attention modules are used
2. causal mask: changing a later summary token never changes the outputs at earlier positions
3. padding masks: adding extra padding to the article or summary does not change the outputs
4. attention weights are proper distributions and ignore padding
5. copy task: a tiny model learns to copy random sequences and greedy decoding reproduces them
   (exercises cross-attention, positional encodings, training loop and generation end to end)

Run:  python sanity_checks.py
"""
import torch
import torch.nn as nn

from model import ModelConfig, TransformerSummarizer, count_parameters, make_src_mask
from utils import get_device, set_seed

PAD, BOS, EOS = 0, 1, 2


def tiny_model(vocab=20, **kw):
    cfg = ModelConfig(vocab_size=vocab, d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                      d_ff=128, dropout=0.0, max_len=64, pad_id=PAD, **kw)
    return TransformerSummarizer(cfg)


def check_no_builtin_modules():
    banned = (nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
              nn.TransformerEncoderLayer, nn.TransformerDecoderLayer, nn.MultiheadAttention)
    model = tiny_model()
    assert not any(isinstance(m, banned) for m in model.modules())
    print("ok  no built-in Transformer/attention modules")


@torch.no_grad()
def check_causal_mask():
    model = tiny_model().eval()
    src = torch.randint(3, 20, (2, 10))
    tgt = torch.randint(3, 20, (2, 8))
    out1 = model(src, tgt)
    for k in range(1, 8):
        tgt2 = tgt.clone()
        tgt2[:, k:] = torch.randint(3, 20, tgt2[:, k:].shape)   # change the "future"
        out2 = model(src, tgt2)
        assert torch.allclose(out1[:, :k], out2[:, :k], atol=1e-5), f"position < {k} saw the future"
    print("ok  causal mask: earlier positions never see later tokens")


@torch.no_grad()
def check_padding_masks():
    model = tiny_model().eval()
    src = torch.randint(3, 20, (2, 10))
    tgt = torch.randint(3, 20, (2, 6))
    out = model(src, tgt)
    src_padded = torch.cat([src, torch.full((2, 7), PAD)], dim=1)
    tgt_padded = torch.cat([tgt, torch.full((2, 5), PAD)], dim=1)
    out_padded = model(src_padded, tgt_padded)
    assert torch.allclose(out, out_padded[:, :6], atol=1e-5), "padding changed the outputs"
    print("ok  padding masks: extra padding in article or summary changes nothing")


@torch.no_grad()
def check_attention_weights():
    model = tiny_model().eval()
    src = torch.randint(3, 20, (2, 10))
    src[0, 6:] = PAD
    src_mask = make_src_mask(src, PAD)
    memory = model.encode(src, src_mask)
    _, attn = model.decoder_layers[-1].cross_attn(memory[:, :3], memory, src_mask)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    assert attn[0, :, :, 6:].abs().max() < 1e-6, "attention on padding"
    print("ok  attention rows sum to 1 and padding gets zero attention")


def check_copy_task(device):
    """Learn to copy sequences of 5-12 random tokens; greedy decoding must reproduce them."""
    set_seed(0)
    vocab = 20
    model = tiny_model(vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    def batch(n=64):
        lengths = torch.randint(5, 13, (n,))
        src = torch.full((n, 12), PAD)
        tgt = torch.full((n, 14), PAD)
        for i, L in enumerate(lengths):
            seq = torch.randint(3, vocab, (L,))
            src[i, :L] = seq
            tgt[i, 0], tgt[i, 1:L + 1], tgt[i, L + 1] = BOS, seq, EOS
        return src.to(device), tgt.to(device)

    model.train()
    for step in range(3000):
        src, tgt = batch()
        logits = model(src, tgt[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    src, tgt = batch(200)
    gen = model.greedy_decode(src, BOS, EOS, max_len=14)
    correct = 0
    for i in range(len(src)):
        expected = tgt[i, 1:][tgt[i, 1:] != PAD].tolist()                 # sequence + EOS
        got = gen[i][: len(expected)].tolist()
        correct += got == expected
    acc = correct / len(src)
    assert acc > 0.9, f"copy-task exact-match accuracy only {acc:.2f} (final loss {loss.item():.3f})"
    print(f"ok  copy task: {acc:.0%} of sequences copied exactly by greedy decoding (loss {loss.item():.4f})")


if __name__ == "__main__":
    device = get_device()
    check_no_builtin_modules()
    check_causal_mask()
    check_padding_masks()
    check_attention_weights()
    check_copy_task(device)
    print(f"example parameter count (tiny test model): {count_parameters(tiny_model()):,}")
    print("All sanity checks passed.")
