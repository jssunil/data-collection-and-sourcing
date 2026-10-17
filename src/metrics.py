"""Tokenizer metrics: fertility, bytes per token, parity, and the tokenizer loader."""
import re
from pathlib import Path

from tokenizers import Tokenizer

from .common import TOK

WORD = re.compile(r"\S+")


def words(text):
    """Whitespace words. All 22 scheduled languages separate words with spaces, so this is a fair unit."""
    return len(WORD.findall(text))


def load(name_or_path):
    p = TOK / f"{name_or_path}.json"
    if not p.exists():
        p = Path(name_or_path)
    return Tokenizer.from_file(str(p))


def count_tokens(tok, texts, batch=256):
    """Number of tokens with no special tokens (no BOS/EOS)."""
    n = 0
    for i in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[i:i + batch], add_special_tokens=False)
        n += sum(len(e.ids) for e in enc)
    return n


def stats(tok, texts):
    """Totals for one text set: tokens, words, chars, bytes, and the derived ratios."""
    t = count_tokens(tok, texts)
    w = sum(words(x) for x in texts)
    c = sum(len(x) for x in texts)
    b = sum(len(x.encode("utf-8")) for x in texts)
    return {
        "tokens": t, "words": w, "chars": c, "bytes": b,
        "fertility": t / max(w, 1),          # tokens per word (lower is better)
        "bytes_per_token": b / max(t, 1),    # higher is better
        "tokens_per_kchar": 1000 * t / max(c, 1),
    }


def parity(lang_tokens, eng_tokens):
    """Premium over English on parallel text (Petrov et al., 2023). 1.0 = same cost as English."""
    return lang_tokens / eng_tokens


def byte_fallback_rate(tok, texts, limit=200):
    """Share of tokens that are a broken UTF-8 piece (decode to U+FFFD alone): the byte-fallback regime."""
    bad = total = 0
    for x in texts[:limit]:
        ids = tok.encode(x, add_special_tokens=False).ids
        total += len(ids)
        bad += sum("�" in tok.decode([i]) for i in ids)
    return bad / max(total, 1)
