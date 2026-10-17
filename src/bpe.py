"""Byte-level BPE training, truncation to a smaller vocabulary, and the training mixes."""
import json
import random

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import INDIC, LANG, TRAIN, script_share, texts

# A document enters the tokenizer sample only if most of its letters are in the language's own script.
# This drops legacy-font mojibake (Sangraha kas) and mislabelled English (Sangraha doi).
MIN_SCRIPT_SHARE = 0.5

# o200k_base pattern with one change: \p{N} (single digits) in place of \p{N}{1,3}.
# The letter classes include \p{M}, so an Indic vowel sign or virama stays inside its word.
PAT_OURS = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# GPT-2 pattern: \p{L}+ does not include \p{M}, so every Indic vowel sign starts a new pre-token.
PAT_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|tool_call|>", "<|tool_result|>",
            "<think>", "</think>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]


def new_tokenizer(pattern=PAT_OURS):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok  # no normalizer: decode(encode(x)) == x exactly. NFC is a cleaning step, not a tokenizer step.


def train(docs, vocab_size, pattern=PAT_OURS, max_token_bytes=32):
    tok = new_tokenizer(pattern)
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), max_token_length=max_token_bytes)
    tok.train_from_iterator(docs, trainer=tr, length=len(docs))
    return tok


def load_file(path):
    return Tokenizer.from_file(str(path))


def truncate(tok, k):
    """The first k entries of a BPE vocabulary are the BPE vocabulary of size k (merges are learned in order).

    tests/test_bpe.py checks this against a tokenizer trained directly at size k.
    """
    d = json.loads(tok.to_str())
    vocab = d["model"]["vocab"]
    base = sum(1 for i in vocab.values() if i < len(SPECIALS) + 256)
    d["model"]["vocab"] = {t: i for t, i in vocab.items() if i < k}
    d["model"]["merges"] = d["model"]["merges"][: k - base]
    return Tokenizer.from_str(json.dumps(d))


# ---------------------------------------------------------------- mixes
# Sangraha verified size in GB per language (Hugging Face file listing, 17 Oct 2026). A proxy for web supply.
SANGRAHA_GB = {"hin": 37.34, "ben": 29.9, "tam": 18.71, "npi": 16.86, "tel": 15.1, "mar": 12.78, "urd": 12.78,
               "mal": 12.17, "guj": 9.93, "kan": 7.65, "san": 4.18, "pan": 4.14, "ory": 3.92, "snd": 1.41,
               "asm": 0.98, "mai": 0.05, "gom": 0.03, "brx": 0.0045, "mni": 0.0046, "kas": 0.0008,
               "sat": 0.0002, "doi": 0.0001}

# Shares of characters in the tokenizer training sample.
MIXES = {
    # India-first: the planned pretraining mix, with Indic up-weighted.
    "india": {"eng_web": 0.26, "code": 0.24, "math": 0.07, "agentic": 0.03, "indic": 0.36, "roman": 0.04},
    # English-first: the share of Indic text in a typical web crawl.
    "english": {"eng_web": 0.60, "code": 0.25, "math": 0.07, "agentic": 0.03, "indic": 0.05, "roman": 0.0},
}


def indic_shares(temperature=0.3, floor=0.01):
    """p_i proportional to size_i^T (T < 1 flattens toward small languages), with a floor per language."""
    w = {c: SANGRAHA_GB[c] ** temperature for c in INDIC}
    z = sum(w.values())
    p = {c: max(v / z, floor) for c, v in w.items()}
    z = sum(p.values())
    return {c: v / z for c, v in p.items()}


def take(pool, n_chars, rng):
    pool = pool[:]
    rng.shuffle(pool)
    out, used = [], 0
    for t in pool:
        if used >= n_chars:
            break
        out.append(t)
        used += len(t)
    return out, used


def build_mix(name, total_chars, seed=0):
    """Return (docs, report). Short pools give what they have; the report records each shortfall."""
    rng = random.Random(seed)
    shares = MIXES[name]
    docs, report = [], {}
    for part in ("eng_web", "code", "math", "agentic"):
        got, used = take(texts(TRAIN / f"{part}.jsonl"), shares[part] * total_chars, rng)
        docs += got
        report[part] = {"want": shares[part] * total_chars, "got": used}
    for code, p in indic_shares().items():
        want = shares["indic"] * total_chars * p
        pool = texts(TRAIN / "indic" / f"{code}.jsonl")
        clean = [t for t in pool if script_share(t, LANG[code]["script"]) >= MIN_SCRIPT_SHARE]
        got, used = take(clean, want, rng)
        docs += got
        report[code] = {"want": want, "got": used, "pool_docs": len(pool), "script_ok_docs": len(clean)}
    if shares["roman"]:
        roman = sorted((TRAIN / "roman").glob("*.jsonl"))
        for f in roman:
            want = shares["roman"] * total_chars / len(roman)
            got, used = take(texts(f), want, rng)
            docs += got
            report["roman_" + f.stem] = {"want": want, "got": used}
    rng.shuffle(docs)
    return docs, report
