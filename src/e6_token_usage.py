"""E6: are all vocabulary rows trained? Token frequencies on the tokenizer's own training mix.

Peer reports set a gate such as "≥ 99.5% of tokens occur ≥ 1,000 times in the final mix". A BPE
merge is learned only from frequent pairs, but later merges can absorb an earlier token so that it
almost never appears in the final encoding (an orphan; Land & Bartolo 2024, "Fishing for Magikarp").
Orphan rows get no gradient and are glitch-token risks. This step encodes the India-first training
sample with each truncated vocabulary and counts how often each id fires.

Run:  python -m src.e6_token_usage
"""
from collections import Counter

import numpy as np

from src import bpe
from src.common import QUICK, TOK, save_json
from src.e2_vocab_sweep import TOTAL_CHARS

SIZES = [131_072, 196_608, 262_144, 393_216]
RUN_TOKENS = 20e12
SCRIPTS = [("Latin", 0x0000, 0x024F), ("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF),
           ("Gurmukhi", 0x0A00, 0x0A7F), ("Gujarati", 0x0A80, 0x0AFF), ("Odia", 0x0B00, 0x0B7F),
           ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF),
           ("Malayalam", 0x0D00, 0x0D7F), ("Arabic", 0x0600, 0x06FF), ("Meetei/Ol Chiki", 0x1C50, 0x1C7F),
           ("Meetei/Ol Chiki", 0xABC0, 0xABFF)]


def script_of(text):
    for ch in text:
        if ch.isalpha():
            cp = ord(ch)
            for name, lo, hi in SCRIPTS:
                if lo <= cp <= hi:
                    return name
            return "Other"
    return "Symbols/digits"


def main():
    chars = TOTAL_CHARS // 30 if QUICK else TOTAL_CHARS
    docs, _ = bpe.build_mix("india", chars)          # seed 0 = the tokenizer's own training sample
    full = bpe.load_file(TOK / "ours_india_384k.json")
    res = {"sample_chars": chars, "run_tokens": RUN_TOKENS, "sizes": {}}
    for V in SIZES:
        tok = bpe.truncate(full, V) if V < full.get_vocab_size() else full
        counts = np.zeros(tok.get_vocab_size(), dtype=np.int64)
        for i in range(0, len(docs), 512):
            for e in tok.encode_batch(docs[i:i + 512], add_special_tokens=False):
                np.add.at(counts, e.ids, 1)
        n_tok = int(counts.sum())
        vocab = {i: s for s, i in tok.get_vocab().items()}
        learned = np.array([vocab[i] not in bpe.SPECIALS for i in range(len(counts))])   # specials never occur in text
        counts = counts[learned]
        ids = np.flatnonzero(learned)
        per_run = counts / n_tok * RUN_TOKENS               # expected occurrences in a 20 T-token run
        zero_ids = ids[counts == 0]
        by_script = Counter(script_of(tok.decode([int(i)])) for i in zero_ids)
        row = {"sample_tokens": n_tok, "learned_tokens": int(learned.sum()), "zero_count": int(len(zero_ids)),
               "zero_share": float((counts == 0).mean()),
               "lt10_share": float((counts < 10).mean()),
               "run_lt_1e5_share": float((per_run < 1e5).mean()),
               "zero_by_script": dict(by_script.most_common()),
               "zero_examples": [vocab[int(i)] for i in zero_ids[:15]]}
        res["sizes"][V] = row
        print(V, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "zero_examples"},
              flush=True)
    save_json("e6_token_usage", res)


if __name__ == "__main__":
    main()
