"""E3: English cleaning rules on clean Indic text, and the language-aware fix.

Sangraha "verified" text was already cleaned and checked by AI4Bharat. Thus every rejection here is a
false positive: good text that an English-tuned pipeline would delete. FineWeb-Edu English is the control.

Output: results/e3_cleaning.json, results/e3_cleaning.png
"""
import random
import re
import unicodedata
from collections import Counter

import numpy as np

from .common import INDIC, LANG, RESULTS, TRAIN, save_json, texts

N_DOCS = 2000
EN_STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}
EN_END = (".", "!", "?", '"')
INDIC_END = EN_END + ("।", "॥", "۔", "؟", "|", ")", "”", "’", ":")
ZW = ("‌", "‍")  # ZWNJ, ZWJ: part of correct spelling in several Indic scripts


def is_letter_en(ch):
    return ch.isalpha()  # Python: categories L* only. Vowel signs and virama (M*) are not letters.


def is_letter_uni(ch):
    return unicodedata.category(ch)[0] in "LM"


def doc_features(doc, letter, ends):
    ws = doc.split()
    lines = [l for l in doc.split("\n") if l.strip()]
    nchar = sum(1 for c in doc if not c.isspace())
    return {
        "n_words": len(ws),
        "mean_word_len": np.mean([len(w) for w in ws]) if ws else 0,
        "letter_ratio": sum(1 for c in doc if letter(c)) / max(nchar, 1),
        "line_end_ratio": sum(l.rstrip().endswith(ends) for l in lines) / max(len(lines), 1),
        "words": ws,
        "lines": lines,
    }


def english_rules(doc):
    """Gopher (Rae et al., 2021), C4 (Raffel et al., 2020) and FineWeb (Penedo et al., 2024) rules as published for English."""
    f = doc_features(doc, is_letter_en, EN_END)
    low = [w.lower().strip(".,!?;:\"'") for w in f["words"]]
    c4_lines = [l for l in f["lines"] if l.rstrip().endswith(EN_END) and len(l.split()) >= 3]
    return {
        "gopher_words_50": f["n_words"] < 50,
        "gopher_mean_word_len_3_10": not (3 <= f["mean_word_len"] <= 10),
        "gopher_stopwords_2": sum(w in EN_STOP for w in low) < 2,
        "letter_ratio_0.75": f["letter_ratio"] < 0.75,  # alphabetic-character share, common in web filters
        "c4_terminal_punct_3_lines": len(c4_lines) < 3,
        "fineweb_line_punct_0.12": f["line_end_ratio"] < 0.12,
    }


def calibrate(docs):
    """Per-language thresholds from the language's own clean text (the FineWeb-2 approach)."""
    feats = [doc_features(d, is_letter_uni, INDIC_END) for d in docs]
    counts = Counter(w for f in feats for w in f["words"])
    stop = {w for w, _ in counts.most_common(20)}
    mwl = [f["mean_word_len"] for f in feats if f["n_words"]]
    return {"stop": stop, "mwl_lo": float(np.percentile(mwl, 0.5)), "mwl_hi": float(np.percentile(mwl, 99.5)),
            "letter_lo": float(np.percentile([f["letter_ratio"] for f in feats], 0.5))}


def aware_rules(doc, cal):
    f = doc_features(doc, is_letter_uni, INDIC_END)
    return {
        "gopher_words_50": f["n_words"] < 50,
        "gopher_mean_word_len_3_10": not (cal["mwl_lo"] <= f["mean_word_len"] <= cal["mwl_hi"]),
        "gopher_stopwords_2": sum(w in cal["stop"] for w in f["words"]) < 2,
        "letter_ratio_0.75": f["letter_ratio"] < cal["letter_lo"],
        "c4_terminal_punct_3_lines": False,  # replaced by the FineWeb ratio with the language's own end marks
        "fineweb_line_punct_0.12": f["line_end_ratio"] < 0.12,
    }


def rates(flags):
    keys = flags[0].keys()
    r = {k: float(np.mean([f[k] for f in flags])) for k in keys}
    r["any"] = float(np.mean([any(f.values()) for f in flags]))
    return r


def unicode_hygiene(docs):
    return {
        "not_nfc": float(np.mean([unicodedata.normalize("NFC", d) != d for d in docs])),
        "has_zwj_zwnj": float(np.mean([any(z in d for z in ZW) for d in docs])),
    }


def main():
    rng = random.Random(0)
    res = {}
    langs = [("eng", TRAIN / "eng_web.jsonl")] + [(c, TRAIN / "indic" / f"{c}.jsonl") for c in INDIC]
    for code, path in langs:
        docs = texts(path)
        docs = rng.sample(docs, min(N_DOCS, len(docs)))
        half = len(docs) // 2
        cal_docs, test_docs = docs[:half], docs[half:]  # calibrate on one half, measure on the other
        cal = calibrate(cal_docs)
        res[code] = {
            "n_test": len(test_docs),
            "english": rates([english_rules(d) for d in test_docs]),
            "aware": rates([aware_rules(d, cal) for d in test_docs]),
            "mwl_band": [cal["mwl_lo"], cal["mwl_hi"]],
            **unicode_hygiene(test_docs),
        }
        r = res[code]
        print(f"{code}: English rules reject {r['english']['any']:.1%}  language-aware reject {r['aware']['any']:.1%}  "
              f"(stopwords {r['english']['gopher_stopwords_2']:.0%}, word len {r['english']['gopher_mean_word_len_3_10']:.0%}, "
              f"letters {r['english']['letter_ratio_0.75']:.0%}, C4 {r['english']['c4_terminal_punct_3_lines']:.0%})  "
              f"not NFC {r['not_nfc']:.1%}  ZWJ/ZWNJ {r['has_zwj_zwnj']:.1%}", flush=True)
    save_json("e3_cleaning", res)
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = list(res)
    x = np.arange(len(codes))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x - 0.2, [100 * res[c]["english"]["any"] for c in codes], 0.4, label="English rules (Gopher + C4 + FineWeb)")
    ax.bar(x + 0.2, [100 * res[c]["aware"]["any"] for c in codes], 0.4, label="Same rules, language-aware")
    ax.set_xticks(x, codes)
    ax.set_ylabel("% of clean documents rejected")
    ax.set_title("False rejections on already-clean text (Sangraha verified; English = FineWeb-Edu)")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_cleaning.png", dpi=120)


if __name__ == "__main__":
    main()
