"""E1: fertility, parity and byte-fallback of 11 released tokenizers on 22 Indian languages and 5 domains.

Output: results/e1_existing.json, results/e1_existing.png
"""
import time

import numpy as np

from .common import EVAL, INDIC, LANG, read_jsonl, save_json, texts
from .metrics import byte_fallback_rate, load, stats, count_tokens

NAMES = ["gemma-4", "gemma-3", "sarvam-30b", "sarvam-1", "brahmic-131k", "krutrim-2",
         "llama-3.1", "qwen3", "deepseek-v3", "mistral-nemo", "gpt-oss"]
DOMAINS = ["eng_web", "science_sciq", "math_gsm8k", "math_web", "code_humaneval", "code_github",
           "agentic_swe", "agentic_tools", "hinglish", "roman_hin"]


def eval_lang_sets():
    sets = {"eng": texts(EVAL / "lang" / "eng.jsonl")}
    eng_for = {}
    for code in INDIC:
        if not (EVAL / "lang" / f"{code}.jsonl").exists():
            continue  # Dogri: the only "Dogri" text found (Sangraha verified/doi) is English
        rows = read_jsonl(EVAL / "lang" / f"{code}.jsonl")
        sets[code] = [r["text"] for r in rows]
        if rows[0].get("parallel"):
            eng_for[code] = [r["eng"] for r in rows] if "eng" in rows[0] else None  # None = FLORES eng
    return sets, eng_for


def evaluate(tok, sets, eng_for, domains):
    out = {"vocab": tok.get_vocab_size(), "lang": {}, "domain": {}}
    eng_tok = count_tokens(tok, sets["eng"])
    for code, xs in sets.items():
        s = stats(tok, xs)
        if code in eng_for:
            ref = eng_tok if eng_for[code] is None else count_tokens(tok, eng_for[code])
            s["parity"] = s["tokens"] / ref
        elif code == "eng":
            s["parity"] = 1.0
        if code != "eng":
            s["byte_fallback"] = byte_fallback_rate(tok, xs)
        out["lang"][code] = s
    for d, xs in domains.items():
        out["domain"][d] = stats(tok, xs)
    tier1 = [c for c in INDIC if LANG[c]["tier"] == 1]
    out["mean_fertility_tier1"] = float(np.mean([out["lang"][c]["fertility"] for c in tier1 if c in sets]))
    out["mean_fertility_tier2"] = float(np.mean([out["lang"][c]["fertility"] for c in INDIC if LANG[c]["tier"] == 2 and c in sets]))
    out["mean_parity_22"] = float(np.mean([out["lang"][c]["parity"] for c in INDIC if c in sets and "parity" in out["lang"][c]]))
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(res, key=lambda n: res[n]["mean_parity_22"])
    cols = ["eng"] + [c for c in INDIC if c in res[names[0]]["lang"]]
    m = np.array([[res[n]["lang"][c]["fertility"] for c in cols] for n in names])
    fig, (a, b) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]})
    im = a.imshow(np.log2(m), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=4)
    for i in range(len(names)):
        for j in range(len(cols)):
            a.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=7)
    a.set_xticks(range(len(cols)), cols, rotation=0, fontsize=8)
    a.set_yticks(range(len(names)), [f"{n} ({res[n]['vocab'] // 1000}K)" for n in names], fontsize=8)
    a.set_title("Fertility (tokens per word) on FLORES devtest (brx, gom: FLORES-IN). Colour = log2. Rows sorted by mean parity")
    fig.colorbar(im, ax=a, label="log2 fertility")
    doms = ["eng_web", "science_sciq", "math_gsm8k", "code_github", "agentic_swe", "agentic_tools", "hinglish"]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        b.bar(np.arange(len(doms)) + i * w, [res[n]["domain"][d]["bytes_per_token"] for d in doms], w, label=n)
    b.set_xticks(np.arange(len(doms)) + 0.4, doms)
    b.set_ylabel("bytes per token (higher = better)")
    b.legend(fontsize=7, ncol=6)
    b.set_title("Compression on English, science, math, code, agent traces and Hinglish")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    sets, eng_for = eval_lang_sets()
    domains = {d: texts(EVAL / "domain" / f"{d}.jsonl") for d in DOMAINS}
    res = {}
    for n in NAMES:
        t0 = time.time()
        res[n] = evaluate(load(n), sets, eng_for, domains)
        r = res[n]
        print(f"{n:14s} V={r['vocab']:>7,}  T1 fert {r['mean_fertility_tier1']:.2f}  T2 fert {r['mean_fertility_tier2']:.2f}  "
              f"parity22 {r['mean_parity_22']:.2f}  eng {r['lang']['eng']['fertility']:.2f}  "
              f"code B/t {r['domain']['code_github']['bytes_per_token']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    save_json("e1_existing", res)
    from .common import RESULTS
    plot(res, RESULTS / "e1_existing.png")


if __name__ == "__main__":
    main()
