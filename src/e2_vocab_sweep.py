"""E2: how fertility falls with vocabulary size, and which size a 40B model should use.

  a. Train one byte-level BPE at V_MAX on the India-first mix. Truncate it to each size in SIZES
     (a BPE vocabulary of size k is the first k entries of a larger one; see tests/test_bpe.py).
  b. The same on an English-first mix (5% Indic): shows that the tokenizer data, not the size, sets Indic cost.
  c. Pre-tokenizer ablation at 64K: GPT-2 pattern (splits at Indic vowel signs) against ours.
  d. Cost model: training FLOPs per byte of the planned pretraining mix, as a function of V.

Output: results/e2_vocab_sweep.json, results/e2_vocab_sweep.png
"""
import sys
import time

import numpy as np

from . import bpe
from .common import EVAL, INDIC, LANG, QUICK, RESULTS, TOK, load_json, save_json, texts
from .e1_existing import DOMAINS, eval_lang_sets
from .metrics import count_tokens, stats

M = 1_000_000
TOTAL_CHARS = (20 if QUICK else 300) * M
V_MAX = 32_768 if QUICK else 393_216
SIZES = [16_384, 32_768] if QUICK else [32_768, 49_152, 65_536, 98_304, 131_072, 163_840, 196_608, 262_144, 327_680, 393_216]

# Planned pretraining mix by bytes (report section 2). Used to weight compression into one cost number.
PLAN_BYTES = {"eng_web": 0.43, "code_github": 0.22, "math_web": 0.08, "agentic_swe": 0.02, "indic": 0.22, "hinglish": 0.03}

# Assumed 40B dense model: d_model 7168, 60 layers, 8K training context.
D_MODEL, LAYERS, CTX = 7168, 60, 8192
N_BODY = 12 * LAYERS * D_MODEL ** 2  # 37.0 B non-embedding weights


def sweep(tok, sizes, sets, eng_for, domains):
    rows = []
    for k in sizes:
        t = bpe.truncate(tok, k) if k < tok.get_vocab_size() else tok
        eng = count_tokens(t, sets["eng"])
        r = {"V": k, "lang": {}, "domain": {}}
        for code, xs in sets.items():
            s = stats(t, xs)
            r["lang"][code] = {"fertility": s["fertility"], "bytes_per_token": s["bytes_per_token"], "tokens": s["tokens"]}
            if code in eng_for and eng_for[code] is None:
                r["lang"][code]["parity"] = s["tokens"] / eng
        for d, xs in domains.items():
            s = stats(t, xs)
            r["domain"][d] = {"bytes_per_token": s["bytes_per_token"], "fertility": s["fertility"]}
        r["mean_fertility_tier1"] = float(np.mean([r["lang"][c]["fertility"] for c in INDIC if LANG[c]["tier"] == 1 and c in sets]))
        r["mean_fertility_tier2"] = float(np.mean([r["lang"][c]["fertility"] for c in INDIC if LANG[c]["tier"] == 2 and c in sets]))
        # Indic bytes per token: all Indic sentences pooled.
        r["indic_bytes_per_token"] = (sum(len(x.encode()) for c in INDIC if c in sets for x in sets[c]) /
                                      sum(r["lang"][c]["tokens"] for c in INDIC if c in sets))
        rows.append(r)
        print(f"  V={k:>7,}  eng {r['lang']['eng']['fertility']:.3f}  T1 {r['mean_fertility_tier1']:.3f}  "
              f"T2 {r['mean_fertility_tier2']:.3f}  code B/t {r['domain']['code_github']['bytes_per_token']:.3f}  "
              f"math B/t {r['domain']['math_web']['bytes_per_token']:.3f}", flush=True)
    return rows


def cost_model(rows):
    """Training FLOPs per byte of the planned mix, relative to the smallest V.

    FLOPs per token = 6 N_body + 6 d V (output layer) + 12 L d CTX (attention scores).
    Tokens per byte = sum over parts of share / bytes_per_token.
    """
    out = []
    for r in rows:
        bpt = dict(r["domain"])
        tok_per_byte = 0
        for part, share in PLAN_BYTES.items():
            b = r["indic_bytes_per_token"] if part == "indic" else bpt[part]["bytes_per_token"]
            tok_per_byte += share / b
        flops_tok = 6 * N_BODY + 6 * D_MODEL * r["V"] + 12 * LAYERS * D_MODEL * CTX
        out.append({"V": r["V"], "tokens_per_byte": tok_per_byte, "flops_per_byte": flops_tok * tok_per_byte,
                    "head_share": 6 * D_MODEL * r["V"] / flops_tok, "embed_params_B": 2 * D_MODEL * r["V"] / 1e9})
    base = out[0]["flops_per_byte"]
    for o in out:
        o["rel_cost"] = o["flops_per_byte"] / base
    return out


def unused_share(tok, docs):
    """Share of the vocabulary that never fires on a held-out sample (untrained rows, glitch-token risk)."""
    seen = set()
    for i in range(0, len(docs), 256):
        for e in tok.encode_batch(docs[i:i + 256], add_special_tokens=False):
            seen.update(e.ids)
    return 1 - len(seen) / tok.get_vocab_size()


def main():
    parts = sys.argv[1:] or ["a", "b", "c"]
    sets, eng_for = eval_lang_sets()
    domains = {d: texts(EVAL / "domain" / f"{d}.jsonl") for d in DOMAINS}
    res = load_json("e2_vocab_sweep") if (RESULTS / "e2_vocab_sweep.json").exists() else {}

    for mix, part in (("india", "a"), ("english", "b")):
        if part not in parts:
            continue
        path = TOK / f"ours_{mix}_{V_MAX // 1024}k.json"
        docs, report = bpe.build_mix(mix, TOTAL_CHARS)
        if path.exists():
            tok = bpe.load_file(path)
        else:
            t0 = time.time()
            tok = bpe.train(docs, V_MAX)
            tok.save(str(path))
            print(f"trained {mix} V={V_MAX:,} on {sum(map(len, docs)) / M:.0f} M chars in {time.time() - t0:.0f}s", flush=True)
        print(f"sweep: {mix} mix", flush=True)
        res[mix] = {"mix_report": report, "rows": sweep(tok, SIZES, sets, eng_for, domains)}
        res[mix]["cost"] = cost_model(res[mix]["rows"])
        held = sum(([x for x in v] for v in sets.values()), []) + sum(domains.values(), [])
        res[mix]["unused_share"] = {k: unused_share(bpe.truncate(tok, k) if k < V_MAX else tok, held) for k in SIZES}
        save_json("e2_vocab_sweep", res)

    if "c" in parts:  # pre-tokenizer ablation on Indic + English only, 64K
        docs, _ = bpe.build_mix("india", TOTAL_CHARS // 4, seed=1)
        out = {}
        for name, pat in (("gpt2", bpe.PAT_GPT2), ("ours", bpe.PAT_OURS)):
            tok = bpe.train(docs, 65_536, pattern=pat)
            out[name] = {c: stats(tok, sets[c])["fertility"] for c in sets}
            print(f"pretok {name}: eng {out[name]['eng']:.3f}  hin {out[name]['hin']:.3f}  tam {out[name]['tam']:.3f}  "
                  f"mal {out[name]['mal']:.3f}", flush=True)
        res["pretok"] = out
        save_json("e2_vocab_sweep", res)
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for mix, ls in (("india", "-"), ("english", "--")):
        if mix not in res:
            continue
        rows = res[mix]["rows"]
        V = [r["V"] / 1024 for r in rows]
        ax[0, 0].plot(V, [r["mean_fertility_tier1"] for r in rows], ls, marker="o", label=f"Indic tier 1 ({mix} mix)")
        ax[0, 0].plot(V, [r["mean_fertility_tier2"] for r in rows], ls, marker="s", label=f"Indic tier 2 ({mix} mix)")
        ax[0, 0].plot(V, [r["lang"]["eng"]["fertility"] for r in rows], ls, marker="^", label=f"English ({mix} mix)")
        for d, m in (("code_github", "o"), ("math_web", "s"), ("agentic_swe", "^"), ("eng_web", "x")):
            ax[0, 1].plot(V, [r["domain"][d]["bytes_per_token"] for r in rows], ls, marker=m, label=f"{d} ({mix})")
        c = res[mix]["cost"]
        ax[1, 0].plot(V, [o["rel_cost"] for o in c], ls, marker="o", label=f"{mix} mix")
    if "india" in res:
        rows = res["india"]["rows"]
        V = [r["V"] / 1024 for r in rows]
        for code in [c for c in INDIC if c in rows[0]["lang"]]:
            ax[1, 1].plot(V, [r["lang"][code]["fertility"] for r in rows], lw=1, alpha=.8)
            ax[1, 1].annotate(code, (V[-1], rows[-1]["lang"][code]["fertility"]), fontsize=7)
    ax[0, 0].set(xscale="log", xlabel="vocabulary (K)", ylabel="tokens per word", title="Fertility on FLORES devtest")
    ax[0, 1].set(xscale="log", xlabel="vocabulary (K)", ylabel="bytes per token", title="Compression on held-out domains")
    ax[1, 0].set(xscale="log", xlabel="vocabulary (K)", ylabel="relative FLOPs per byte",
                 title="40B training cost per byte of the planned mix")
    ax[1, 1].set(xscale="log", xlabel="vocabulary (K)", ylabel="tokens per word", title="Each Indic language (India-first mix)")
    for a in ax.flat:
        a.grid(alpha=.3)
        a.set_xticks([32, 64, 128, 256, 384], ["32", "64", "128", "256", "384"])
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].legend(fontsize=6, ncol=2)
    ax[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_vocab_sweep.png", dpi=120)


if __name__ == "__main__":
    main()
