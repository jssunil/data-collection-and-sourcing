"""Decisions from E1 + E2: chosen vocabulary size, per-language table, cost sensitivity.

Writes "chosen_V" into results/e2_vocab_sweep.json (E4 reads it) and results/summary.json (the report reads it).
"""
from .common import INDIC, LANG, load_json, save_json
from .e2_vocab_sweep import CTX, D_MODEL, LAYERS, N_BODY

CHOSEN_V = 262_144
COMPARE = ["gemma-4", "sarvam-30b", "brahmic-131k", "gpt-oss", "llama-3.1"]


def cost_curve(rows, indic_share):
    """Relative training FLOPs per byte when the Indic share of bytes is indic_share (rest scaled)."""
    base = {"eng_web": 0.43, "code_github": 0.22, "math_web": 0.08, "agentic_swe": 0.02, "hinglish": 0.03}
    z = sum(base.values())
    out = []
    for r in rows:
        tpb = indic_share / r["indic_bytes_per_token"]
        tpb += sum((1 - indic_share) * s / z / r["domain"][d]["bytes_per_token"] for d, s in base.items())
        out.append((6 * N_BODY + 6 * D_MODEL * r["V"] + 12 * LAYERS * D_MODEL * CTX) * tpb)
    return [o / out[0] for o in out]


def main():
    e1 = load_json("e1_existing")
    e2 = load_json("e2_vocab_sweep")
    rows = {r["V"]: r for r in e2["india"]["rows"]}
    ours = rows[CHOSEN_V]
    langs = ["eng"] + [c for c in INDIC if c in ours["lang"]]
    table = []
    for c in langs:
        row = {"lang": c, "name": "English" if c == "eng" else LANG[c]["name"],
               "tier": 0 if c == "eng" else LANG[c]["tier"], "ours": ours["lang"][c]["fertility"]}
        for n in COMPARE:
            row[n] = e1[n]["lang"][c]["fertility"]
        row["ours_parity"] = ours["lang"][c].get("parity")
        row["gemma4_parity"] = e1["gemma-4"]["lang"][c].get("parity")
        table.append(row)
    domains = {}
    for d in ["eng_web", "science_sciq", "math_gsm8k", "math_web", "code_humaneval", "code_github",
              "agentic_swe", "agentic_tools", "hinglish", "roman_hin"]:
        domains[d] = {"ours": ours["domain"][d]["bytes_per_token"],
                      **{n: e1[n]["domain"][d]["bytes_per_token"] for n in COMPARE}}
    sens = {f"{s:.2f}": dict(zip([r["V"] for r in e2["india"]["rows"]], cost_curve(e2["india"]["rows"], s)))
            for s in (0.0, 0.1, 0.22, 0.4)}
    best = {s: min(c, key=c.get) for s, c in sens.items()}
    out = {"chosen_V": CHOSEN_V, "per_language": table, "domains": domains, "cost_sensitivity": sens,
           "best_V_by_indic_share": best,
           "tier_means": {"ours": (ours["mean_fertility_tier1"], ours["mean_fertility_tier2"]),
                          **{n: (e1[n]["mean_fertility_tier1"], e1[n]["mean_fertility_tier2"]) for n in COMPARE}}}
    if "english" in e2:
        en = {r["V"]: r for r in e2["english"]["rows"]}[CHOSEN_V]
        out["english_mix_at_chosen"] = {"tier1": en["mean_fertility_tier1"], "tier2": en["mean_fertility_tier2"],
                                        "eng": en["lang"]["eng"]["fertility"],
                                        "code": en["domain"]["code_github"]["bytes_per_token"]}
    if "pretok" in e2:
        p = e2["pretok"]
        out["pretok"] = {k: {c: p[k][c] for c in ("eng", "hin", "ben", "tam", "mal", "urd")} for k in p}
    e2["chosen_V"] = CHOSEN_V
    save_json("e2_vocab_sweep", e2)
    save_json("summary", out)
    print(f"{'lang':5s} {'ours':>6s} " + " ".join(f"{n:>12s}" for n in COMPARE) + "  parity(ours/gemma4)")
    for r in table:
        p = f"{r['ours_parity']:.2f}/{r['gemma4_parity']:.2f}" if r["ours_parity"] else ""
        print(f"{r['lang']:5s} {r['ours']:6.2f} " + " ".join(f"{r[n]:12.2f}" for n in COMPARE) + "  " + p)
    for d, v in domains.items():
        print(f"{d:15s} " + " ".join(f"{k}={x:.2f}" for k, x in v.items()))
    print("best V by Indic byte share:", best)
    for k in ("english_mix_at_chosen", "pretok"):
        if k in out:
            print(k, out[k])


if __name__ == "__main__":
    main()
