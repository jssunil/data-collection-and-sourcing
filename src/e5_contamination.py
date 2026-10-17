"""E5: is any evaluation text inside the training pools? (13-gram overlap, the GPT-3 / Llama test.)

An evaluation item is contaminated when at least one of its 13-word sequences appears in a training document.
Items shorter than 13 words are tested with their full word sequence.

Output: results/e5_contamination.json
"""
from collections import defaultdict

from .common import EVAL, INDIC, TRAIN, read_jsonl, save_json, texts

N = 13


def norm_words(t):
    return [w.strip(".,!?;:\"'()[]{}।॥").lower() for w in t.split()]


def grams(ws, n=N):
    if len(ws) < n:
        return [tuple(ws)] if ws else []
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def main():
    evals = {f"flores_{c}": texts(EVAL / "lang" / f"{c}.jsonl") for c in ["eng"] + INDIC if (EVAL / "lang" / f"{c}.jsonl").exists()}
    for d in ["math_gsm8k", "code_humaneval", "science_sciq"]:
        evals[d] = texts(EVAL / "domain" / f"{d}.jsonl")
    index = defaultdict(set)  # n-gram hash -> {(set, item)}
    for name, items in evals.items():
        for i, t in enumerate(items):
            ws = norm_words(t)
            if len(ws) >= N:
                for g in grams(ws):
                    index[hash(g)].add((name, i))
    pools = [TRAIN / "eng_web.jsonl", TRAIN / "code.jsonl", TRAIN / "math.jsonl", TRAIN / "agentic.jsonl"]
    pools += sorted((TRAIN / "indic").glob("*.jsonl")) + sorted((TRAIN / "roman").glob("*.jsonl"))
    hits = defaultdict(set)
    examples = {}
    for p in pools:
        for doc in read_jsonl(p):
            ws = norm_words(doc["text"])
            for i in range(len(ws) - N + 1):
                h = hash(tuple(ws[i:i + N]))
                if h in index:
                    for item in index[h]:
                        hits[item[0]].add(item[1])
                        examples.setdefault(item[0], (p.name, " ".join(ws[i:i + N])))
    # short items (8 to 12 words): whole-sequence match. Shorter items match by chance, so they are not tested.
    short = {(n, i): tuple(norm_words(t)) for n, items in evals.items() for i, t in enumerate(items)
             if 8 <= len(norm_words(t)) < N}
    if short:
        by_len = defaultdict(dict)
        for key, g in short.items():
            by_len[len(g)][g] = key
        for p in pools:
            for doc in read_jsonl(p):
                ws = norm_words(doc["text"])
                for L, table in by_len.items():
                    for i in range(len(ws) - L + 1):
                        key = table.get(tuple(ws[i:i + L]))
                        if key:
                            hits[key[0]].add(key[1])
    res = {name: {"items": len(items), "contaminated": len(hits[name]), "rate": len(hits[name]) / len(items),
                  "example": examples.get(name)} for name, items in evals.items()}
    for name, r in res.items():
        print(f"{name:20s} {r['contaminated']:>5}/{r['items']:<5} {r['rate']:.2%}  {r['example'] or ''}", flush=True)
    save_json("e5_contamination", res)


if __name__ == "__main__":
    main()
