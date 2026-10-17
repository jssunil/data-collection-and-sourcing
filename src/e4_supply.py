"""E4: how many unique Indic tokens exist, per language, and how many epochs a 40B run would need.

Reads only the parquet footers of ai4bharat/sangraha (verified, unverified, synthetic) to get the
uncompressed text bytes per language. Converts bytes to tokens with the measured bytes-per-token of
each tokenizer on that language's own training pool.

    python -m src.e4_supply bytes     # footers only (network)
    python -m src.e4_supply           # tokens and epochs (needs E1 and E2 results)

Output: results/e4_supply.json
"""
import sys
from collections import defaultdict

from .common import INDIC, LANG, RESULTS, TRAIN, load_json, save_json, setup_net, texts

# Planned run (report section 2).
RUN_TOKENS = 20e12
INDIC_SHARE = 0.10          # native-script Indic share of all training tokens
MAX_EPOCHS = 4              # Muennighoff et al. (2023): up to about 4 epochs cost little


# FineWeb-2 words per language (dataset card, read 17 Oct 2026). Other scheduled languages are not listed.
FINEWEB2_WORDS = {"hin": 11_173_681_651, "ben": 6_153_579_265, "tam": 1_937_150_898, "npi": 1_642_856_349,
                  "mar": 1_541_225_070, "mal": 1_054_187_581, "urd": 2_733_266_493, "tel": 891_002_487,
                  "kan": 748_850_327, "guj": 934_124_052, "pan": 522_788_467, "ory": 333_760_951}


def footer_bytes():
    setup_net()
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    root = "datasets/ai4bharat/sangraha"
    out = defaultdict(dict)
    for subset in ("verified", "unverified", "synthetic"):
        for d in fs.ls(f"{root}/{subset}", detail=False):
            folder = d.split("/")[-1]
            total = 0
            for f in fs.ls(d, detail=False):
                if not f.endswith(".parquet"):
                    continue
                md = pq.ParquetFile(fs.open(f)).metadata
                names = [md.schema.column(i).name for i in range(md.num_columns)]
                j = names.index("text")
                total += sum(md.row_group(r).column(j).total_uncompressed_size for r in range(md.num_row_groups))
            out[subset][folder] = total
            print(subset, folder, f"{total / 1e9:.2f} GB", flush=True)
    save_json("e4_supply_bytes", out)


def tokens(tok_name, tok):
    from .metrics import stats
    bytes_ = load_json("e4_supply_bytes")
    rows = {}
    for c in INDIC:
        s = LANG[c]["sangraha"]
        sample = texts(TRAIN / "indic" / f"{c}.jsonl")[:3000]
        st = stats(tok, sample)
        bpt, tpw = st["bytes_per_token"], st["fertility"]
        v = bytes_["verified"].get(s, 0)
        u = bytes_["unverified"].get(s, 0)
        rows[c] = {"bytes_per_token": bpt, "tokens_per_word": tpw, "verified_B": v / bpt / 1e9,
                   "unverified_B": u / bpt / 1e9, "fineweb2_B": FINEWEB2_WORDS.get(c, 0) * tpw / 1e9}
    tot_v = sum(r["verified_B"] for r in rows.values())
    tot_u = sum(r["unverified_B"] for r in rows.values())
    tot_f = sum(r["fineweb2_B"] for r in rows.values())
    need = RUN_TOKENS * INDIC_SHARE / 1e9
    # Sangraha and FineWeb-2 both come from Common Crawl and overlap; the larger of the two is the
    # lower bound of unique supply, the sum is the upper bound.
    lo, hi = max(tot_v + tot_u, tot_f), tot_v + tot_u + tot_f
    return {"per_lang": rows, "verified_B": tot_v, "unverified_B": tot_u, "fineweb2_B": tot_f,
            "unique_lo_B": lo, "unique_hi_B": hi, "need_B": need,
            "epochs_needed_lo": need / hi, "epochs_needed_hi": need / lo,
            "max_native_share_at_4_epochs": MAX_EPOCHS * hi * 1e9 / RUN_TOKENS}


def main():
    if sys.argv[1:] == ["bytes"]:
        footer_bytes()
        return
    from .metrics import load
    from . import bpe
    from .common import TOK
    res = {"run_tokens": RUN_TOKENS, "indic_share": INDIC_SHARE, "max_epochs": MAX_EPOCHS}
    cands = {"gemma-4": load("gemma-4"), "sarvam-30b": load("sarvam-30b")}
    final = load_json("e2_vocab_sweep").get("chosen_V")
    if final:
        big = bpe.load_file(sorted(TOK.glob("ours_india_*k.json"))[0])
        cands[f"ours-{final // 1024}k"] = bpe.truncate(big, final)
    for n, t in cands.items():
        res[n] = tokens(n, t)
        r = res[n]
        print(f"{n}: Sangraha verified {r['verified_B']:.0f} B + unverified {r['unverified_B']:.0f} B, FineWeb-2 {r['fineweb2_B']:.0f} B tokens; "
              f"unique {r['unique_lo_B']:.0f}-{r['unique_hi_B']:.0f} B; {INDIC_SHARE:.0%} of the run needs {r['need_B']:.0f} B "
              f"= {r['epochs_needed_lo']:.0f}-{r['epochs_needed_hi']:.0f} epochs; max native share at 4 epochs {r['max_native_share_at_4_epochs']:.1%}", flush=True)
    save_json("e4_supply", res)


if __name__ == "__main__":
    main()
