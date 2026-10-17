"""Build report/index.html from report/template.html and the results JSON files.

The charts and the per-language table read the injected DATA object, so their numbers come
straight from results/. Run after all experiments:  python report/build.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import INDIC, LANG, load_json  # noqa: E402


def main():
    s = load_json("summary")
    e2 = load_json("e2_vocab_sweep")
    e3 = load_json("e3_cleaning")
    data = {
        "sweep": {mix: [{"V": r["V"], "t1": r["mean_fertility_tier1"], "t2": r["mean_fertility_tier2"],
                         "eng": r["lang"]["eng"]["fertility"], "code": r["domain"]["code_github"]["bytes_per_token"]}
                        for r in e2[mix]["rows"]] for mix in ("india", "english") if mix in e2},
        "cost": [{"V": c["V"], "rel": c["rel_cost"]} for c in e2["india"]["cost"]],
        "ref": {n: s["tier_means"][n][0] for n in ("gemma-4", "sarvam-30b", "brahmic-131k", "gpt-oss")},
        "cleaning": [{"lang": c, "en": e3[c]["english"]["any"], "aware": e3[c]["aware"]["any"]} for c in e3],
        "langs": s["per_language"],
        "domains": s["domains"],
    }
    html = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    (ROOT / "report" / "index.html").write_text(html, encoding="utf-8")
    print("wrote report/index.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
