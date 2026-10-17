"""Download the samples for all experiments into data/ (about 8 GB of parquet reads, about 0.6 GB kept).

Training pools (for tokenizer training and the cleaning study) and evaluation sets come from
different files of each dataset, so no document is in both.

    python scripts/get_data.py            # all
    python scripts/get_data.py eval       # evaluation sets only
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import EVAL, INDIC, LANG, QUICK, TOK, TRAIN, setup_net, write_jsonl  # noqa: E402

setup_net()
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem, hf_hub_download  # noqa: E402

FS = HfFileSystem()
SCALE = 0.05 if QUICK else 1.0
M = 1_000_000


def ds(repo, path):
    return f"datasets/{repo}/{path}"


def sample_parquet(paths, budget_chars, text_fn, columns=None, seed=0, max_doc_chars=100_000):
    """Read row batches from remote parquet files until the character budget is used.

    Documents are taken in batch order, with a random skip so that one source does not dominate.
    Long documents are cut at max_doc_chars to keep the sample diverse.
    """
    rng = random.Random(seed)
    out, used = [], 0
    for p in paths:
        f = pq.ParquetFile(FS.open(p))
        for batch in f.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                if rng.random() < 0.5:
                    continue
                t = text_fn(row)
                if not t or not t.strip():
                    continue
                t = t[:max_doc_chars]
                out.append(t)
                used += len(t)
                if used >= budget_chars:
                    return out
    return out


def save(path, docs, **meta):
    write_jsonl(path, [{"text": t, **meta} for t in docs])
    print(f"{path.relative_to(path.parents[2])}: {len(docs)} docs, {sum(map(len, docs)) / M:.2f} M chars", flush=True)


def swe_smith(row):
    msgs = json.loads(row["messages"])
    return "\n".join(f"<|{m['role']}|>\n{m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])}" for m in msgs)


# ---------------------------------------------------------------- training pools
# Budgets in characters. Pools are larger than any single mix needs, so that mixes are subsamples.
SANGRAHA_FILES = {  # smallest verified file per language (sizes listed in PLAN.md section 4)
    "hin": "data-31", "ben": "data-67", "mar": "data-25", "tel": "data-1", "tam": "data-41", "guj": "data-8",
    "urd": "data-16", "kan": "data-1", "ori": "data-4", "mal": "data-1", "pan": "data-9", "asm": "data-2",
    "mai": "data-0", "sat": "data-0", "kas": "data-0", "nep": "data-11", "snd": "data-1", "doi": "data-0",
    "gom": "data-0", "mni": "data-0", "brx": "data-0", "san": "data-8",
}
ROMAN = ["hin", "ben", "mar", "tam", "tel", "guj", "kan", "mal", "pan", "urd"]
ROMAN_FILE = {"hin": "0043", "ben": "0045", "mar": "0038", "tam": "0029", "tel": "0028", "guj": "0035",
              "kan": "0033", "mal": "0039", "pan": "0031", "urd": "0020"}


def get_train():
    save(TRAIN / "eng_web.jsonl",
         sample_parquet([ds("HuggingFaceFW/fineweb-edu", f"sample/10BT/00{i}_00000.parquet") for i in range(3)],
                        180 * M * SCALE, lambda r: r["text"], ["text"]), source="fineweb-edu")
    save(TRAIN / "code.jsonl",
         sample_parquet([ds("codeparrot/github-code-clean", f"data/train-0000{i}-of-00880.parquet") for i in range(4)],
                        80 * M * SCALE, lambda r: r["code"], ["code", "language"], max_doc_chars=20_000), source="github-code-clean")
    save(TRAIN / "math.jsonl",
         sample_parquet([ds("HuggingFaceTB/finemath", "finemath-3plus/train-00000-of-00128.parquet")],
                        25 * M * SCALE, lambda r: r["text"], ["text"]), source="finemath-3plus")
    save(TRAIN / "agentic.jsonl",
         sample_parquet([ds("SWE-bench/SWE-smith-trajectories", f"data/ticks-0000{i}-of-00008.parquet") for i in range(2)],
                        10 * M * SCALE, swe_smith, ["messages"], max_doc_chars=30_000), source="swe-smith")
    for code in INDIC:
        s = LANG[code]["sangraha"]
        docs = sample_parquet([ds("ai4bharat/sangraha", f"verified/{s}/{SANGRAHA_FILES[s]}.parquet")],
                              25 * M * SCALE, lambda r: r["text"], ["text"], max_doc_chars=20_000)
        if code == "doi":  # no FLORES Dogri: hold out the last 20% of the only file for evaluation
            k = int(len(docs) * 0.8)
            save(EVAL / "lang" / "doi.jsonl", docs[k:], lang="doi", parallel=False)
            docs = docs[:k]
        save(TRAIN / "indic" / f"{code}.jsonl", docs, lang=code, source="sangraha-verified")
    for code in ROMAN:
        save(TRAIN / "roman" / f"{code}.jsonl",
             sample_parquet([ds("ai4bharat/sangraha", f"synthetic/{code}_Latn/wiki_{code}_Latn_{ROMAN_FILE[code]}_of_0063.parquet")],
                            2 * M * SCALE, lambda r: r["text"], ["text"], max_doc_chars=20_000), lang=code, source="sangraha-synthetic-latn")


# ---------------------------------------------------------------- evaluation sets
def get_eval():
    import pandas as pd

    flores = pd.read_parquet(hf_hub_download("mteb/flores", "devtest.parquet", repo_type="dataset"))
    write_jsonl(EVAL / "lang" / "eng.jsonl", [{"text": t, "lang": "eng", "parallel": True} for t in flores["eng_Latn"]])
    for code in INDIC:
        fl = LANG[code]["flores"]
        if fl:
            rows = [{"text": t, "lang": code, "parallel": True} for t in flores[fl]]
        elif code in ("brx", "gom"):  # IndicGenBench FLORES-IN: same English devtest sentences, new targets
            ex = json.load(open(hf_hub_download("google/IndicGenBench_flores_in", f"flores_en_{code}_test.json",
                                                repo_type="dataset"), encoding="utf-8"))["examples"]
            rows = [{"text": e["target"], "lang": code, "parallel": True, "eng": e["source"]} for e in ex]
        else:
            continue  # doi is written by get_train
        write_jsonl(EVAL / "lang" / f"{code}.jsonl", rows)
    # IndicGenBench gives the English source per row; keep it for exact parity.
    print("FLORES devtest:", len(flores), "sentences per language")

    he = pd.read_parquet(hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset"))
    save(EVAL / "domain" / "code_humaneval.jsonl", [p + s for p, s in zip(he["prompt"], he["canonical_solution"])], domain="code")
    save(EVAL / "domain" / "code_github.jsonl",
         sample_parquet([ds("codeparrot/github-code-clean", "data/train-00500-of-00880.parquet")], 3 * M,
                        lambda r: r["code"], ["code"], seed=1, max_doc_chars=20_000), domain="code")
    gsm = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    save(EVAL / "domain" / "math_gsm8k.jsonl", [q + "\n" + a for q, a in zip(gsm["question"], gsm["answer"])], domain="math")
    save(EVAL / "domain" / "math_web.jsonl",
         sample_parquet([ds("HuggingFaceTB/finemath", "finemath-3plus/train-00100-of-00128.parquet")], 2 * M,
                        lambda r: r["text"], ["text"], seed=1), domain="math")
    sciq = pd.read_parquet(hf_hub_download("allenai/sciq", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    save(EVAL / "domain" / "science_sciq.jsonl",
         [f"{s}\n{q}\n{a}" for s, q, a in zip(sciq["support"], sciq["question"], sciq["correct_answer"])], domain="science")
    save(EVAL / "domain" / "eng_web.jsonl",
         sample_parquet([ds("HuggingFaceFW/fineweb-edu", "sample/10BT/010_00000.parquet")], 2 * M,
                        lambda r: r["text"], ["text"], seed=1), domain="english")
    save(EVAL / "domain" / "agentic_swe.jsonl",
         sample_parquet([ds("SWE-bench/SWE-smith-trajectories", "data/ticks-00007-of-00008.parquet")], 3 * M,
                        swe_smith, ["messages"], seed=1, max_doc_chars=30_000), domain="agentic")
    api = json.load(open(hf_hub_download("Salesforce/APIGen-MT-5k", "apigen-mt_5k.json", repo_type="dataset"), encoding="utf-8"))
    tool_docs = [json.dumps(x["tools"], ensure_ascii=False) + "\n" +
                 "\n".join(f"<|{m['from']}|>\n{m['value']}" for m in x["conversations"]) for x in api[:400]]
    save(EVAL / "domain" / "agentic_tools.jsonl", tool_docs, domain="agentic")
    lines = open(hf_hub_download("findnitai/english-to-hinglish", "hinglish_upload_v1.json", repo_type="dataset"),
                 encoding="utf-8").read().splitlines()
    rng = random.Random(0)
    pairs = [json.loads(x)["translation"] for x in rng.sample(lines, 3000)]
    save(EVAL / "domain" / "hinglish.jsonl", [p["hi_ng"] for p in pairs], domain="romanized")
    save(EVAL / "domain" / "roman_hin.jsonl",
         sample_parquet([ds("ai4bharat/sangraha", "synthetic/hin_Latn/wiki_hin_Latn_0001_of_0063.parquet")], 1 * M,
                        lambda r: r["text"], ["text"], seed=1), domain="romanized")


# ---------------------------------------------------------------- existing tokenizers
EXISTING = {
    "gemma-4": "google/gemma-4-31B-it",
    "gemma-3": "unsloth/gemma-3-1b-it",
    "sarvam-30b": "sarvamai/sarvam-30b",
    "sarvam-1": "sarvamai/sarvam-1",
    "brahmic-131k": "theschoolofai/BrahmicTokenizer-131K",
    "krutrim-2": "krutrim-ai-labs/Krutrim-2-instruct",
    "llama-3.1": "unsloth/Llama-3.1-8B",
    "qwen3": "Qwen/Qwen3-8B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Base-2407",
    "gpt-oss (o200k)": "openai/gpt-oss-20b",
}


def get_tokenizers():
    import shutil
    TOK.mkdir(parents=True, exist_ok=True)
    for name, repo in EXISTING.items():
        p = hf_hub_download(repo, "tokenizer.json")
        shutil.copy(p, TOK / f"{name.split(' ')[0]}.json")
        print("tokenizer", name, "<-", repo, flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "tokenizers"):
        get_tokenizers()
    if what in ("all", "eval"):
        get_eval()
    if what in ("all", "train"):
        get_train()
