# Data for a 40B India-First Model (Session 3)

ERA V5 Session 3 assignment: decide the data, cleaning, evaluation, fertility targets and vocabulary size for a 40B model that is as good as Gemma 4, strong in code and agentic work, strong in Indian languages, and India-first. The task text is in [assignment.md](assignment.md).

**Submission:** the short report in [`report/index.html`](report/index.html), deployed on Netlify (link at the end of this file). This repository holds the evidence: code, data manifests, results and logs.

## Documents

| File | Content |
|---|---|
| [report/index.html](report/index.html) | The submission: decisions and numbers only. |
| [PLAN.md](PLAN.md) | Experiment plan: words, design options with reasons, procedures, budget, tests. |
| [SESSION_03_EXPLAINER.md](SESSION_03_EXPLAINER.md) | Session 3 in depth: token budget, tiers, mining loop, erasure risk, Unicode issues, tokenizer theory, evaluation; 40 papers (arXiv numbers checked). |
| [FIRST_PRINCIPLES_QA.md](FIRST_PRINCIPLES_QA.md) | 46 questions that rebuild each decision from first principles, with answers from the measurements. |

## How to run

```
pip install -r requirements.txt truststore
python scripts/get_data.py          # tokenizers, eval sets, training pools (~8 GB read, 0.6 GB kept)
python -m src.e1_existing           # E1: 11 released tokenizers           (1 min)
python -m src.e2_vocab_sweep        # E2: BPE sweep 32K-384K, 2 mixes, pre-tokenizer ablation (~30 min)
python -m src.summary               # decisions table, chosen V
python -m src.e3_cleaning           # E3: English rules on clean Indic text (2 min)
python -m src.e4_supply bytes       # E4: Sangraha parquet footers (network)
python -m src.e4_supply             # E4: unique Indic tokens, epochs
python -m src.e5_contamination      # E5: 13-gram overlap (10 min)
pytest                              # CPU only, no download
```

`QUICK=1` makes every step small (smoke test). Python HTTPS uses the Windows certificate store through `truststore` (`src/common.py`), because a local TLS proxy breaks Python's own certificate bundle.

Each step writes `results/<step>.json`, a plot where useful, and `results/logs/<step>.log`.

## Results summary

Run on 17 Oct 2026, one desktop CPU (Core Ultra 7 265K, 63 GB RAM).

| Exp. | Question | Answer |
|---|---|---|
| E1 | How good are released tokenizers on the 22 scheduled languages? | Best: Sarvam-30B (tier-1 fertility 2.03). Gemma 4 = Gemma 3 tokenizer (2.59). Llama 3.1 and Qwen3 are in byte fallback (9–10). BrahmicTokenizer-131K drops the Arabic script: Urdu, Kashmiri, Sindhi 7.4–8.9. |
| E2 | Which vocabulary size? | **262,144.** Cost per byte falls 14.5% from 32K to 256K and is flat (±0.7%) to 384K. At 262K: tier-1 fertility 1.77, tier-2 1.99, English 1.25, code 3.84 bytes per token. |
| E2b | Does the tokenizer's training mix matter? | Yes. Same V, English-first mix: tier-1 fertility 2.02 (+14% Indic tokens), English 1.21 (−3%). |
| E3 | Do English cleaning rules work on Indic text? | No. They reject 75–100% of already-clean text (English control 11%). Per-language versions reject 2–21%. 43% of Marathi and 59% of Manipuri documents are not NFC. |
| E3 | Is "verified" data clean? | Not always. Sangraha `verified/kas` is legacy-font mojibake (93 of 154 docs fail a script check). `verified/doi` has 24 docs, 9 not in Devanagari (English). |
| E4 | How many unique native Indic tokens exist? | 67–111 B under our tokenizer (Sangraha + FineWeb-2, overlapping). At 4 epochs: at most 2.2% of a 20 T run, so the 8% always-on channel is 2% native + 6% translated. |
| E2c | Does the pre-tokenizer matter? | Yes. GPT-2 pattern at 64K: Hindi 3.33 vs 1.58 tokens per word, Tamil 6.83 vs 2.61. |
| E5 | Is evaluation text in the training pools? | FLORES-Indic 0%, FLORES-English 0.2%, HumanEval 0.6%, SciQ 0.4%; the matches are generic phrases. |

## Limits

- **Tokenizer training sample: 300 M characters.** Production tokenizers train on 10–100× more. The ranking of sizes is robust; the exact fertility at 262K can move by a few percent.
- **FLORES is formal news and travel text.** Web, chat and code-mixed text have different fertility. The Hinglish and romanized sets are small.
- **Dogri has no evaluation set.** No public parallel Dogri text was found; the Sangraha folder labelled Dogri is mostly English.
- **The cost model** counts FLOPs only. It ignores memory bandwidth at inference and the quality effect of vocabulary size; Tao et al. (2024) model the latter.
- **E4** uses the Sangraha parquet footers and the FineWeb-2 dataset card. Both come from Common Crawl and overlap; the report gives a range.

## Deploying the report

```
python report/build.py                              # injects results/*.json into report/index.html
npx netlify-cli deploy --dir report --prod          # first time: npx netlify-cli login
```

Netlify link: https://era-v5-s3-data-india-40b.netlify.app
