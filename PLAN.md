# PLAN: Data for a 40B India-first model (Session 3)

This plan uses ASD-STE100 Simplified Technical English rules:
- Procedure sentences have 20 words or fewer. Description sentences have 25 words or fewer.
- Use the active voice. Procedures use the imperative.
- A word has one meaning in this document. Each term is defined before it is used.

The task text is in [assignment.md](assignment.md). The theory is in [SESSION_03_EXPLAINER.md](SESSION_03_EXPLAINER.md). The questions that build the reasoning from first principles are in [FIRST_PRINCIPLES_QA.md](FIRST_PRINCIPLES_QA.md).

---

## 1. Goal

Decide the data for a 40B model that is as good as Gemma 4 31B, strong in code and agentic work, strong in Indian languages, and India-first. Measure each number that a decision depends on. Do not accept a number that you did not measure or cite.

The assignment asks four questions:

| # | Question | Where the answer comes from |
|---|---|---|
| Q1 | What data do you collect for pre-training, post-training and RL, and why? | Literature + E4 (supply of Indic tokens) |
| Q2 | How do you clean it for these objectives? | E3 (English rules on Indic text) + E5 (contamination) |
| Q3 | How do you test the model against the objectives? | Literature + E1/E2 (tokenizer gates) |
| Q4 | Which fertility targets, which languages, and which vocabulary size? | E1 (released tokenizers) + E2 (vocabulary sweep and cost model) |

The submission is a short report on Netlify (`site/index.html`, built from `report/template.html`). The assignment says: "Longer submissions will result in lower scores." Thus the report gives decisions and numbers only. This repository holds the evidence.

---

## 2. Words

| Word | Meaning in this document |
|---|---|
| token | One unit that the model reads. One entry of the vocabulary. |
| vocabulary (V) | The set of all tokens. V is its size. |
| BPE | Byte-pair encoding. Start from the 256 byte values. Merge the most frequent adjacent pair. Repeat until the vocabulary has V entries. |
| pre-tokenizer | A regular expression that cuts text into pieces before BPE. BPE never merges across two pieces. |
| word | A run of characters between spaces. All 22 scheduled languages put spaces between words. |
| fertility | Tokens per word on a fixed text. Lower is better. |
| bytes per token | UTF-8 bytes per token. Higher is better. Used for code, math and agent traces, where "word" is not clear. |
| parity | Tokens for a sentence in language X ÷ tokens for the same sentence in English. 1.0 = same cost as English. |
| byte fallback | A character that the vocabulary does not contain is cut into its UTF-8 bytes. An Indic character is 3 bytes, so it costs 3 tokens. |
| tier 1 language | One of the 11 scheduled languages with 30 M or more first-language speakers (Census 2011). |
| tier 2 language | One of the other 11 scheduled languages. |
| FLORES | FLORES-200 devtest: the same 1,012 sentences in 200 languages. Bodo and Konkani come from IndicGenBench FLORES-IN (same English sentences). |
| Sangraha | The AI4Bharat Indic pretraining corpus (Khan et al., 2024). "Verified" = cleaned and checked by AI4Bharat. |
| mix | The share of each data source in a sample. |
| always-on channel | A fixed share of every training batch that is Indic, chosen before any quality filter (course notes). |
| contamination | Evaluation text inside the training data. |
| BPB | Bits per byte: the loss normalized by text bytes, not tokens. It compares models with different tokenizers. |
| golden proxy | A sealed, held-out evaluation set that no data engineer can read (course notes). |

---

## 3. Hardware, software and budget

| Item | Value |
|---|---|
| Machine | Intel Core Ultra 7 265K (20 threads), 63 GB RAM, RTX 5070 Ti (not needed: tokenizers run on CPU) |
| Software | Python 3.13, HF `tokenizers` 0.23, `pyarrow`, `huggingface_hub` |
| Network | Python HTTPS uses the Windows certificate store (`truststore`); see `src/common.py` |
| Download | About 8 GB of parquet reads; 0.6 GB kept in `data/` |

**Frugal rule.** The full study must run in about one hour on one desktop. A tokenizer is trained once at the largest size and cut to smaller sizes (section 6.3). This replaces ten training runs with one.

---

## 4. Data used by the experiments

| Pool | Source (Hugging Face) | Licence | Size kept | Used by |
|---|---|---|---|---|
| English web | `HuggingFaceFW/fineweb-edu` sample-10BT, files 000–002 | ODC-By | 180 M chars | E2, E3 |
| Code | `codeparrot/github-code-clean`, files 0–3 (30 languages) | per-file licences (permissive and GPL) | 80 M chars | E2 |
| Math | `HuggingFaceTB/finemath` finemath-3plus, file 0 | ODC-By | 25 M chars | E2 |
| Agent traces | `SWE-bench/SWE-smith-trajectories`, files 0–1 | MIT | 10 M chars | E2 |
| Indic, 22 languages | `ai4bharat/sangraha` verified, smallest file per language | CC-BY-4.0 | ≤ 25 M chars each | E2, E3, E4 |
| Romanized Indic, 10 languages | `ai4bharat/sangraha` synthetic `*_Latn` | CC-BY-4.0 | 2 M chars each | E2 |

Evaluation sets come from other files of the same datasets, or from benchmarks:

| Set | Source | Items |
|---|---|---|
| 21 languages + English | FLORES-200 devtest (`mteb/flores`); Bodo, Konkani from `google/IndicGenBench_flores_in` | 1,012 sentences each |
| Code | HumanEval (prompt + solution); github-code-clean file 500 | 164; 3 M chars |
| Math | GSM8K test; finemath file 100 | 1,319; 2 M chars |
| Science | SciQ test (support + question + answer) | 1,000 |
| Agentic | SWE-smith file 7; APIGen-MT-5k (tool schemas + multi-turn calls) | 3 M chars; 400 dialogues |
| Romanized | Hinglish (`findnitai/english-to-hinglish`); Sangraha `hin_Latn` file 1 | 3,000; 1 M chars |

**Dogri has no evaluation set.** FLORES-200 and IndicGenBench do not have Dogri. The only "Dogri" text found (Sangraha `verified/doi`) is English (section 7.3).

---

## 5. E1: fertility of released tokenizers

### 5.1 Purpose
Find what fertility is possible today. Find which tokenizer families fail on which scripts.

### 5.2 Design options

| Option | Advantage | Disadvantage | Selected |
|---|---|---|---|
| Tokens per word on parallel sentences (FLORES) | Same meaning in every language. Parity is exact. | Short, formal sentences. | **Yes** |
| Tokens per word on web text per language | Real pretraining text. | Different content per language; not comparable. | E4 only |
| Bytes per token | No word definition needed. | Indic UTF-8 is 3 bytes per character. A good Indic tokenizer has low bytes per token anyway. | For code, math, agentic |
| Characters per token | Script-neutral. | Hides the 3-byte cost. | No |
| Parity against English | One number that says "how much more does India pay". | Needs parallel text. | **Yes** |

### 5.3 Tokenizers
Gemma 4 (262K), Gemma 3 (262K), Sarvam-30B (262K), Sarvam-1 (68K), BrahmicTokenizer-131K (the V4 course tokenizer), Krutrim-2 (131K), Llama 3.1 (128K), Qwen3 (152K), DeepSeek-V3 (129K), Mistral-Nemo Tekken (131K), gpt-oss / o200k (200K). All files are public `tokenizer.json` files. Gemma 4 comes from `google/gemma-4-31B-it`.

### 5.4 Procedure
1. Load each `tokenizer.json` with HF `tokenizers`.
2. Encode each set without special tokens.
3. Calculate fertility, bytes per token and parity for each language.
4. Calculate the byte-fallback rate: the share of tokens that decode to a broken character.
5. Save `results/e1_existing.json` and the heat map.

---

## 6. E2: vocabulary sweep

### 6.1 Purpose
Find how fertility falls when V grows. Find the V that minimizes training cost per byte of the planned mix.

### 6.2 Design options

| Decision | Options | Selected and why |
|---|---|---|
| Algorithm | BPE; Unigram (SentencePiece); WordPiece | **Byte-level BPE.** No unknown token. Exact round trip. All 2025–26 frontier tokenizers use it. |
| Normalizer | NFC, NFKC, none | **None in the tokenizer.** NFC is a cleaning step. Then `decode(encode(x)) == x` for every input (a V5 rule). |
| Pre-tokenizer | GPT-2 pattern; o200k pattern | **o200k pattern with `\p{M}` in the letter class.** GPT-2's `\p{L}+` cuts a Hindi word at every vowel sign. E2c measures the cost. |
| Digits | 1-digit; 1–3 digits (o200k, Llama 3); none | **1 digit.** Same as Gemma. Each number has one spelling, which helps arithmetic. Costs more tokens on numbers. |
| Maximum token length | none; 32 bytes | **32 bytes.** Same as BrahmicTokenizer-131K. Keeps the byte-pooled (Kronecker) embedding option open. |
| Training mix | English-first; India-first | **India-first (36% Indic, temperature 0.3, floor 1%).** E2b measures the English-first alternative. |
| Script filter | none; ≥ 50% of letters in own script | **≥ 50%.** Drops mojibake and mislabelled English (section 7.3). |

### 6.3 One training run, ten sizes
BPE learns merges in order. The vocabulary of size k is the first k entries of a larger BPE vocabulary trained on the same text. `tests/test_bpe.py` proves this: the cut tokenizer gives the same ids as a tokenizer trained at size k.

Train once at V_max = 393,216 (384K). Cut to 32K, 48K, 64K, 96K, 128K, 160K, 192K, 256K, 320K, 384K.

### 6.4 Cost model for a 40B model
Assume d_model = 7,168, 60 layers, context 8,192. Non-embedding weights N = 12 · L · d² = 37.0 B.

Training FLOPs per token = 6N + 6dV (output layer) + 12 · L · d · context (attention scores).
Tokens per byte of the planned mix = Σ share_i ÷ (bytes per token)_i.
Cost per byte = (FLOPs per token) × (tokens per byte).

A larger V lowers tokens per byte. It raises the output-layer cost. The best V is where the two effects balance.

### 6.5 Procedure
1. Build the India-first mix (300 M characters).
2. Train BPE at 384K. Save it.
3. Cut to each size. Measure every FLORES language and every domain.
4. Calculate the cost model and the unused-token share.
5. Repeat 1–4 with the English-first mix (E2b).
6. Train two 64K tokenizers on a 75 M-character mix: GPT-2 pattern and ours (E2c).
7. Save `results/e2_vocab_sweep.json` and the plot.

---

## 7. E3: English cleaning rules on Indic text

### 7.1 Purpose
Show what happens when an English pipeline cleans Indic text. Show the fix.

### 7.2 Method
Sangraha verified text is already clean. Thus each rejection is a false positive. Apply six published English rules: Gopher word count, Gopher mean word length 3–10, Gopher stop words, letter ratio ≥ 0.75, C4 terminal punctuation, FineWeb line punctuation. Then apply the same rules with per-language settings:
- letters = Unicode `L*` and `M*` (vowel signs are letters);
- line ends include `।`, `॥`, `۔`;
- stop words = the 20 most frequent words of the language;
- word-length band = 0.5th to 99.5th percentile of the language.

Calibrate on one half of the documents. Measure on the other half. English (FineWeb-Edu) is the control.

### 7.3 Data-quality checks found during the work
- `sangraha/verified/kas` (Kashmiri) is legacy-font mojibake: Latin bytes such as `keÀes íes›` from a non-Unicode font.
- `sangraha/verified/doi` (Dogri): 9 of its 24 documents are not Devanagari; the first is English.
- Santali has 0.12 M characters; Kashmiri 0.45 M; Dogri none.

A script-purity filter (≥ 50% of letters in the expected script) catches the first two. "Verified" does not mean "checked for your purpose".

---

## 8. E4: supply of Indic tokens

1. Read only the parquet footers of Sangraha (verified, unverified, synthetic). Sum the uncompressed text bytes per language.
2. Convert bytes to tokens with each tokenizer's bytes per token on that language.
3. Compare with the planned Indic share of a 20 T-token run. Report the epochs needed.

---

## 9. E5: contamination check

1. Take every 13-word sequence of each evaluation item (GPT-3 and Llama 3 use 13-grams).
2. Scan every training pool for these sequences.
3. Mark an item contaminated if one sequence matches. Test items of 8–12 words as whole sequences.
4. Report the rate per set.

---

## 10. Budget

| Step | Time (about) |
|---|---|
| Downloads | 15 min |
| E1 | 1 min |
| E2a training (384K, 300 M chars) + sweep | 25–40 min |
| E2b, E2c | 25 min |
| E3 | 2 min |
| E4 footers | 15 min (network) |
| E5 | 10 min |

## 11. Tests

`tests/test_bpe.py` (CPU, no download): truncation equals direct training; exact round trip on Indic, emoji, CRLF and zero-width characters; single digits; vowel signs stay inside a pre-token; ≤ 32 bytes per token; metric arithmetic; Indic shares sum to 1.
