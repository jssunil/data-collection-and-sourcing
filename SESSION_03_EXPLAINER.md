# Session 3 Explainer: Data Collection and Sourcing

This document uses ASD-STE100 Simplified Technical English rules: short sentences, active voice, one meaning for each word. The terms are defined in the "Words" table of [PLAN.md](PLAN.md#2-words).

Sources: the course notes (`era-v5-capstone-tracker/sessions/session_03_data_collection_and_sourcing.md`), the BrahmicTokenizer-131K report (the V4 course tokenizer), the 85 top peer submissions (section 10), and the papers in section 12. The measurements are from this repository (`results/`). Each arXiv number in section 12 was checked on arxiv.org on 17 Oct 2026.

---

## 1. How to study Session 3 with this assignment

| Session idea | Assignment question | What you must be able to say after it |
|---|---|---|
| The mixture is the model | Q1 data | "I can say what each source teaches the model, and how many unique tokens of it exist." |
| Overtraining | Q1 data | "I can calculate the token budget of a 40B model and the epochs each source needs." |
| Tiered pools, iterative mining | Q1, Q2 | "I know how a seed set grows into a domain corpus without drift." |
| Erasure risk, always-on channel | Q2 cleaning | "I can show that English filters delete Indic text, and how to fix them." |
| Golden proxy, contamination | Q3 testing | "I know how evaluation text leaks, and how to stop it." |
| Tokenizer (from Session 2) | Q4 fertility, vocab | "I can choose V from a measured curve and a cost model, not from habit." |

**Study procedure (about 5 hours):**
1. Read the course notes once.
2. Read sections 2–4 of this document. Do Q1 of [FIRST_PRINCIPLES_QA.md](FIRST_PRINCIPLES_QA.md) on paper.
3. Read FineWeb §3 and FineWeb-2 §3 (how per-language thresholds are set). Run E3.
4. Read the BrahmicTokenizer-131K report §2.3 and §5. Run E1.
5. Read "Scaling laws with vocabulary" (Tao et al.). Run E2.
6. Read DeepSeekMath §2 (the mining loop) and Llama 3 §3.1 (data) and §5 (contamination).
7. Read the report (`site/index.html`, or the Netlify link in README) and check each number against `results/`.
8. Read section 10, then two peer submissions. For each, find one number that is cited but not measured, and one practice this repository lacks.

---

## 2. Why the data decides the model

A transformer of a given size and a given compute budget is close to standard. Two labs with the same architecture get different models because the data differs. The data decides three things:

1. **What the model can know.** A fact that is not in the data is not in the model.
2. **What the model does well.** The share of each domain sets the share of capacity it gets.
3. **How the model sees the world.** The sources decide whose view is the default. For an India-first model, "default" must mean an Indian reader: lakh and crore, ₹, IST, Indian law, Indian maps, Indian festivals, Indian names.

### 2.1 Token budget of a 40B model

Chinchilla (Hoffmann et al., 2022) gives about 20 tokens per weight for the best loss at fixed training compute: 0.8 T tokens for 40B. Modern models train far longer, because inference cost depends on the model size, not on the training tokens.

| Model | Weights | Training tokens | Tokens per weight |
|---|---|---|---|
| Chinchilla-optimal 40B | 40 B | 0.8 T | 20 |
| Llama 3 8B | 8 B | 15 T | 1,875 |
| Gemma 3 27B | 27 B | 14 T | 520 |
| Sarvam-30B (MoE, 2.4B active) | 30 B | 16 T | 530 |
| **Plan: 40B dense** | 40 B | **20 T** | 500 |

Training compute: 6 · N · D = 6 × 40e9 × 20e12 = 4.8e24 FLOPs. At 40% MFU on H100 (about 400 TFLOP/s), this is about 3.3 M GPU-hours.

### 2.2 Unique tokens limit the mix

Muennighoff et al. (2023) show that up to about 4 epochs of the same data cost little. After about 16 epochs, more repeats add nothing. Thus the share of a source in a 20 T run is limited by its unique tokens:

> share × 20 T ≤ 4 × unique tokens

English web has more than 15 T unique high-quality tokens (FineWeb, DCLM, Nemotron-CC). Code has several trillion (The Stack v2). **Indic text does not.** E4 measures it: see the report. E4: 67–111 B unique native Indic tokens; 4 epochs of that is at most 2.2% of 20 T. This single fact shapes the Indic plan: synthetic translation, multi-epoch native data, and a tokenizer that does not waste the scarce Indic tokens.

**Do not plan against a headline count.** Sangraha's dataset card gives 251 B tokens, but only 64.3 B are verified; 24.3 B are unverified and 162.7 B (65%) are synthetic (translated Wikimedia and romanized copies). Telugu falls from 16.3 B to 3.7 B verified, Odia from 12.5 B to 1.2 B. E4 reads the parquet footers and agrees: 47 B verified + 20 B unverified tokens under our tokenizer.

---

## 3. The four tiers and the mining loop

The course divides the pretraining corpus into four tiers:

| Tier | Content | Role |
|---|---|---|
| 1 | Curated seed: books, papers, Wikipedia, documentation, licensed code | Core capability; trains the quality classifiers |
| 2 | Synthetic: rephrased web, reasoning chains, verified code traces, translations | Fills gaps the web does not cover |
| 3 | Web, top quality-classifier scores | Breadth with quality |
| 4 | Web, heuristic filters and dedup only | Coverage of rare facts and styles |

**The DeepSeekMath loop** (Shao et al., 2024) turns a small seed into a large domain corpus:
1. Train a fastText classifier: seed (positive) against random web (negative).
2. Score Common Crawl. Keep pages above a threshold.
3. Find domains (URL hosts) where many pages were kept. Annotate their other pages.
4. Add the verified pages to the seed. Retrain. Repeat.

DeepSeekMath used OpenWebMath as the seed (500,000 positive examples in round 1) and reached 35.5 M pages, 120 B tokens, after four rounds. The course notes say "1 GB seed"; the paper's seed is OpenWebMath. For an India-first model the same loop finds Indian legal text, Indian government text, Indian educational text, and Indic text in each domain.

**Why the loop does not drift:** the seed stays fixed and high-precision. Each round adds only verified pages. The negative class is re-sampled each round.

---

## 4. The erasure risk

Quality filters learn "good text" from a reference set. If the reference is English, the filter learns "good = English-like".

E3 measures this directly. Six published English rules were applied to Sangraha verified text, which AI4Bharat had already cleaned. Each rejection is a false positive.

| Rule | Why it fails on Indic text |
|---|---|
| Gopher stop words (≥ 2 of "the, be, to, of, and, that, have, with") | No English words in Hindi text. |
| Letter ratio ≥ 0.75 (Python `isalpha`) | Python does not count vowel signs (`ि`, `ा`), virama (`्`) or anusvara (`ं`) as letters. They are 25–40% of Indic characters. |
| C4 terminal punctuation (`. ! ? "`) | Hindi, Bengali, Odia, Punjabi end sentences with `।`. Urdu uses `۔`. |
| Gopher mean word length 3–10 | Malayalam and Tamil join words (agglutination). Mean word length is 6–11 characters. |

Result (E3): English rules reject 96–100% of clean text in every scheduled language except Manipuri (75%). English control: 11%. The same rules with per-language settings reject 2–21%. FineWeb-2 (Penedo et al., 2025) uses the same idea for 1,000+ languages: it sets each threshold from the language's own reference distribution.

**The always-on channel** is the second defence. A fixed share of every batch is Indic, decided before any filter runs. A filter can then lower quality inside the channel, but it cannot delete the channel.

### 4.1 Unicode problems specific to Indic text
- **Normalization.** The same visible text has several code-point sequences (for example, a nukta letter as one code point or two). E3 found 43% of Marathi, 59% of Manipuri and 27% of Konkani documents not in NFC. Normalize to NFC in cleaning, before dedup and before tokenizer training. Do not put NFC inside the tokenizer: then `decode(encode(x)) ≠ x`.
- **ZWJ and ZWNJ** (U+200D, U+200C) are part of correct spelling in Malayalam (chillu forms), Hindi and Bengali. A rule that deletes "invisible characters" corrupts words. E3 found them in 0.3–3.5% of documents.
- **Legacy fonts.** Much Indian government and news text was typed in non-Unicode fonts (Krutidev, Shree-Lipi). Extracted text is Latin mojibake. Sangraha `verified/kas` is an example (`keÀes íes›`). Detect it by script purity; convert with font maps where they exist.
- **Mislabelled language.** Sangraha `verified/doi` (Dogri) is English. Language ID must be checked per document, including for "verified" sources.

---

## 5. Tokenization from first principles

### 5.1 What a tokenizer costs
Every downstream cost is per token: training FLOPs, context length, inference latency and price. If Hindi needs 2× the tokens of English for the same meaning, then a Hindi user gets half the context and pays 2× per answer, and the model sees half the Hindi meaning per training FLOP. Petrov et al. (2023) call this the tokenization premium. Ahia et al. (2023) measure the price difference across commercial APIs.

### 5.2 The three regimes for Brahmic scripts
The BrahmicTokenizer-131K report gives three regimes:

| Regime | What the vocabulary has | Fertility on a Brahmic word |
|---|---|---|
| Byte fallback | No merge for the 3 UTF-8 bytes of a character | 3+ tokens per character; 10–18 per word |
| Subword | Character tokens and some subword pieces | 2–3 |
| Whole word | Single tokens for frequent words | 1–2 |

E1 shows all three: Llama 3.1 and Qwen3 are in byte fallback on most Brahmic scripts (tier-1 mean 9–10 tokens per word); Gemma 4 and gpt-oss are in the subword regime; Sarvam-30B and our 256K are close to the whole-word regime for tier 1.

### 5.3 The pre-tokenizer decides what BPE can learn
BPE never merges across two pre-tokens. GPT-2's pattern uses `\p{L}+` for a word. In Unicode, Indic vowel signs and the virama are marks (`\p{M}`), not letters. Thus the GPT-2 pattern cuts `विद्यालय` at every vowel sign, and no Hindi word can become one token. The o200k pattern includes `\p{M}` in its letter classes. `tests/test_bpe.py::test_vowel_signs_stay_in_word` proves the difference; E2c measures its cost: at 64K, Hindi 3.33 vs 1.58 tokens per word, Tamil 6.83 vs 2.61.

### 5.4 Why the vocabulary size has an optimum
A larger V gives fewer tokens per byte, so less compute per byte of text. It also gives:
- a larger output layer (6 · d · V FLOPs per token, and V · d weights; 2 · V · d if untied);
- more tokens that are rarely seen, so their embeddings are under-trained (Land and Bartolo, 2024, find such "glitch" tokens in many released models);
- a larger softmax at inference.

Tao et al. (2024) fit scaling laws with vocabulary. They find that the best V grows with model size, and that most models are trained with a V that is too small. Their estimate for Llama 2 70B is at least 216K. A 40B model trained on 20 T tokens is in the same range. E2 measures our own curve and finds the minimum of a simple cost model.

### 5.5 Fertility is a property of the training mix, not only of V
At the same V, a tokenizer trained on English-first data gives Indic text far fewer slots. BrahmicTokenizer-131K shows this from the other side: it retrofits 2,372 Brahmic slots into o200k and lowers Indic tokens by 27% against Tekken at the same V. E2b trains the same V on an English-first mix: tier-1 fertility rises from 1.77 to 2.02.

### 5.6 Digits
Gemma splits numbers into single digits. o200k and Llama 3 use groups of 1–3 digits from the left. Singh and Strouse (2024) show that left-to-right 3-digit groups (as in o200k) hurt arithmetic, and that forcing right-to-left groups fixes most of the loss. Single digits avoid the problem: each number has one spelling. They are the simple, robust choice. They cost more tokens on number-heavy text (E2 math bytes per token).

---

## 6. Evaluation

### 6.1 Compare models with bits per byte, not loss per token
Loss per token depends on the tokenizer. A tokenizer with longer tokens has a higher loss per token for the same model quality. Bits per byte (BPB) divides total loss by text bytes. Use BPB per language and per domain on the golden proxy for all data ablations.

### 6.2 Contamination
Test text leaks in through web copies, GitHub mirrors, and translated benchmark sets. The standard test is n-gram overlap (13-grams in GPT-3 and Llama 3). E5 shows its weakness: generic phrases ("0 1 2 3 … a b c", "people who are at high risk of developing serious complications") match by chance. Use a frequency-aware rule: ignore n-grams that are frequent in the training data. Also test translated copies: an Indic benchmark often exists in English first.

### 6.3 The golden proxy
A sealed held-out set, owned by the evaluation team only. It contains the same domains as the mix: English, each Indic language, code, math, agent traces, India-specific facts. Data engineers see only the BPB numbers, never the text.

### 6.4 Rules for a fair comparison with Gemma 4
- **Use the numbers Gemma 4 reports, and know which are missing.** The Gemma 4 report (Table 5, thinking mode) gives for 31B: MMLU-Pro 85.2, GPQA-Diamond 84.3, AIME 2026 89.2, LiveCodeBench v6 80.0, Codeforces 2150, MMMLU 88.4, τ²-bench airline 75.0 / retail 86.4 / telecom 69.3, Terminal-Bench Hard 36.0. It gives **no SWE-bench Verified and no Indic score**. For those, run Gemma 4 and our model in the same harness.
- **Compare thinking mode with thinking mode.** Report both modes separately.
- **Report the worst language, not only the mean.** A good Hindi score can hide a broken Malayalam score. Report multi-script languages (Santali, Manipuri, Sindhi, Kashmiri, Konkani) per script.
- **Give uncertainty.** Three seeds for stochastic agent runs; 95% bootstrap confidence intervals.
- **Drop saturated sets as targets.** HumanEval, MBPP and GSM8K are near 95% for frontier models and are in many crawls. Keep them only as smoke tests.
- **Prefer native-authored Indic tests.** MILU is mostly from real Indian exams. Translated sets (much of IndicGenBench, IndicXTREME) measure translation more than Indian knowledge.
- **Put canary strings in private test sets.** If a checkpoint completes a canary, the test text leaked into training.

---

## 7. Post-training and RL data

| Stage | Data | Why |
|---|---|---|
| SFT | Instruction following (multilingual), code (verified solutions), agent trajectories (SWE, terminal, tool calls), Indic instructions written by native speakers, India-perspective answers | Teach the format and the default view |
| Preference | Pairs rated by Indian annotators, per language | Teach "better" as an Indian user judges it |
| RL with verifiers | Math (answer check), code (unit tests), SWE (tests in Docker), tool use (simulated environments), instruction constraints (IFEval-style, in Indic languages) | Rewards that cannot be gamed by style |

Tulu 3 (Lambert et al., 2024) gives an open recipe for SFT → DPO → RL with verifiable rewards. DeepSeek-R1 (2025) shows RL with verifiable rewards alone can teach long reasoning. SWE-Gym and SWE-smith give executable environments for agentic RL.

**Test the verifier before it gives reward.** In RL the model optimizes the reward directly, so a wrong verifier teaches the wrong behaviour efficiently. The errors are measured:
- Wang et al. (2025) re-checked SWE-bench patches that pass the tests. 29.6% of these "plausible" patches behave differently from the reference fix, 28.6% of the divergent ones are certainly wrong, and the reported solve rates are inflated by 6.2 points.
- Huang et al. (2025) find that rule-based math checkers reject correct answers in other formats (false negatives), and the loss grows as the policy improves; model-based checkers accept wrong patterns (reward hacking).

Thus: add hidden tests and a second checker, and have humans audit a sample of rewarded rollouts per environment and per language before RL starts.

**Hold the anneal data back from day 0.** The best data (verified native Indic, tested code, competition math, agent traces) has the most effect at the end of training, when the learning rate decays. OLMo 2 (2025) raised 7B GSM8K from 24.1 to 67.5 with its mid-training mix. If this data is spent in the general pool, the lever is gone.

---

## 8. India-first is a data property

A model "views the world from an Indian perspective" when the Indian answer is its default, without a prompt that asks for it. Three data levers do this:
1. **India-centric English** (Indian news, NCERT and state textbooks, court judgments, Gazette, PIB, Sansad debates). Most Indian knowledge is written in English.
2. **Native Indic text** in all 22 scheduled languages, with an always-on share.
3. **Post-training defaults**: SFT and preference data where the Indian frame is the expected answer (₹, lakh/crore, IST, DD-MM-YYYY, Indian maps and law), written and rated by Indian annotators.

Test it with paired questions that have a different default answer in India and elsewhere (section 6 of the report).

---

**Fluency is not perspective.** Agarwal et al. (2025, "Fluent but Foreign") tested six Indic and six global models on Indian values and practices. The Indic models were not closer to Indian norms than the global ones; "a U.S. respondent is a closer proxy for Indian values than any Indic model". Regional fine-tuning did not fix it. Thus Indic fluency data and India-perspective data are two separate pools with two separate tests.

**India-first is not one view.** Keep plural sources on contested history and politics, and attribute each view. Store a date and a jurisdiction (Union or state) with each legal or policy fact: the Bharatiya Nyaya Sanhita replaced the IPC on 1 July 2024, and a model trained on undated text mixes both.

---

## 9. Corrections to the course notes

- **"Chinchilla: 20 tokens per parameter."** This is the compute-optimal point for training loss at fixed training compute. It is not a rule for a deployed model.
- **"An English-trained fastText classifier assigns high perplexity."** fastText is a classifier, not a language model; it gives a probability, not a perplexity. The erasure risk is the same for both: a KenLM perplexity filter (CCNet) and a fastText quality classifier both learn English-like = good.
- **"8% always-on channel."** The number must come from supply and targets. E4 shows that native Indic supply limits the share; 8–10% needs synthetic translation and multi-epoch repeats.

---

## 10. Best practices from the 85 top submissions

The course published 85 top submissions for this assignment. We read the 78 that render as text (17 Oct 2026). The count column is the number of those 78 that mention the practice (text search), so it shows what is common, not what is right. The last column says what this repository does with it.

| Practice | Count | In this repository |
|---|---|---|
| Hinglish / romanized / code-mixed text as a real register | 62 | Already: 1% of the mix, romanized eval sets, tokenizer trained on `*_Latn` |
| Anneal (cooldown) phase with the best data | 46 | Added: the reserve is held back from day 0 (OLMo 2: 7B GSM8K 24.1 → 67.5) |
| OCR of Indian print; ASR of Indian audio | 40; 30 | ASR added as a supply source (radio, Sansad, court audio). OCR already there |
| Proxy-model ablations before the full run | 40 | Already in the plan: 1B proxies, RegMix, bits per byte (not run here) |
| Indian court judgments as India-centric text | 39 | Added the open source: 25 High Courts, CC-BY-4.0, AWS bucket `indian-high-court-judgments` |
| Provenance / licence ledger at collection time | 38 | Added to cleaning step 8 |
| Single-digit numbers | 33 | Already (E2 math cost) |
| Plural viewpoints, no partisan default | 33 | Added to section 8 |
| Native-authored SFT, not translated | 31 | Already |
| Tokenizer fertility from published tables (MUTANT, Sarvam) | 26 | We measure instead (E1, E2); MUTANT's plateau beyond 200K agrees with E2's flat cost curve from 192K |
| Reserved control tokens | 24 | 11 today; reserve a block of 1,024 ids so tool and thinking tokens never shift ids |
| Verifier audit / reward hacking | 22 | Added (section 7): hidden tests, a second checker, human audit, a 5% stop rule |
| Translationese detection | 21 | Added: native-vs-translated classifier; fluency tested on native text only |
| Canary strings in private test sets | 20 | Added (6.4) |
| Per-snapshot dedup (FineWeb) | 17 | Kept per-language MinHash; per-snapshot is an ablation for the English pool |
| Dated legal and policy facts (IPC → BNS) | 15 | Added (section 8 and India-centric pool) |
| Confidence intervals, seeds | 15 | Added (6.4) |
| Release / stop rules | 14 | Added to the report (risks) |
| Bits per byte for cross-language comparison | 6 | Already central (6.1) |
| Own measurements (trained a tokenizer, read the data) | very few | E1–E6 |

**Where we disagree with the common choice.**
- **Synthetic share of the Indic channel.** Many reports cap it at 25–40%. That cap needs 3–5× more native supply than E4 measures, or more than 4 epochs. Our channel is 75% translated by arithmetic, with the controls above (FIRST_PRINCIPLES_QA R3).
- **Fertility targets copied from another tokenizer's table.** A target that a measured tokenizer already beats is not a target. Our targets are set against E2's measured 262K tokenizer and Gemma 4 on the same text.
- **"Gemma 4 τ²-bench 76.9".** Several reports quote this average; the Gemma 4 report's Table 5 gives only the three domains (75.0 / 86.4 / 69.3).
- **Morphology-aware pre-tokenization for Dravidian languages.** A pre-tokenizer boundary only removes merges that BPE could learn, so at a fixed V it does not, in general, lower fertility. E2c shows the opposite lever: keeping vowel signs inside words (`\p{M}`) halves Indic fertility.

**E6: are all vocabulary rows trained?** One peer report proposed the gate "≥ 99.5% of tokens occur ≥ 1,000 times in the final mix". E6 tests it on our tokenizer's own 300 M-character training sample (73 M tokens): 775 learned tokens never occur at 262K (0.30%; 131K: 173; 384K: 2,055), mostly Latin pieces absorbed by later merges. 2.7% occur fewer than 10 times. Any token that occurs once in the sample occurs about 270,000 times in a 20 T run, so the gate passes for all but the orphans. The fix: retrain on ≥ 10 GB and drop or re-map zero-count tokens before the freeze.

---

## 11. Results in one table

| Experiment | Finding |
|---|---|
| E1 | Gemma 4 = Gemma 3 tokenizer. Sarvam-30B is the best released tokenizer on Indic (tier-1 fertility 2.03, parity 1.71). Llama 3.1, Qwen3 are in byte fallback (parity 5.6–5.8). |
| E2 | V = 262,144. Cost per byte is flat (±0.7%) from 192K to 384K. At 262K: tier-1 fertility 1.77, English 1.25, code 3.84 bytes per token. English-first mix at the same V: tier-1 2.02. GPT-2 pre-tokenizer at 64K: Hindi 3.33 vs 1.58. |
| E3 | English rules reject 75–100% of clean Indic text; per-language rules 2–21%. |
| E4 | 67–111 B unique native Indic tokens (Sangraha + FineWeb-2, overlapping). At 4 epochs: at most 2.2% of a 20 T run. |
| E5 | 0% FLORES-Indic, 0.2% FLORES-English, 0.6% HumanEval, 0.4% SciQ; the matches are generic phrases. |
| E6 | At 262K, 775 learned tokens (0.30%) never occur in the tokenizer's own training sample; 384K: 2,055 (0.52%). Mostly Latin pieces. |

---

## 12. References

1. Hoffmann et al., Training Compute-Optimal Large Language Models (Chinchilla), arXiv:2203.15556
2. Muennighoff et al., Scaling Data-Constrained Language Models, arXiv:2305.16264
3. Shao et al., DeepSeekMath, arXiv:2402.03300
4. Llama Team, The Llama 3 Herd of Models, arXiv:2407.21783
5. Rae et al., Scaling Language Models: Methods, Analysis & Insights from Training Gopher, arXiv:2112.11446
6. Raffel et al., Exploring the Limits of Transfer Learning (T5, C4), arXiv:1910.10683
7. Penedo et al., The FineWeb Datasets, arXiv:2406.17557
8. Penedo et al., FineWeb2: One Pipeline to Scale Them All, arXiv:2506.20920
9. Li et al., DataComp-LM (DCLM), arXiv:2406.11794
10. Su et al., Nemotron-CC, arXiv:2412.02595
11. Lee et al., Deduplicating Training Data Makes Language Models Better, arXiv:2107.06499
12. Petrov et al., Language Model Tokenizers Introduce Unfairness Between Languages, arXiv:2305.15425
13. Ahia et al., Do All Languages Cost the Same?, arXiv:2305.13707
14. Rust et al., How Good is Your Tokenizer?, arXiv:2012.15613
15. Tao et al., Scaling Laws with Vocabulary, arXiv:2407.13623
16. Land and Bartolo, Fishing for Magikarp: Automatically Detecting Under-trained Tokens, arXiv:2405.05417
17. Singh and Strouse, Tokenization Counts: the Impact of Tokenization on Arithmetic, arXiv:2402.14903
18. Liu et al., SuperBPE, arXiv:2503.13423
19. Shravan, BrahmicTokenizer-131K, arXiv:2605.29379
20. Rana et al., MUTANT: A Recipe for Multilingual Tokenizer Design, arXiv:2511.03237
21. Khan et al., IndicLLMSuite (Sangraha), arXiv:2403.06350
22. Gala et al., IndicTrans2 (IN22), arXiv:2305.16307
23. NLLB Team, No Language Left Behind (FLORES-200), arXiv:2207.04672
24. Verma et al., MILU, arXiv:2411.02538
25. BhashaBench V1, arXiv:2510.25409
26. Singh et al., IndicGenBench, arXiv:2404.16816
27. Singh et al., Global-MMLU, arXiv:2412.03304
28. Gemma Team, Gemma 3 Technical Report, arXiv:2503.19786
29. Gemma Team, Gemma 4 Technical Report, arXiv:2607.02770
30. Lambert et al., Tulu 3, arXiv:2411.15124
31. DeepSeek-AI, DeepSeek-R1, arXiv:2501.12948
32. Jimenez et al., SWE-bench, arXiv:2310.06770
33. Pan et al., SWE-Gym, arXiv:2412.21139
34. Yang et al., SWE-smith, arXiv:2504.21798
35. Yao et al., τ-bench, arXiv:2406.12045
36. Jain et al., LiveCodeBench, arXiv:2403.07974
37. Kargaran et al., GlotLID, arXiv:2310.16248
38. Liu et al., RegMix: Data Mixture as Regression, arXiv:2407.01492
39. Xie et al., DoReMi, arXiv:2305.10429
40. Kudugunta et al., MADLAD-400, arXiv:2309.04662
41. Team OLMo, 2 OLMo 2 Furious, arXiv:2501.00656
42. Agarwal et al., Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment, arXiv:2505.21548
43. Wang et al., Are "Solved Issues" in SWE-bench Really Solved Correctly?, arXiv:2503.15223
44. Huang et al., From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning, arXiv:2505.22203

Data sources checked on 17 Oct 2026: Sangraha dataset card (huggingface.co/datasets/ai4bharat/sangraha: verified 64,306 M, unverified 24,308 M, synthetic 162,708 M tokens); AWS Open Data registry, Indian High Court Judgments (25 High Courts, CC-BY-4.0, `s3://indian-high-court-judgments`, no account needed).
