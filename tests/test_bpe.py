"""CPU tests for the tokenizer code and the metrics. No download needed."""
import random

from src import bpe
from src.metrics import stats, words

HI = "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं और विद्यालय में बच्चे पढ़ते हैं।"
TA = "தமிழ் ஒரு பழமையான மொழி. இந்தியாவில் பல மொழிகள் பேசப்படுகின்றன."
EN = "The quick brown fox jumps over the lazy dog. Prices rose 12345 percent in 2024!"
CODE = "def add(a, b):\n    return a + b  # sum\n\nfor i in range(10):\n    print(add(i, 2))\n"


def corpus(n=400, seed=0):
    rng = random.Random(seed)
    base = [HI, TA, EN, CODE]
    out = []
    for _ in range(n):
        out.append(" ".join(" ".join(rng.choice(base).split(" ")[: rng.randint(3, 12)]) for _ in range(3)))
    return out


def test_truncate_equals_direct_training():
    docs = corpus()
    big = bpe.train(docs, 420)
    small = bpe.train(docs, 340)
    cut = bpe.truncate(big, 340)
    assert cut.get_vocab_size() == small.get_vocab_size() == 340
    for x in [HI, TA, EN, CODE]:
        assert cut.encode(x).ids == small.encode(x).ids


def test_round_trip_exact():
    tok = bpe.train(corpus(), 700)
    for x in [HI, TA, EN, CODE, "  leading spaces\t\ttabs\r\nCRLF", "emoji 🙂 and zero-width​space", "ଓଡ଼ିଆ ভাষা"]:
        assert tok.decode(tok.encode(x, add_special_tokens=False).ids) == x


def test_digits_are_single_tokens():
    tok = bpe.train(corpus(), 700)
    pieces = tok.encode("12345", add_special_tokens=False).tokens
    assert pieces == ["1", "2", "3", "4", "5"]


def test_vowel_signs_stay_in_word():
    """With PAT_OURS a Hindi word is one pre-token; with the GPT-2 pattern it splits at each vowel sign."""
    from tokenizers import Regex, pre_tokenizers
    w = "विद्यालय"
    ours = pre_tokenizers.Split(Regex(bpe.PAT_OURS), behavior="isolated").pre_tokenize_str(w)
    gpt2 = pre_tokenizers.Split(Regex(bpe.PAT_GPT2), behavior="isolated").pre_tokenize_str(w)
    assert len(ours) == 1
    assert len(gpt2) > 1


def test_max_token_bytes():
    tok = bpe.train(["a" * 200] * 50 + corpus(), 800)
    longest = max(len(tok.decode([i]).encode("utf-8")) for i in range(tok.get_vocab_size())
                  if tok.id_to_token(i) not in bpe.SPECIALS)
    assert longest <= 32


def test_stats():
    tok = bpe.train(corpus(), 700)
    s = stats(tok, [EN])
    assert s["words"] == words(EN) == 15
    assert abs(s["fertility"] - s["tokens"] / 15) < 1e-12
    assert s["bytes"] == len(EN.encode("utf-8"))


def test_indic_shares_sum_to_one_and_floor():
    p = bpe.indic_shares()
    assert abs(sum(p.values()) - 1) < 1e-9
    assert min(p.values()) > 0.009
