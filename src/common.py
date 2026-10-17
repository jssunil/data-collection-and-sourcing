"""Shared paths, language table, JSONL helpers and network setup."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN = DATA / "train"
EVAL = DATA / "eval"
TOK = DATA / "tokenizers"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"

QUICK = os.environ.get("QUICK") == "1"


def setup_net():
    """Use the Windows certificate store. Python's own bundle fails behind the local TLS proxy."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


# The 22 languages of the Eighth Schedule, plus English.
# speakers_m: first-language speakers in millions, Census of India 2011 (Hindi includes its grouped varieties).
# flores: FLORES-200 code (None = not in FLORES-200). sangraha: folder in ai4bharat/sangraha/verified.
# tier 1 = 30 M+ speakers; tier 2 = the others.
LANGS = [
    # code,  name,        script,       speakers_m, flores,     sangraha, tier
    ("hin", "Hindi",      "Devanagari", 528.3, "hin_Deva", "hin", 1),
    ("ben", "Bengali",    "Bengali",     97.2, "ben_Beng", "ben", 1),
    ("mar", "Marathi",    "Devanagari",  83.0, "mar_Deva", "mar", 1),
    ("tel", "Telugu",     "Telugu",      81.1, "tel_Telu", "tel", 1),
    ("tam", "Tamil",      "Tamil",       69.0, "tam_Taml", "tam", 1),
    ("guj", "Gujarati",   "Gujarati",    55.5, "guj_Gujr", "guj", 1),
    ("urd", "Urdu",       "Perso-Arabic",50.8, "urd_Arab", "urd", 1),
    ("kan", "Kannada",    "Kannada",     43.7, "kan_Knda", "kan", 1),
    ("ory", "Odia",       "Odia",        37.5, "ory_Orya", "ori", 1),
    ("mal", "Malayalam",  "Malayalam",   34.8, "mal_Mlym", "mal", 1),
    ("pan", "Punjabi",    "Gurmukhi",    33.1, "pan_Guru", "pan", 1),
    ("asm", "Assamese",   "Bengali",     15.3, "asm_Beng", "asm", 2),
    ("mai", "Maithili",   "Devanagari",  13.6, "mai_Deva", "mai", 2),
    ("sat", "Santali",    "Ol Chiki",     7.4, "sat_Olck", "sat", 2),
    ("kas", "Kashmiri",   "Perso-Arabic", 6.8, "kas_Arab", "kas", 2),
    ("npi", "Nepali",     "Devanagari",   2.9, "npi_Deva", "nep", 2),
    ("snd", "Sindhi",     "Perso-Arabic", 2.8, "snd_Arab", "snd", 2),
    ("doi", "Dogri",      "Devanagari",   2.6, None,       "doi", 2),
    ("gom", "Konkani",    "Devanagari",   2.3, None,       "gom", 2),
    ("mni", "Manipuri",   "Bengali",      1.8, "mni_Beng", "mni", 2),
    ("brx", "Bodo",       "Devanagari",   1.5, None,       "brx", 2),
    ("san", "Sanskrit",   "Devanagari",   0.02,"san_Deva", "san", 2),
]
LANG = {row[0]: dict(zip(("code", "name", "script", "speakers_m", "flores", "sangraha", "tier"), row)) for row in LANGS}
INDIC = [row[0] for row in LANGS]


SCRIPT_RANGES = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Ol Chiki": [(0x1C50, 0x1C7F)],
    "Perso-Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
}


def script_share(text, script):
    """Share of letters (Unicode L* and M*) that belong to the expected script."""
    import unicodedata
    ranges = SCRIPT_RANGES[script]
    n = hit = 0
    for ch in text:
        if unicodedata.category(ch)[0] in "LM":
            n += 1
            o = ord(ch)
            hit += any(a <= o <= b for a, b in ranges)
    return hit / n if n else 0.0


def write_jsonl(path, docs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def texts(path):
    return [d["text"] for d in read_jsonl(path)]


def save_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(name):
    with open(RESULTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
