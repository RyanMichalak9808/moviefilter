"""Profanity detection over a word-timestamped transcript.

Matches (in order of confidence):
  * list   - word regex from the word list (per level / language)
  * phrase - consecutive words ("son of a bitch", "oh my god")
  * masked - words Whisper censored itself ("f***", "s**t")
  * fuzzy  - low-confidence words that are spelled almost like a strong word ("fuk")
"""
import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEVELS = ("strong", "slurs", "mild", "religious")
DEFAULT_LEVELS = ("strong", "slurs")
DATA = Path(__file__).with_name("data") / "wordlist.json"

MASKED = re.compile(r"^[^\W\d_]{1,2}[*#@$%]+[^\W\d_]*$")
LOW_CONF = 0.5          # below this a match is flagged for review
FUZZY_MAX_PROB = 0.75   # only consider fuzzy matches for words Whisper was unsure about
FUZZY_RATIO = 0.84


def norm(token):
    t = unicodedata.normalize("NFKC", token).casefold().replace("’", "'")
    t = re.sub(r"[^\w'*#@$%-]", "", t)
    return t.strip("'-")


@dataclass
class Detection:
    start: float
    end: float
    text: str
    level: str
    kind: str
    prob: float
    i0: int = 0
    i1: int = 0
    flags: list = field(default_factory=list)
    context: str = ""

    def to_dict(self):
        return asdict(self)


def load_wordlist(path=None):
    data = json.loads(Path(path or DATA).read_text(encoding="utf-8"))
    data.pop("_comment", None)
    return data


def _alt(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None


class Matcher:
    def __init__(self, languages=("en",), levels=DEFAULT_LEVELS, wordlist=None,
                 extra_words=(), allow=()):
        data = load_wordlist(wordlist)
        unknown = set(levels) - set(LEVELS) - {"custom"}
        if unknown:
            raise ValueError(f"unknown level(s): {', '.join(sorted(unknown))}; choose from {', '.join(LEVELS)}")
        self.levels = [lv for lv in LEVELS if lv in levels]
        self.word_rx = {}       # level -> compiled alternation
        self.phrases = []       # (level, [compiled per token])
        self.fuzzy = []         # (level, base word)
        allow_pats, name_pats = list(allow), []
        langs = [lang for lang in dict.fromkeys(languages) if lang in data]
        for lv in self.levels:
            words, phrases = [], []
            for lang in langs:
                spec = data[lang]["levels"].get(lv, {})
                words += spec.get("words", [])
                phrases += spec.get("phrases", [])
            self.word_rx[lv] = _alt(words)
            self.phrases += [(lv, [re.compile(p) for p in ph]) for ph in phrases]
        for lang in langs:
            allow_pats += data[lang].get("allow", [])
            name_pats += data[lang].get("possible_names", [])
            for lv, bases in data[lang].get("fuzzy", {}).items():
                if lv in self.levels:
                    self.fuzzy += [(lv, b) for b in bases]
        if extra_words:
            self.levels.append("custom")
            self.word_rx["custom"] = _alt([re.escape(norm(w)) for w in extra_words])
        self.allow = _alt(allow_pats)
        self.names = _alt(name_pats)
        self.languages = langs

    # ------------------------------------------------------------------ matching
    def _word_level(self, tok):
        if not tok or (self.allow and self.allow.fullmatch(tok)):
            return None
        for lv in self.levels:
            rx = self.word_rx.get(lv)
            if rx and rx.fullmatch(tok):
                return lv
        return None

    def find(self, words):
        """words: list of {start, end, word, prob?}. Returns list[Detection], time-sorted."""
        toks = [norm(w["word"]) for w in words]
        probs = [float(w.get("prob", 1.0)) for w in words]
        dets = []

        def add(i0, i1, level, kind, flags=()):
            seg = words[i0:i1]
            d = Detection(start=seg[0]["start"], end=seg[-1]["end"],
                          text=" ".join(w["word"].strip() for w in seg), level=level, kind=kind,
                          prob=round(min(probs[i0:i1]), 3), i0=i0, i1=i1, flags=list(flags))
            if d.prob < LOW_CONF:
                d.flags.append("low-confidence")
            lo, hi = max(0, i0 - 6), min(len(words), i1 + 6)
            d.context = " ".join(
                ("[" + w["word"].strip() + "]") if i0 <= j < i1 else w["word"].strip()
                for j, w in enumerate(words[lo:hi], start=lo))
            dets.append(d)

        for i, tok in enumerate(toks):
            lv = self._word_level(tok)
            if lv:
                flags = []
                if self.names and self.names.fullmatch(tok) and _looks_like_name(words, i):
                    flags.append("possible-name")
                add(i, i + 1, lv, "list", flags)
            elif tok and MASKED.fullmatch(tok):
                add(i, i + 1, "strong", "masked")
            elif (self.fuzzy and probs[i] < FUZZY_MAX_PROB and len(tok) >= 3 and tok.isalpha()
                  and not (self.allow and self.allow.fullmatch(tok))):
                for lv, base in self.fuzzy:
                    if tok[0] == base[0] and difflib.SequenceMatcher(None, tok, base).ratio() >= FUZZY_RATIO:
                        add(i, i + 1, lv, "fuzzy", ["fuzzy"])
                        break

        for lv, pats in self.phrases:
            n = len(pats)
            for i in range(len(toks) - n + 1):
                if all(p.fullmatch(toks[i + j]) for j, p in enumerate(pats)):
                    add(i, i + n, lv, "phrase")

        dets.sort(key=lambda d: (d.start, d.end))
        return dets


def _looks_like_name(words, i):
    raw = words[i]["word"].strip()
    if not raw[:1].isupper() or i == 0:
        return False
    prev = words[i - 1]["word"].strip()
    return not prev.endswith((".", "!", "?", ":", '"'))
