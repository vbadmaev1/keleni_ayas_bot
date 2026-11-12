"""
Транскрипция INEL (латиница) -> современная калмыцкая кириллица.

Буква-в-букву не получается: орфография не пишет редуцированную ə
(xurəldə -> хурлд, eckə -> эцк), j + гласная то сливается (ju -> ю),
то нет (jovad -> йовад), начальное e -> э и т. д. Поэтому каждое
неоднозначное место раскрывается в несколько вариантов, а победителя
выбирает частотный словарь калмыцких слов (из todorxoi-bot:
model/letters_vocab.tsv.gz, ~108 тыс. словоформ). Если ни одного варианта
в словаре нет — берётся вариант по умолчанию (первый в каждом списке).
"""

import gzip
import itertools
import re
from pathlib import Path

VOCAB_PATH = Path.home() / "Desktop/todorxoi-bot/model/letters_vocab.tsv.gz"

VOWELS = set("aeiouäöüə")
CYR_RE = re.compile(r"[а-яёәөүһҗң]", re.I)


def load_vocab(path=VOCAB_PATH):
    vocab = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


# многобуквенные сочетания (проверяются первыми); значение — варианты
MULTI = [
    ("zh", ["ж"]),
    ("šč", ["щ"]),
    ("ʼa", ["ья", "я"]),
    ("ʼu", ["ью", "ю"]),
    ("ʼe", ["ье", "е"]),
    ("ʼo", ["ьо", "ё"]),
    ("ya", ["я"]),          # русские заимствования: buudya, zayavlenʼ
    ("yu", ["ю"]),
]

SIMPLE = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "z": "з", "i": "и",
    "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р",
    "s": "с", "t": "т", "u": "у", "f": "ф", "x": "х", "c": "ц", "č": "ч",
    "š": "ш", "ž": "ж", "ä": "ә", "ö": "ө", "ü": "ү", "ʁ": "һ", "ǯ": "җ",
    "ŋ": "ң", "h": "һ", "q": "к", "y": "ы", "ʼ": "ь", "ă": "а", "w": "в",
}


# второй проход: орфография пишет звонкую там, где слышится глухая
# (küüktän -> күүкдән), и русские заимствования (Sivir -> Сибирь)
ALT = {"т": ["т", "д"], "к": ["к", "г"], "в": ["в", "б"], "с": ["с", "з"],
       "п": ["п", "б"], "х": ["х", "к"], "ш": ["ш", "ж"]}


def segments(word, extended=False):
    """Слово (строчными) -> список списков вариантов."""
    out = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        prev = word[i - 1] if i else ""
        nxt = word[i + 1] if i + 1 < n else ""
        two = word[i:i + 2]
        hit = next((m for m in MULTI if two == m[0]), None)
        if hit:
            out.append(hit[1]); i += 2; continue
        if ch == "j":
            if nxt == "a":
                # ja / jaa -> я / яа ; после согласной редко
                out.append(["я", "йа"]); i += 2; continue
            if nxt == "u":
                out.append(["ю", "йу"]); i += 2; continue
            if nxt == "o":
                out.append(["йо", "ё"]); i += 2; continue
            if nxt == "e":
                out.append(["е", "йе", "э"]); i += 2; continue
            if nxt == "ä":
                out.append(["я", "йә", "йа"]); i += 2; continue
            out.append(["й"]); i += 1; continue
        if ch == "e":
            if prev == "e":
                out.append(["е", "э"])
            elif i == 0 or prev in VOWELS or prev == "-":
                out.append(["э", "е"])
            else:
                out.append(["е", "э"])
            i += 1; continue
        if ch == "ə":
            # первый слог: гласную пишут; дальше обычно нет
            first = not any(c in VOWELS for c in word[:i])
            if first:
                out.append(["ы", "ә", "а", "и", ""])
            elif i == n - 1:
                # конечную ə орфография не пишет почти никогда (xärə -> хәр);
                # None — маркер «отступление стоит вдвое дороже»
                out.append(["", None, "ә", "а", "и", "ү", "у"])
            else:
                out.append(["", "ә", "а", "ы", "и", "ү", "у"])
            i += 1; continue
        if ch in SIMPLE:
            c = SIMPLE[ch]
            if extended and c in ALT and 0 < i < n - 1:
                out.append(ALT[c])
            else:
                out.append([c])
            i += 1; continue
        out.append([ch]); i += 1
    return out


def candidates(word, limit=400, extended=False):
    segs = segments(word, extended)
    combos = 1
    for s in segs:
        combos *= len([v for v in s if v is not None])
    if combos > limit:
        # слишком много ə: оставляем только вариант «не писать» для лишних
        segs = [[v for v in s if v is not None] for s in segs]
        segs = [s[:2] if len(s) > 2 else s for s in segs]
        combos = 1
        for s in segs:
            combos *= len(s)
        if combos > limit * 8:
            segs = [s[:1] for s in segs]
    idx = [[k for k, v in enumerate(x) if v is not None] for x in segs]
    for choice in itertools.product(*idx):
        # dev — сколько раз ушли от варианта по умолчанию
        dev = sum(1 + (None in segs[k]) for k, c in enumerate(choice) if c)
        yield "".join(segs[k][c] for k, c in enumerate(choice)), dev


PENALTY = 0.3   # каждое отступление от правила по умолчанию


class Converter:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else load_vocab()
        self.cache = {}

    def word(self, w):
        """Одно слово без пунктуации. Возвращает (кириллица, найдено_в_словаре)."""
        if not w or CYR_RE.search(w) or not re.search(r"[a-zäöüəʁǯčšžŋ]", w, re.I):
            return w, True
        if "-" in w:
            parts = [self.word(p) for p in w.split("-")]
            return "-".join(p[0] for p in parts), all(p[1] for p in parts)
        key = w
        if key not in self.cache:
            low = w.lower().replace("ǝ", "ə")
            low = low.replace("_", " ")
            if " " in low:
                res = [self.word(p) for p in low.split()]
                self.cache[key] = (" ".join(r[0] for r in res), all(r[1] for r in res))
            else:
                best, best_f, default = None, 0, None
                for extended in (False, True):
                    for cand, dev in candidates(low, extended=extended):
                        if default is None:
                            default = cand
                        f = self.vocab.get(cand, 0) * PENALTY ** dev
                        if f > best_f:
                            best, best_f = cand, f
                    if best:
                        break
                self.cache[key] = (best or default, best is not None)
        out, found = self.cache[key]
        m = re.match(r"^[A-ZÄÖÜČŠŽ]{2,}", w)
        if m:
            k = len(m.group(0))
            out = out[:k].upper() + out[k:]
        elif w[:1].isupper():
            out = " ".join(p[:1].upper() + p[1:] for p in out.split(" "))
        return out, found

    TOKEN = re.compile(r"([^\s]+)")

    def sentence(self, text):
        """Предложение целиком; пунктуация и русские вставки остаются."""
        text = clean_markup(text)
        unknown = []

        def repl(m):
            tok = m.group(1)
            mm = re.match(r"^([\"“(«]*)(.*?)([\"”)».,!?:;…]*)$", tok)
            pre, core, post = mm.groups()
            cyr, found = self.word(core)
            if not found:
                unknown.append(core)
            return pre + cyr + post

        return self.TOKEN.sub(repl, text), unknown


def clean_markup(text):
    text = text.replace("ǝ", "ə")
    text = re.sub(r"\(\(+[^)]*\)+\)", " ", text)       # ((PAUSE)), ((…)), ((%%))
    text = re.sub(r"\(([^()]*?)=\)", r"\1…", text)     # (mal=) — оборванное слово
    text = text.replace("=", "")
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    conv = Converter()
    for line in sys.stdin:
        print(conv.sentence(line.strip())[0])
