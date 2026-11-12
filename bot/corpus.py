"""Корпус: загрузка data/corpus.json, поиск, вспомогательные функции."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"

# Свёртка для поиска: прощаем отсутствие калмыцких букв (я/ю/h вместо ә/ү/һ
# и т. п.), ё/е, регистр. Применяется и к запросу, и к тексту.
_FOLD = str.maketrans({
    "ә": "а", "ө": "о", "ү": "у", "һ": "г", "җ": "ж", "ң": "н", "ё": "е",
    "h": "г",
    "ä": "a", "ö": "o", "ü": "u", "ʁ": "g", "ǯ": "j", "č": "c", "š": "s",
    "ž": "z", "ŋ": "n", "ə": "", "ʼ": "", "’": "", "'": "",
})
_DEGRADED = [("я", "а"), ("ю", "у")]  # «бяядл» -> «баадл»: только для запроса
_WORD = re.compile(r"[\w-]+", re.U)


def fold(text: str) -> str:
    return text.lower().translate(_FOLD)


def _fold_query(q: str) -> List[str]:
    """Варианты запроса: как есть и с я->а / ю->у (если писали без ә/ү)."""
    base = fold(q)
    out = [base]
    alt = base
    for a, b in _DEGRADED:
        alt = alt.replace(a, b)
    if alt != base:
        out.append(alt)
    return out


# GOST 7.79-B (так в метаданных записаны имена) -> кириллица
_GOST = [("shh", "щ"), ("zh", "ж"), ("ch", "ч"), ("sh", "ш"), ("yu", "ю"),
         ("ya", "я"), ("yo", "ё"), ("e`", "э"), ("y`", "ы"), ("``", "ъ"),
         ("`", "ь"), ("x", "х"), ("c", "ц"), ("j", "й"), ("a", "а"), ("b", "б"),
         ("v", "в"), ("g", "г"), ("d", "д"), ("e", "е"), ("z", "з"), ("i", "и"),
         ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
         ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("f", "ф"), ("h", "х"),
         ("y", "ы")]


def gost_to_cyr(name: str) -> str:
    out, i, low = [], 0, name.lower()
    while i < len(name):
        for lat, cyr in _GOST:
            if low.startswith(lat, i):
                piece = cyr.upper() if name[i].isupper() else cyr
                out.append(piece)
                i += len(lat)
                break
        else:
            out.append(name[i])
            i += 1
    return "".join(out)


def person(name: str) -> str:
    """'Sangadzhieva, Kermen' -> 'Кермен Сангаджиева'."""
    name = re.sub(r"\s+", " ", name).strip()
    if "," in name:
        fam, given = [p.strip() for p in name.split(",", 1)]
        name = f"{given} {fam}"
    return gost_to_cyr(name)


@dataclass
class Sentence:
    text_id: str
    n: int
    start: float
    end: float
    lat: str
    cyr: str
    ru: str
    en: str
    folded: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Text:
    id: str
    genre: str
    genre_ru: str
    title_ru: str
    title_en: str
    dialect: str
    year: str
    place: str
    speakers: List[dict]
    audio: bool
    aligned: bool
    duration: float
    partial: bool
    sentences: List[Sentence]

    @property
    def full_audio(self) -> Optional[Path]:
        p = AUDIO / "full" / f"{self.id}.ogg"
        return p if self.audio and p.exists() else None

    def clip(self, n: int) -> Optional[Path]:
        p = AUDIO / "sent" / self.id / f"{n:03d}.ogg"
        return p if self.aligned and p.exists() else None

    @property
    def speaker_names(self) -> str:
        return ", ".join(person(s["name"]) for s in self.speakers if s.get("name"))


class Corpus:
    def __init__(self, path: Path = ROOT / "data" / "corpus.json"):
        raw = json.load(open(path, encoding="utf-8"))
        self.texts: Dict[str, Text] = {}
        self.order: List[str] = []
        for t in raw:
            sents = [Sentence(text_id=t["id"], **{k: s[k] for k in
                              ("n", "start", "end", "lat", "cyr", "ru", "en")})
                     for s in t["sentences"]]
            for s in sents:
                s.folded = " | ".join(fold(x) for x in (s.cyr, s.lat, s.ru, s.en))
            fields = {k: t[k] for k in Text.__dataclass_fields__ if k != "sentences"}
            self.texts[t["id"]] = Text(**fields, sentences=sents)
            self.order.append(t["id"])
        # сортировка каталога: по названию внутри жанра
        self.order.sort(key=lambda i: (self.texts[i].genre, self.texts[i].title_ru))
        self.index = {tid: k for k, tid in enumerate(self.order)}
        self._playable: Dict[tuple, List[Sentence]] = {}

    # --- выборки -----------------------------------------------------------
    def by_genre(self, genre: str) -> List[Text]:
        return [self.texts[i] for i in self.order if self.texts[i].genre == genre]

    def genres(self) -> List[Tuple[str, str, int]]:
        seen = {}
        for i in self.order:
            t = self.texts[i]
            seen.setdefault(t.genre, [t.genre_ru, 0])[1] += 1
        return [(g, name, n) for g, (name, n) in seen.items()]

    def with_audio(self) -> List[Text]:
        return [self.texts[i] for i in self.order if self.texts[i].full_audio]

    def playable_sentences(self, max_len: float = 20.0, min_words: int = 3,
                           max_words: int = 1000) -> List[Sentence]:
        key = (max_len, min_words, max_words)
        if key not in self._playable:
            self._playable[key] = [
                s for i in self.order for s in self.texts[i].sentences
                if self.texts[i].clip(s.n) and s.duration <= max_len and s.ru
                and min_words <= len(s.cyr.split()) <= max_words]
        return self._playable[key]

    def random_text(self, rng=random) -> Text:
        return rng.choice(self.with_audio())

    def random_sentence(self, rng=random, **kw) -> Sentence:
        return rng.choice(self.playable_sentences(**kw))

    def sentence(self, text_id: str, n: int) -> Sentence:
        return self.texts[text_id].sentences[n - 1]

    # --- поиск -------------------------------------------------------------
    def search(self, query: str, limit: int = 200) -> List[Sentence]:
        """Ищем во всех полях: кириллица, латиница, русский, английский.
        Сначала совпадения с началом слова, потом подстрока."""
        variants = [v.strip() for v in _fold_query(query) if v.strip()]
        if not variants:
            return []
        tiers: List[List[Sentence]] = [[], [], []]
        pats = [(re.compile(r"(?<![\w])" + re.escape(v) + r"(?![\w])"),
                 re.compile(r"(?<![\w])" + re.escape(v))) for v in variants]
        for i in self.order:
            for s in self.texts[i].sentences:
                best = 3
                for v, (whole, start) in zip(variants, pats):
                    if v not in s.folded:
                        continue
                    best = min(best, 0 if whole.search(s.folded) else
                               1 if start.search(s.folded) else 2)
                if best < 3:
                    tiers[best].append(s)
        return (tiers[0] + tiers[1] + tiers[2])[:limit]


def highlight(text: str, query: str) -> Tuple[int, int]:
    """Позиция запроса в исходном тексте с учётом свёртки (или (-1,-1))."""
    for v in _fold_query(query):
        # свёртка посимвольная, кроме удаляемых знаков; ищем по окнам
        f = fold(text)
        if len(f) == len(text):
            p = f.find(v)
            if p >= 0:
                return p, p + len(v)
    return -1, -1
