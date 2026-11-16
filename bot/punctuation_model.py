"""Статистическое восстановление знака после слова в калмыцком тексте."""
import zlib
from functools import lru_cache
import logging
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

MARKS = ("", ".", ",", "?")
FEATURE_COUNT = 1 << 16
ACOUSTIC_FEATURE_COUNT = 10
MODEL_PATH = Path(__file__).with_name("punctuation_model.npz")
QUESTION_WORDS = frozenset("кен ю юн юун юуга юунд юуна юңгад яһад яһҗ ямар ямаран альк аль кезә кедү хама хальд".split())


def normalize(word: str) -> str:
    return word.lower().replace("ӛ", "ө")


@lru_cache(maxsize=32768)
def feature_id(feature: str) -> int:
    return zlib.crc32(feature.encode("utf-8")) % FEATURE_COUNT


def features(left: list[str], right: list[str]) -> tuple[int, ...]:
    previous = left[-2] if len(left) > 1 else ""
    word = left[-1]
    following, after = (right + ["", ""])[:2]
    values = [
        "w:" + word, "p:" + previous, "n:" + following, "a:" + after,
        "w+n:" + word + "/" + following,
        "p+w:" + previous + "/" + word,
        "n+a:" + following + "/" + after,
    ]
    # Avoid artificial features that only occur at the ends of input buffers.
    for n in (1, 2, 3, 4):
        if len(word) >= n:
            values.append(f"s{n}:" + word[-n:])
        if len(following) >= n:
            values.append(f"f{n}:" + following[:n])
    if following and word:
        values.extend(("s+n:" + word[-2:] + "/" + following,
                       "s+f:" + word[-2:] + "/" + following[:2]))
    question = next((w for w in reversed(left[-5:]) if w in QUESTION_WORDS), "")
    if question:
        values.extend(("q:" + question, "q+s:" + question + "/" + word[-2:]))
    return tuple(feature_id(value) for value in values if not value.endswith((':', '/')))


def acoustic_features(text_scores: np.ndarray, quiet: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    # A CTC gap can contain sound. Keep its duration and the verified silence
    # as separate features, with bounded influence from very long gaps.
    pauses = np.clip(quiet, 0, 1.5)
    gap = np.clip(gaps, 0, 1.5)
    relative = np.clip(text_scores[:, 1:] - text_scores[:, :1], -8, 8)
    return np.column_stack((relative, pauses, gap, pauses >= .14, pauses >= .40,
                            pauses >= .68, pauses >= 1.0, gap >= .22))


class PunctuationModel:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = data["weights"]
            self.bias = data["bias"]
            self.audio_weights = data["audio_weights"]
            self.audio_bias = data["audio_bias"]
        if (self.weights.shape != (FEATURE_COUNT, len(MARKS))
                or self.bias.shape != (len(MARKS),)
                or self.audio_weights.shape != (ACOUSTIC_FEATURE_COUNT, len(MARKS))
                or self.audio_bias.shape != (len(MARKS),)):
            raise ValueError("Неверная форма весов модели пунктуации")

    def text_scores(self, words: list[str]) -> np.ndarray:
        tokens = [normalize(w) for w in words]
        return np.array([
            self.weights[list(features(tokens[max(0, i-4):i+1], tokens[i+1:i+3]))].sum(axis=0)
            + self.bias for i in range(len(tokens))
        ]).reshape(len(tokens), len(MARKS))

    def score(self, words: list[str], quiet=None, gaps=None) -> np.ndarray:
        scores = self.text_scores(words)
        if quiet is not None and gaps is not None:
            gaps = np.asarray(gaps)
            timed = np.isfinite(gaps)
            calibrated = acoustic_features(scores[:-1][timed], np.asarray(quiet)[timed], gaps[timed])
            scores[np.flatnonzero(timed)] = calibrated @ self.audio_weights + self.audio_bias
        return scores

    def predict(self, words: list[str]) -> list[str]:
        return [MARKS[i] for i in self.score(words).argmax(axis=1)]


def timed_mark_indices(scores: np.ndarray, rule_questions: np.ndarray) -> np.ndarray:
    # Preserve the existing question detector until there is enough
    # annotated speech to validate a replacement for this rare class.
    return np.where(rule_questions, MARKS.index("?"), scores[:, :MARKS.index("?")].argmax(axis=1))


@lru_cache(maxsize=1)
def load_model() -> PunctuationModel | None:
    if not MODEL_PATH.is_file():
        return None
    try:
        return PunctuationModel(MODEL_PATH)
    except (OSError, KeyError, ValueError, BadZipFile):
        logging.getLogger(__name__).exception("Не удалось загрузить модель пунктуации")
        return None


def restore_text(text: str) -> str:
    """Восстанавливает пунктуацию в тексте без пунктуации и без таймкодов."""
    words = text.split()
    if not words:
        return ""
    model = load_model()
    if model is None:
        raise FileNotFoundError(MODEL_PATH)
    marks = model.predict(words)
    marks[-1] = "?" if marks[-1] == "?" else "."
    return " ".join(render_sentences(words, marks))


def render_sentences(words: list[str], marks: list[str]) -> list[str]:
    sentences, phrase = [], []
    for word, mark in zip(words, marks):
        if not phrase:
            word = word[0].upper() + word[1:]
        phrase.append(word + mark)
        if mark in {".", "?"}:
            sentences.append(" ".join(phrase))
            phrase = []
    if phrase:
        sentences.append(" ".join(phrase))
    return sentences


if __name__ == "__main__":
    import sys

    print(restore_text(sys.stdin.read()))
