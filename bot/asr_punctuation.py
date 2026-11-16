"""Консервативная пунктуация для калмыцкой расшифровки с таймкодами CTC."""
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional

import numpy as np

FRAME_SECONDS = 0.02
QUESTION_WORDS = frozenset({
    "кен", "ю", "юн", "юун", "юуга", "юунд", "юуна", "юңгад", "яһад",
    "яһҗ", "ямар", "ямаран", "альк", "аль", "кезә", "кедү", "хама", "хальд",
})
QUESTION_ENDINGS = frozenset({
    "кемб", "кенб", "күмб", "күмбч", "юв", "юмб", "юмбч", "юмбт",
    "яахв", "яһхв", "яһв", "яахмб", "яһсмб", "яһла", "болхв", "болсмб", "билү", "бәәнү",
})
REPORTING_WORDS = frozenset({
    "гих", "гихлә", "гиҗ", "гисн", "гиһәд", "гигәд", "гидг",
    "гиҗәнә", "гинә", "гив", "сурв", "сурна", "сурҗана", "келв", "келнә",
})
SPEECH_TAGS = frozenset({"гиҗ", "гинә", "гив", "гиҗәнә"})
CONVERB_ENDINGS = ("аад", "әәд", "иһәд", "гөөд", "аһад", "әһәд", "ад", "әд", "җ")
CONJUNCTIONS = frozenset({"болн", "эс", "бую", "эсвл", "болв", "зуг"})
COMMA_BEFORE = frozenset({"болв", "болвчн", "зуг"})

@dataclass(frozen=True)
class TimedWord:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


def beam_words(text: str, frames: Iterable, offset: float) -> list[TimedWord]:
    words = text.split()
    timings = list(frames)
    if words != [word for word, _ in timings]:
        return [TimedWord(word) for word in words]
    return [
        TimedWord(word, offset + a * FRAME_SECONDS, offset + b * FRAME_SECONDS)
        if 0 <= a < b else TimedWord(word)
        for word, (a, b) in timings
    ]


def greedy_words(ids, tokens: dict[int, str], blank: int, offset: float) -> list[TimedWord]:
    words = []
    letters = []
    start = end = None
    last = blank
    for frame, token in enumerate([*ids, blank]):
        symbol = tokens[int(token)]
        if frame == len(ids) or symbol == '|':
            if letters:
                words.append(TimedWord(''.join(letters), offset + start * FRAME_SECONDS,
                                       offset + end * FRAME_SECONDS))
                letters = []
        elif len(symbol) == 1:
            if last != token:
                if not letters:
                    start = frame
                letters.append(symbol)
            if letters:
                end = frame + 1
        last = token
    return words


def quiet_frames(wave: np.ndarray, rate: int) -> np.ndarray:
    samples = round(rate * FRAME_SECONDS)
    size = len(wave) // samples
    if not size:
        return np.empty(0, dtype=bool)
    rms = np.sqrt(np.mean(wave[:size * samples].reshape(size, samples) ** 2, axis=1))
    floor, speech = np.percentile(rms, [15, 85])
    threshold = max(0.0001, min(2.5 * floor, .16 * speech), .025 * speech)
    return rms <= threshold


def pause_seconds(left: TimedWord, right: TimedWord, quiet: np.ndarray) -> float:
    if left.end is None or right.start is None or right.start <= left.end:
        return 0.0
    a, b = (round(t / FRAME_SECONDS) for t in (left.end, right.start))
    silence = quiet[max(0, a):b]
    if not len(silence):
        return 0.0
    # CTC blank frames are common inside speech. Confirm a continuous quiet
    # interval in the waveform before treating the gap as a pause.
    changes = np.flatnonzero(np.diff(np.r_[False, silence, False]))
    return int((changes[1::2] - changes[::2]).max(initial=0)) * FRAME_SECONDS


def is_question(words: list[str], before_speech_tag: bool) -> bool:
    if words[-1] in QUESTION_ENDINGS:
        return True
    if len(words) == 1:
        return words[0] in QUESTION_WORDS
    if before_speech_tag:
        return (any(w in QUESTION_WORDS for w in words[-6:])
                and words[-1] not in REPORTING_WORDS)
    return (len(words) <= 12 and any(w in QUESTION_WORDS for w in words[:3])
            and all(w not in REPORTING_WORDS for w in words[1:]))


def punctuate(words: list[TimedWord], quiet: np.ndarray) -> list[str]:
    if not words:
        return []
    pauses = [pause_seconds(a, b, quiet) for a, b in zip(words, words[1:])]
    positive_pauses = [p for p in pauses if p >= .08]
    turn_pause = max(.46, min(.66, 2.2 * median(positive_pauses))) if positive_pauses else .46
    output = []
    phrase = []
    marks = []
    words_since_mark = 0
    for i, word in enumerate(words):
        phrase.append(word.text)
        words_since_mark += 1
        value = word.text.casefold()
        before_speech_tag = False
        if i == len(words) - 1:
            boundary = True
            comma = False
        else:
            pause = pauses[i]
            following = words[i + 1].text.casefold()
            before_speech_tag = following in SPEECH_TAGS
            question_mark = (
                value in QUESTION_ENDINGS and (pause >= .12 or before_speech_tag)
                or before_speech_tag and pause >= .14
                and is_question([token.casefold() for token in phrase], True)
            )
            boundary = question_mark or value not in CONJUNCTIONS and not before_speech_tag and (
                (len(phrase) >= 4 and pause >= turn_pause)
                or (len(phrase) >= 2 and pause >= .76)
            )
            comma = words_since_mark >= 3 and (
                (pause >= .20 and (following in COMMA_BEFORE or before_speech_tag))
                or (pause >= .22 and value.endswith(CONVERB_ENDINGS))
                or (pause >= .32 and value not in CONJUNCTIONS)
            )
        if boundary:
            plain = [token.casefold() for token in phrase]
            text = ' '.join(part + mark for part, mark in zip(phrase, [*marks, '']))
            output.append(text[0].upper() + text[1:] + ('?' if is_question(plain, before_speech_tag) else '.'))
            phrase = []
            marks = []
            words_since_mark = 0
        else:
            marks.append(',' if comma else '')
            if comma:
                words_since_mark = 0
    return output
