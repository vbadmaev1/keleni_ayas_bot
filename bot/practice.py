"""Логика упражнений без Telegram: сравнение диктанта, варианты для теста."""

import difflib
import html
import random
import re
from typing import List, Tuple

from .corpus import Corpus, Sentence, fold

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.U)


def words(text: str) -> List[str]:
    return _WORD.findall(text)


def _norm(w: str) -> str:
    # для диктанта прощаем пропущенные калмыцкие буквы (я/ю вместо ә/ү)
    return fold(w).replace("я", "а").replace("ю", "у")


def dictation_score(reference: str, answer: str) -> Tuple[float, str]:
    """Сравнение по словам. Возвращает (доля верных слов, HTML-разбор).

    Верное слово — как есть, ошибочное — зачёркнуто ваше и рядом правильное,
    пропущенное — подчёркнуто.
    """
    ref = words(reference)
    ans = words(answer)
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in ref], b=[_norm(w) for w in ans],
                                 autojunk=False)
    parts, good = [], 0
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            good += a2 - a1
            parts += [html.escape(w) for w in ref[a1:a2]]
        elif op == "replace":
            for k in range(max(a2 - a1, b2 - b1)):
                r = ref[a1 + k] if a1 + k < a2 else None
                u = ans[b1 + k] if b1 + k < b2 else None
                if r and u:
                    parts.append(f"<s>{html.escape(u)}</s> <b>{html.escape(r)}</b>")
                elif r:
                    parts.append(f"<u>{html.escape(r)}</u>")
                else:
                    parts.append(f"<s>{html.escape(u)}</s>")
        elif op == "delete":
            parts += [f"<u>{html.escape(w)}</u>" for w in ref[a1:a2]]
        elif op == "insert":
            parts += [f"<s>{html.escape(w)}</s>" for w in ans[b1:b2]]
    score = good / len(ref) if ref else 0.0
    return score, " ".join(parts)


def quiz_options(corpus: Corpus, s: Sentence, lang: str = "ru", k: int = 4,
                 rng=random) -> Tuple[List[str], int]:
    """Правильный перевод + k-1 переводов других предложений похожей длины."""
    field = "en" if lang == "en" else "ru"
    right = getattr(s, field)
    pool = [x for x in corpus.playable_sentences()
            if getattr(x, field) and getattr(x, field) != right
            and (x.text_id, x.n) != (s.text_id, s.n)]
    L = len(right)
    pool.sort(key=lambda x: abs(len(getattr(x, field)) - L))
    near = pool[:60]
    wrong = rng.sample(near, min(k - 1, len(near)))
    opts = [getattr(x, field) for x in wrong] + [right]
    rng.shuffle(opts)
    return opts, opts.index(right)
