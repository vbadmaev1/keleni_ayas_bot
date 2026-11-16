"""Обучение пунктуации из corpus_sentences.jsonl и калибровка по размеченным паузам."""
import argparse
from array import array
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.punctuation_model import FEATURE_COUNT, MARKS, acoustic_features, features, normalize, timed_mark_indices

TOKEN = re.compile(r"[а-яёҗңүһәөӛ]+|[.,!?;:…–—]", re.I)


def blocks(path):
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        key = lambda r: (r["book_id"], r["section_id"])
        for (book, _), section in groupby(rows, key=key):
            text = []
            for row in section:
                if (row.get("verse") or row.get("ocr_suspect")
                        or row.get("k_share", 1) < .75 or row.get("unk_share", 0) > .65):
                    if text:
                        yield book, " ".join(text)
                        text = []
                    continue
                text.append(row["text"])
            if text:
                yield book, " ".join(text)


def tokenize(text):
    words, marks = [], []
    for token in TOKEN.findall(text):
        if token.isalpha():
            words.append(normalize(token))
            marks.append("")
        elif marks:
            if token == "!":
                token = "."
            if token == "…" and marks[-1] == "":
                token = "."
            if marks[-1] not in {".", "?"} or token == "?":
                marks[-1] = token
    return words, marks


def vectors(path, test_book):
    indices = array("i")
    offsets = array("i", [0])
    labels, test = bytearray(), bytearray()
    totals = Counter()
    for book, text in blocks(path):
        words, marks = tokenize(text)
        for i, mark in enumerate(marks[:-1]):
            if mark not in MARKS:
                continue
            ids = features(words[max(0, i - 4):i + 1], words[i + 1:i + 3])
            indices.extend(ids)
            offsets.append(len(indices))
            labels.append(MARKS.index(mark))
            test.append(book == test_book)
            totals[book] += 1
    if test_book not in totals or len(totals) < 2:
        raise ValueError("Укажите отдельную книгу для проверки")
    matrix = csr_matrix((np.ones(len(indices), dtype=np.float32),
                         np.frombuffer(indices, dtype=np.int32), np.frombuffer(offsets, dtype=np.int32)),
                        shape=(len(labels), FEATURE_COUNT))
    return matrix, np.frombuffer(labels, dtype=np.uint8), np.frombuffer(test, dtype=np.bool_), totals


def scores(target, predicted):
    per_mark = f1_score(target, predicted, labels=[1, 2, 3], average=None, zero_division=0)
    return np.round(100 * per_mark, 1).tolist()


def acoustic_data(paths, text_model):
    rows = []
    for path in paths:
        if path.suffix == ".json":
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    ids = [features([normalize(w) for w in r["left"]], [normalize(w) for w in r["right"]])
           for r in rows]
    logits = np.stack([text_model.coef_[:, list(idx)].sum(axis=1) + text_model.intercept_ for idx in ids])
    matrix = acoustic_features(logits, np.array([r["pause"] for r in rows]),
                              np.array([r["gap"] for r in rows]))
    y = np.array([MARKS.index(r["ref"]) if r["ref"] in MARKS else 0 for r in rows])
    groups = np.array([str(r["track"]) for r in rows])
    baseline = np.array([MARKS.index(r["hyp"]) if r["hyp"] in MARKS else 0 for r in rows])
    return matrix, y, groups, logits.argmax(axis=1), baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audio-boundaries", type=Path, required=True, nargs="+")
    parser.add_argument("--test-book", default="эрнҗана_ботхн_һалан_хадһл")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    x, y, test, totals = vectors(args.corpus, args.test_book)
    print("BOOKS", dict(totals), "COUNTS", Counter(y.tolist()), flush=True)
    train = ~test
    lex = SGDClassifier(loss="log_loss", alpha=2e-6, average=True, max_iter=17, tol=None,
                        class_weight={0: 1, 1: 2.1, 2: 2.0, 3: 4.5}, random_state=94)
    lex.fit(x[train], y[train])
    print("TEXT", int(train.sum()), int(test.sum()), scores(y[test], lex.predict(x[test])), flush=True)

    matrix, labels, groups, text_pred, heuristic = acoustic_data(args.audio_boundaries, lex)
    fusion = LogisticRegression(solver="newton-cholesky", C=1, max_iter=500,
                                class_weight={0: 1, 1: 1.7, 2: 1.6, 3: 2.3})
    fused = np.empty_like(labels)
    for train_idx, test_idx in LeaveOneGroupOut().split(matrix, labels, groups):
        fusion.fit(matrix[train_idx], labels[train_idx])
        logits = fusion.decision_function(matrix[test_idx])
        fused[test_idx] = timed_mark_indices(logits, heuristic[test_idx] == MARKS.index("?"))
    print("AUDIO_COUNTS", len(labels), len(set(groups)), dict(Counter(labels.tolist())), flush=True)
    print("HEURISTIC", scores(labels, heuristic), "TEXT_ONLY", scores(labels, text_pred),
          "FUSED", scores(labels, fused), flush=True)

    fusion.fit(matrix, labels)
    if not np.array_equal(fusion.classes_, np.arange(len(MARKS))):
        raise ValueError("В разметке пауз нужны примеры каждого из трёх знаков и границ без знака")
    np.savez_compressed(
        args.output,
        weights=lex.coef_.T.astype(np.float32), bias=lex.intercept_.astype(np.float32),
        audio_weights=fusion.coef_.T.astype(np.float32), audio_bias=fusion.intercept_.astype(np.float32),
    )


if __name__ == "__main__":
    main()
