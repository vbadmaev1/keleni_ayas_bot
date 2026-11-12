"""
Сжимаем записи для Telegram: OGG/Opus 32 кбит/с, моно, 48 кГц.
Для речи это на слух почти не отличается от исходных mp3 192 кбит/с,
а весит в 6 раз меньше; такой файл Telegram показывает как голосовое
сообщение (волна, ускорение воспроизведения).

  audio/full/<id>.ogg          — запись целиком
  audio/sent/<id>/<NNN>.ogg    — отдельные предложения (по таймкодам ts)

Громкость выравнивается (loudnorm): записи 2007–2018 гг. сильно разные.
Запуск: python tools/build_audio.py corpus/raw.json
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path.home() / "Documents/keleni_ayas/kalmyk-1-3"
OUT = Path(__file__).resolve().parent.parent / "audio"
PAD = 0.15          # секунды до/после предложения
OPUS = ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "32k",
        "-application", "voip", "-af", "loudnorm=I=-18:TP=-2:LRA=11"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("FAIL", " ".join(cmd), r.stderr[-500:], file=sys.stderr)


def aligned(t):
    """У части текстов таймкоды фиктивные (все предложения по 3 с)."""
    durs = {round(s["end"] - s["start"], 2) for s in t["sentences"]}
    return t["audio"] and len(durs) > 1


def jobs(texts):
    for t in texts:
        if not t["audio"]:
            continue
        src = next(SRC.glob(f"*/{t['id']}/{t['id']}.mp3"))
        full = OUT / "full" / f"{t['id']}.ogg"
        if not full.exists():
            full.parent.mkdir(parents=True, exist_ok=True)
            yield ["ffmpeg", "-y", "-v", "error", "-i", str(src), *OPUS, str(full)]
        if not aligned(t):
            continue
        for i, s in enumerate(t["sentences"], 1):
            clip = OUT / "sent" / t["id"] / f"{i:03d}.ogg"
            if clip.exists() or s["end"] <= s["start"]:
                continue
            clip.parent.mkdir(parents=True, exist_ok=True)
            a = max(0.0, s["start"] - PAD)
            dur = s["end"] - s["start"] + 2 * PAD
            yield ["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}",
                   "-i", str(src), *OPUS, str(clip)]


if __name__ == "__main__":
    texts = json.load(open(sys.argv[1]))
    todo = list(jobs(texts))
    print(len(todo), "files to encode")
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(run, todo))
    print("done")
