#!/usr/bin/env python3
"""Ручная разметка -> corpus/manual.json + audio/{full,sent}/manual_*.
Источник: research/w2v/review/labels.json (строки со статусом ✓ — проверенные человеком) и review/audio/<id>.m4a.
Тексты — кириллица, исправленная вручную после распознавания моделью; перевода и транскрипции нет.
Запуск:  python3 tools/add_manual.py  """
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "research/w2v/review"
OPUS = ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", "-application", "voip",
        "-af", "loudnorm=I=-18:TP=-2:LRA=11"]
NOTE = "Текст проверен вручную. В режиме «По предложениям» доступны проверенные фрагменты. Перевода нет."
SARPA_RECORDINGS = {"sarpa", "sarpa2", "sarpa3", "video_0900"}
TITLES = {
    "sarpa": "Открытие хурула в посёлке Сарпа",
    "sarpa2": "Сарпа: Павел Лиджиев о совхозе и «Джангаре»",
    "sarpa3": "Сарпа: Канур Утаджиев о посёлке и хуруле",
    "kalyan_sanji": "Санджи Каляев о калмыцком языке и грамотности",
    "video_0859": "О писательском мастерстве",
    "video_0900": "Воспоминания о Хошутовском хуруле",
    "vk_173737045_896": "О писателе Басане Дорджиеве",
    "vk_173737045_814": "Геше Санал о буддийском образовании",
}


def run(cmd):
    if subprocess.run(cmd).returncode:
        sys.exit(f"ffmpeg не справился: {' '.join(map(str, cmd))}")


def main():
    labels = json.load(open(REV / "labels.json", encoding="utf-8"))
    out = []
    for fid, title in TITLES.items():
        segs = labels[fid]
        src = REV / "audio" / f"{fid}.m4a"
        good = sorted((s for s in segs if s.get("status") == "ok" and s["text"].strip()
                       and s["end"] - s["start"] >= 0.8), key=lambda s: s["start"])
        if not good or not src.exists():
            sys.exit(f"Нет подтверждённых сегментов или аудио для {fid}")
        tid = f"manual_{fid}"
        (ROOT / "audio/full").mkdir(parents=True, exist_ok=True)
        (ROOT / "audio/sent" / tid).mkdir(parents=True, exist_ok=True)
        full = ROOT / "audio/full" / f"{tid}.ogg"
        if not full.exists():
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *OPUS, str(full)])
        sents = []
        for n, s in enumerate(good, 1):
            clip = ROOT / "audio/sent" / tid / f"{n:03d}.ogg"
            a, dur = max(0.0, s["start"] - 0.05), s["end"] - s["start"] + 0.1
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", str(src), *OPUS, str(clip)])
            text = " ".join(s["text"].split())
            sents.append({"n": n, "start": round(s["start"], 2), "end": round(s["end"], 2), "speaker": "",
                          "lat": "", "cyr": text[0].upper() + text[1:], "ru": "", "en": ""})
        duration = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(full)],
            text=True,
        ))
        if fid in SARPA_RECORDINGS:
            genre, genre_ru = "manual", "Сарпа"
        else:
            genre, genre_ru = "manual_other", "Другое"
        out.append({"id": tid, "genre": genre, "genre_ru": genre_ru, "title_ru": title,
                    "title_en": "", "dialect": "", "year": "", "place": "", "speakers": [], "audio": True,
                    "aligned": True, "duration": round(duration, 1), "partial": False,
                    "note": NOTE, "sentences": sents})
        print(fid, len(sents), "предложений")
    json.dump(out, open(ROOT / "corpus/manual.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записей:", len(out), "предложений:", sum(len(t["sentences"]) for t in out))


if __name__ == "__main__":
    main()
