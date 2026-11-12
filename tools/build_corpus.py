"""
raw.json + kalmyk.coma -> data/corpus.json для бота.

У каждого предложения: латиница (исходная транскрипция без разметки),
кириллица (tools/to_cyrillic.py), переводы ru/en, таймкоды и список слов,
которых не нашлось в словаре (кандидаты на ручную правку,
см. data/review_unknown.tsv и data/overrides.tsv).

Ручные правки: data/overrides.tsv, строки «латиница<TAB>кириллица».
Применяются к слову целиком (без учёта регистра первой буквы).
"""

import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path.home() / "Documents/keleni_ayas/kalmyk-1-3"
sys.path.insert(0, str(ROOT / "tools"))
from to_cyrillic import Converter, clean_markup  # noqa: E402

GENRES = {"flk": "Фольклор", "nar": "Рассказ", "conv": "Разговор"}
DIALECTS = {"Derbet": "дербетский", "Torgut": "торгутский", "Buzava": "бузавский"}


def coma_meta():
    root = ET.parse(SRC / "kalmyk.coma").getroot()
    speakers = {}
    for s in root.iter("Speaker"):
        keys = {k.get("Name"): (k.text or "") for k in s.iter("Key")}
        speakers[s.get("Id")] = dict(
            code=s.findtext("Sigle"), name=s.findtext("Pseudo") or "",
            sex=s.findtext("Sex") or "", born=keys.get("4 Year of birth", ""))
    comms = {}
    for c in root.iter("Communication"):
        desc = c.find("Description")
        keys = {k.get("Name"): (k.text or "") for k in desc.iter("Key")} if desc is not None else {}
        loc = {k.get("Name"): (k.text or "") for k in c.findall("./Location/Description/Key")}
        people = [speakers[p.text] for p in c.findall("./Setting/Person") if p.text in speakers]
        comms[c.get("Name")] = dict(
            title_en=keys.get("0a Title", ""), title_ru=keys.get("0b Title (RU)", ""),
            dialect=keys.get("3 Dialect", ""), year=keys.get("2b Year of recording", ""),
            place=loc.get("3b Settlement (RU)", ""), speakers=people)
    return comms


def audio_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return float(out or 0)


def main():
    raw = json.load(open(ROOT / "data/raw.json"))
    meta = coma_meta()
    conv = Converter()
    overrides = {}
    ov = ROOT / "data/overrides.tsv"
    if ov.exists():
        for row in csv.reader(open(ov), delimiter="\t"):
            if len(row) >= 2 and not row[0].startswith("#"):
                overrides[row[0].lower()] = row[1]
    for lat, cyr in overrides.items():
        conv.cache[lat] = (cyr, True)

    unknown = Counter()
    unknown_ex = {}
    texts = []
    for t in raw:
        m = meta.get(t["id"], {})
        durs = {round(s["end"] - s["start"], 2) for s in t["sentences"]}
        aligned = t["audio"] and len(durs) > 1
        sents = []
        for i, s in enumerate(t["sentences"], 1):
            cyr, unk = conv.sentence(s["ts"])
            cyr = cyr[:1].upper() + cyr[1:]
            for u in unk:
                unknown[u] += 1
                unknown_ex.setdefault(u, (t["id"], i))
            sents.append(dict(n=i, start=round(s["start"], 2), end=round(s["end"], 2),
                              speaker=s["speaker"], lat=clean_markup(s["ts"]), cyr=cyr,
                              ru=s["ru"], en=s["en"]))
        full_sec = 0.0
        if t["audio"]:
            full_sec = audio_seconds(next(SRC.glob(f"*/{t['id']}/{t['id']}.mp3")))
        last = t["sentences"][-1]["end"] if t["sentences"] else 0
        texts.append(dict(
            id=t["id"], genre=t["genre"], genre_ru=GENRES.get(t["genre"], t["genre"]),
            title_ru=m.get("title_ru") or t["id"], title_en=m.get("title_en", ""),
            dialect=DIALECTS.get(m.get("dialect", ""), m.get("dialect", "")),
            year=m.get("year", ""), place=m.get("place", ""),
            speakers=m.get("speakers", []), audio=t["audio"], aligned=aligned,
            duration=round(full_sec, 1),
            # текст расшифрован только для начала записи
            partial=bool(aligned and full_sec and last < full_sec * 0.8),
            sentences=sents))

    json.dump(texts, open(ROOT / "data/corpus.json", "w"), ensure_ascii=False, indent=1)
    with open(ROOT / "data/review_unknown.tsv", "w") as f:
        f.write("# латиница\tкириллица (авто)\tсколько раз\tпример (текст, предложение)\n")
        for w, n in unknown.most_common():
            f.write(f"{w}\t{conv.word(w)[0]}\t{n}\t{unknown_ex[w][0]} #{unknown_ex[w][1]}\n")
    total = sum(len(s["cyr"].split()) for t in texts for s in t["sentences"])
    print(f"{len(texts)} texts, {sum(len(t['sentences']) for t in texts)} sentences, "
          f"{total} tokens, not in vocab: {sum(unknown.values())} ({len(unknown)} types)")


if __name__ == "__main__":
    main()
