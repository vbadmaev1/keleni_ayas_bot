"""Достаём из EXB предложения: время, ts, слова tx + mb/ge, переводы."""
import glob, json, os, sys, xml.etree.ElementTree as ET

SRC = os.path.expanduser("~/Documents/keleni_ayas/kalmyk-1-3")

def load(path):
    root = ET.parse(path).getroot()
    tl = {x.get("id"): float(x.get("time") or 0) for x in root.iter("tli")}
    order = [x.get("id") for x in root.iter("tli")]
    pos = {k: i for i, k in enumerate(order)}
    tiers = {}
    for tr in root.iter("tier"):
        key = (tr.get("speaker"), tr.get("category"))
        tiers[key] = [(e.get("start"), e.get("end"), e.text or "") for e in tr.findall("event")]
    speakers = sorted({k[0] for k in tiers})
    sents = []
    for spk in speakers:
        ts = tiers.get((spk, "ts"), [])
        words = tiers.get((spk, "tx"), [])
        ann = {c: {s: t for s, _, t in tiers.get((spk, c), [])} for c in ("mb", "ge", "gr", "ps")}
        fr = {s: t for s, _, t in tiers.get((spk, "fr"), [])}
        en = {s: t for s, _, t in tiers.get((spk, "lte"), []) + tiers.get((spk, "fe"), [])}
        for s, e, text in ts:
            a, b = pos[s], pos[e]
            ws = [dict(tx=t.strip(), **{c: ann[c].get(ws_, "") for c in ann})
                  for ws_, _, t in words if a <= pos[ws_] < b]
            sents.append(dict(speaker=spk, start=tl[s], end=tl[e], ts=text.strip(),
                              words=ws, ru=fr.get(s, "").strip(), en=en.get(s, "").strip()))
    sents.sort(key=lambda x: x["start"])
    return sents

out = []
for f in sorted(glob.glob(f"{SRC}/*/*/*.exb")):
    name = os.path.basename(f)[:-4]
    mp3 = f[:-4] + ".mp3"
    out.append(dict(id=name, genre=f.split("/")[-3], audio=os.path.exists(mp3) and os.path.getsize(mp3) > 0,
                    sentences=load(f)))
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(len(out), sum(len(t["sentences"]) for t in out))
