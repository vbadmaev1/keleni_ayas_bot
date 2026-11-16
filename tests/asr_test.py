"""Проверка защиты распознавания от перегрузки. Без сети и без модели; если модель есть
(ASR_MODEL_PATH), дополнительно прогоняем реальную запись.   Запуск:  python tests/asr_test.py"""
import asyncio, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from bot import asr
from bot.asr import AsrError, Limits, Recognizer, decode_ids, split

# 1. лимиты пользователя
L = Limits(cooldown=20, daily=3); t0 = 1_000_000.0
assert L.check(1, t0) is None
assert "Подождите" in L.check(1, t0 + 5)            # слишком часто
assert L.check(1, t0 + 25) is None and L.check(1, t0 + 50) is None
assert "лимит" in L.check(1, t0 + 75)               # четвёртый за сутки
assert L.check(2, t0) is None                        # другой пользователь не затронут
L.refund(2); assert L.check(2, t0 + 1) is None       # отказ не тратит квоту
print("лимиты: ок")

# 2. резка по паузам: куски не длиннее 18 с, пауза — граница
sr = asr.SR; rng = np.random.default_rng(0)
speech = lambda s: (rng.standard_normal(int(sr * s)) * 0.1).astype(np.float32)
quiet = lambda s: np.zeros(int(sr * s), np.float32)
w = np.concatenate([speech(5), quiet(0.6), speech(7), quiet(0.6), speech(40)])
segs = split(w); assert segs and all(b - a <= 18.5 for a, b in segs), segs
assert segs[0][1] < 6.5 and segs[-1][1] >= 52          # граница у первой паузы, хвост не потерян
assert split(np.zeros(10, np.float32)) == []
print("резка:", len(segs), "кусков, ок")

# 3. декодер CTC: повторы схлопываются, blank и UNK убираются
inv = {0: "а", 1: "б", 2: "|", 3: "[PAD]", 4: "[UNK]"}
assert decode_ids([0, 0, 3, 0, 2, 1, 1, 4, 2, 2], inv, 3) == "аа б"
print("декодер: ок")

# 4. отказы до распознавания
async def limits_tests():
    keys = ("ASR_ENABLED", "ASR_QUEUE", "ASR_MAX_SEC", "ASR_MIN_FREE_MB", "ASR_MODEL_PATH", "ASR_MODEL_URL",
            "ASR_LM_PATH", "ASR_LM_URL")
    previous = {k: os.environ.get(k) for k in keys}
    with tempfile.TemporaryDirectory() as td:
        os.environ.update(ASR_ENABLED="1", ASR_QUEUE="1", ASR_MAX_SEC="10",
                          ASR_MIN_FREE_MB="1500",
                          ASR_MODEL_PATH=str(Path(td) / "missing.onnx"), ASR_MODEL_URL="",
                          ASR_LM_PATH=str(Path(td) / "missing.arpa"), ASR_LM_URL="")
        rec = Recognizer(Path(td)); assert rec.enabled and not rec.available()   # модели нет и URL нет
        try: rec._load(); raise SystemExit("ожидали отказ без модели")
        except AsrError: pass
        # очередь: один запрос занимает слот, второй получает «занято», а не ставится бесконечно
        started = asyncio.Event(); release = asyncio.Event()
        def slow(src, punctuation): time.sleep(0.3); return ["x"], 1.0
        rec._run = slow
        t1 = asyncio.create_task(rec.transcribe_file("a")); await asyncio.sleep(0.05)
        try: await rec.transcribe_file("b"); raise SystemExit("ожидали «занято»")
        except AsrError as e: assert "много запросов" in str(e)
        assert (await t1)[0] == ["x"]
        # длинная запись отклоняется до загрузки модели
        wav = Path(td) / "long.wav"
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "30", str(wav)], check=True)
        rec2 = Recognizer(Path(td))
        try: rec2._run(str(wav)); raise SystemExit("ожидали отказ: длинная запись")
        except AsrError as e: assert "длиннее" in str(e)
        # нехватка памяти
        orig = asr.free_mb; asr.free_mb = lambda: 100.0
        rec3 = Recognizer(Path(td)); rec3.path = Path(td) / "m.onnx"; rec3.path.write_bytes(b"x")
        try: rec3._load(); raise SystemExit("ожидали отказ: мало памяти")
        except AsrError as e: assert "памяти" in str(e)
        asr.free_mb = orig
        rec.queue_max = 2
        assert rec.estimate_seconds(60) > rec.estimate_seconds(15)
        assert rec.estimate_seconds(60, 10 * 1024**2) > rec.estimate_seconds(60)
        assert rec.estimate_seconds(60, ahead=1) > rec.estimate_seconds(60)
        assert rec.estimate_seconds(None) is None
        first_started = asyncio.Event()
        first_release = asyncio.Event()
        second_started = asyncio.Event()

        async def hold_first():
            first_started.set()
            await first_release.wait()

        async def start_second():
            second_started.set()

        first = asyncio.create_task(rec.transcribe_file("a", on_start=hold_first))
        await first_started.wait()
        assert rec.try_reserve() and rec.requests_ahead == 1
        second = asyncio.create_task(rec.transcribe_file("b", reserved=True, on_start=start_second))
        await asyncio.sleep(0)
        assert not second_started.is_set()
        first_release.set()
        await asyncio.gather(first, second)
        assert second_started.is_set() and rec._waiting == 0
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
asyncio.run(limits_tests()); print("отказы (очередь, длина, память): ок")

# 5. реальная модель, если есть
mp = os.getenv("ASR_MODEL_PATH")
if mp and Path(mp).exists():
    clip = sorted((Path(__file__).resolve().parent.parent / "audio/sent/KSG_NNL_XZD_2008_Dialog_conv").glob("*.ogg"))[0]
    os.environ["ASR_ENABLED"] = "1"; rec = Recognizer(Path(tempfile.mkdtemp()))
    t = time.time(); texts, dur = asyncio.run(rec.transcribe_file(str(clip)))
    print("модель:", texts, f"({dur:.1f} с звука за {time.time() - t:.1f} с)"); assert texts and all(isinstance(x, str) for x in texts)
print("OK")
