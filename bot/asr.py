"""Распознавание калмыцкой речи (wav2vec2 CTC, int8 ONNX) с защитой от перегрузки.

Переменные окружения (всё необязательно; ASR выключен, пока ASR_ENABLED != 1):
  ASR_ENABLED        1 — включить распознавание голосовых/аудио/видео (по умолчанию включено)
  ASR_MODEL_PATH     файл модели (по умолчанию DATA_DIR/asr/model_int8.onnx)
  ASR_MODEL_URL      адрес модели; по умолчанию — GitHub Release asr-r4
  ASR_MODEL_SHA256   контрольная сумма модели (если задана — проверяется)
  ASR_LM_PATH/URL/SHA256  файл/адрес/сумма контрольной языковой модели (LM)
  ASR_LM_ENABLED     1 — включить декодирование со словарём (по умолчанию включено)
  ASR_THREADS        потоков ONNX Runtime (по умолчанию 2)
  ASR_MAX_SEC        самая длинная запись, секунд (60)
  ASR_MAX_MB         самый большой файл, МБ (20; у Telegram Bot API лимит скачивания 20 МБ)
  ASR_USER_COOLDOWN  пауза между запросами одного пользователя, секунд (20)
  ASR_USER_DAILY     запросов в сутки на пользователя (20); у админов лимитов нет
  ASR_QUEUE          сколько запросов может ждать очереди (2): дальше бот отвечает «занято»
  ASR_MIN_FREE_MB    не загружать модель, если памяти меньше необходимого резерва (1500)
  ASR_IDLE_UNLOAD    выгружать модель из памяти после N секунд простоя (600)
"""
import asyncio, hashlib, json, logging, os, re, subprocess, tempfile, time, urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("keleni.asr")
HERE = Path(__file__).resolve().parent
SR = 16000
RELEASE = "https://github.com/vbadmaev1/keleni_ayas_bot/releases/download/asr-r4"
MODEL_URL = RELEASE + "/model_int8.onnx"
LM_URL = RELEASE + "/lm_final.arpa"
MODEL_SHA256 = "530d3742e58ffe22b4191de4d369d5e79bcca9d50feeabf4c73cfd55aec04c7f"
LM_SHA256 = "2b5f4cb20fa3b722f9f6c106f8da836bf5a3e31cbb58b849e16fac2a12d06f18"


class AsrError(Exception):
    """Ошибка с текстом, который можно показать пользователю."""


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class Limits:
    """Лимиты на пользователя: пауза и суточная квота. Хранится в памяти (после перезапуска сбрасывается)."""

    def __init__(self, cooldown: int, daily: int):
        self.cooldown, self.daily = cooldown, daily
        self.last: Dict[int, float] = {}
        self.day: Dict[int, Tuple[str, int]] = {}

    def check(self, uid: int, now: Optional[float] = None) -> Optional[str]:
        """None — можно; иначе текст отказа. Если можно, запрос тут же засчитывается."""
        now = time.time() if now is None else now
        wait = self.cooldown - (now - self.last.get(uid, -1e9))
        if wait > 0:
            return f"Подождите ещё {int(wait) + 1} с перед следующей записью."
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        d, n = self.day.get(uid, (today, 0))
        if d != today:
            n = 0
        if n >= self.daily:
            return f"На сегодня лимит расшифровок исчерпан ({self.daily}). Завтра можно снова."
        self.last[uid] = now
        self.day[uid] = (today, n + 1)
        return None

    def refund(self, uid: int) -> None:
        """Запрос не дошёл до распознавания (ошибка/отказ) — не тратим квоту."""
        d, n = self.day.get(uid, ("", 0))
        if n:
            self.day[uid] = (d, n - 1)
        self.last.pop(uid, None)


def free_mb() -> Optional[float]:
    """Memory available to this container, respecting cgroup limits."""
    for base in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        try:
            limit = (base / "memory.max").read_text().strip()
            current = int((base / "memory.current").read_text().strip())
            if limit != "max":
                return max(0, int(limit) - current) / 1024**2
        except (OSError, ValueError):
            pass
    try:
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        current = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
        if limit < 2**60:
            return max(0, limit - current) / 1024**2
    except (OSError, ValueError):
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None   # macOS/Windows: не проверяем


def split(wav: np.ndarray, minlen: float = 2.0, maxlen: float = 18.0, pause: float = 0.30) -> List[Tuple[float, float]]:
    """Режем запись по паузам на куски 2–18 с (модель обучена на коротких предложениях)."""
    fr = int(SR * 0.02)
    n = len(wav) // fr
    if n == 0:
        return []
    e = np.sqrt((wav[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-8
    thr = max(np.percentile(e, 20) * 2.5, 0.004)
    sil = e < thr
    cuts, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * 0.02 >= pause:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    bounds, segs, s = [0] + cuts + [n], [], 0
    for b in bounds[1:]:
        if (b - s) * 0.02 > maxlen:
            k = s
            while (b - k) * 0.02 > maxlen:
                lo, hi = k + int(minlen / 0.02), k + int(maxlen / 0.02)
                m = lo + int(np.argmin(e[lo:hi]))
                segs.append((k, m))
                k = m
            s = k
        segs.append((s, b))
        s = b
    out: List[Tuple[int, int]] = []
    for a, b in segs:
        if out and (out[-1][1] - out[-1][0]) * 0.02 < minlen and (b - out[-1][0]) * 0.02 <= maxlen:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    if len(out) > 1 and (out[-1][1] - out[-1][0]) * 0.02 < minlen and (out[-1][1] - out[-2][0]) * 0.02 <= maxlen:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return [(a * 0.02, b * 0.02) for a, b in out if (b - a) * 0.02 >= 0.4]


def decode_ids(ids, inv: Dict[int, str], pad: int) -> str:
    out, prev = [], -1
    for t in ids:
        if t != prev and t != pad:
            ch = inv[int(t)]
            out.append(" " if ch == "|" else "" if ch == "[UNK]" else ch)
        prev = t
    return re.sub(r"\s+", " ", "".join(out)).strip()


class Recognizer:
    def __init__(self, data_dir: Path):
        self.enabled = os.getenv("ASR_ENABLED", "1") == "1"
        self.path = Path(os.getenv("ASR_MODEL_PATH") or data_dir / "asr" / "model_int8.onnx")
        self.url = os.getenv("ASR_MODEL_URL", MODEL_URL)
        self.sha = os.getenv("ASR_MODEL_SHA256", MODEL_SHA256).lower()
        self.lm_enabled = os.getenv("ASR_LM_ENABLED", "1") == "1"
        self.lm_path = Path(os.getenv("ASR_LM_PATH") or data_dir / "asr" / "lm_final.arpa")
        self.lm_url = os.getenv("ASR_LM_URL", LM_URL)
        self.lm_sha = os.getenv("ASR_LM_SHA256", LM_SHA256).lower()
        self.lm_params = json.loads(os.getenv("ASR_LM_PARAMS", '{"alpha":0.45,"beta":3.0,"unk":-3.5}'))
        self.threads = env_int("ASR_THREADS", 2)
        self.max_sec = env_int("ASR_MAX_SEC", 60)
        self.max_bytes = env_int("ASR_MAX_MB", 20) * 1024 * 1024
        self.queue_max = env_int("ASR_QUEUE", 2)
        self.min_free = env_int("ASR_MIN_FREE_MB", 1500)
        self.idle = env_int("ASR_IDLE_UNLOAD", 600)
        self.limits = Limits(env_int("ASR_USER_COOLDOWN", 20), env_int("ASR_USER_DAILY", 20))
        vocab = json.load(open(HERE / "asr_vocab.json", encoding="utf-8"))
        self.inv = {v: k for k, v in vocab.items()}
        self.pad = vocab["[PAD]"]
        self.session = None
        self._decoder = None
        self._model_verified = False
        self._lm_verified = False
        self.last_used = 0.0
        self._sem = asyncio.Semaphore(1)     # одновременно распознаём только одну запись
        self._waiting = 0

    # --- модель -------------------------------------------------------------
    def available(self) -> bool:
        return self.enabled and (self.path.exists() or bool(self.url))

    @staticmethod
    def _download(url: str, path: Path, sha: str, max_bytes: int, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        log.info("скачиваю %s %s", label, url)
        h = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise AsrError(f"Файл {label} больше допустимого размера.")
                    f.write(chunk)
                    h.update(chunk)
            if sha and h.hexdigest() != sha:
                raise AsrError(f"Файл {label} скачался с ошибкой (не сошлась SHA-256).")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _fetch(self) -> None:
        if not self.url:
            raise AsrError("Модель распознавания не установлена.")
        self._download(self.url, self.path, self.sha, 400 * 1024**2, "модели распознавания")
        self._model_verified = True

    def _fetch_lm(self) -> None:
        if not self.lm_url:
            raise AsrError("Файл языковой модели не установлен.")
        self._download(self.lm_url, self.lm_path, self.lm_sha, 100 * 1024**2, "языковой модели")
        self._lm_verified = True

    @staticmethod
    def _verify_cached(path: Path, sha: str, max_bytes: int, label: str) -> None:
        if path.stat().st_size > max_bytes:
            raise AsrError(f"Локальный файл {label} больше допустимого размера.")
        if sha:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            if h.hexdigest() != sha:
                raise AsrError(f"Локальный файл {label} повреждён (не сошлась SHA-256).")

    def _get_decoder(self):
        if not self.lm_enabled:
            return None
        if self._decoder is not None:
            return self._decoder
        fm = free_mb()
        if fm is not None and fm < env_int("ASR_LM_MIN_FREE_MB", 256):
            raise AsrError("Сейчас не хватает памяти для языковой модели. Попробуйте позже или отключите ASR_LM_ENABLED.")
        if not self.lm_path.exists():
            self._fetch_lm()
        elif not self._lm_verified:
            self._verify_cached(self.lm_path, self.lm_sha, 100 * 1024**2, "языковой модели")
            self._lm_verified = True
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as e:
            raise AsrError("Для декодирования со словарём не установлена pyctcdecode.") from e
        labels = ["" for _ in self.inv]
        for i, ch in self.inv.items():
            labels[i] = " " if ch == "|" else "" if ch == "[PAD]" else "⁇" if ch == "[UNK]" else ch
        words = []
        with open(self.lm_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3 and re.match(r"^-?[\d.]+$", parts[0]) and " " not in parts[1]:
                    words.append(parts[1])
        words = sorted(set(words) - {"<s>", "</s>", "<unk>"})
        self._decoder = build_ctcdecoder(
            labels, kenlm_model_path=str(self.lm_path), unigrams=words,
            alpha=float(self.lm_params["alpha"]), beta=float(self.lm_params["beta"]),
            unk_score_offset=float(self.lm_params["unk"]),
        )
        log.info("языковая модель загружена")
        return self._decoder

    def _load(self):
        if self.session is not None:
            return self.session
        fm = free_mb()
        if fm is not None and fm < self.min_free:
            raise AsrError("Сейчас не хватает памяти на сервере. Попробуйте позже.")
        if not self.path.exists():
            self._fetch()
        elif not self._model_verified:
            self._verify_cached(self.path, self.sha, 400 * 1024**2, "модели распознавания")
            self._model_verified = True
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.enable_mem_pattern = False   # меньше пик памяти на коротких кусках
        self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        log.info("модель загружена: %s", self.path)
        return self.session

    def unload_if_idle(self) -> None:
        if self.session is not None and self.idle and time.time() - self.last_used > self.idle and not self._sem.locked():
            self.session = None
            self._decoder = None
            log.info("модель выгружена из памяти (простой)")

    # --- распознавание ------------------------------------------------------
    def _run(self, src: str) -> Tuple[List[str], float]:
        from imageio_ffmpeg import get_ffmpeg_exe

        try:
            decoded = subprocess.run(
                [get_ffmpeg_exe(), "-nostdin", "-v", "error", "-threads", "1",
                 "-protocol_whitelist", "file,pipe", "-i", src, "-map", "0:a:0",
                 "-t", str(self.max_sec + 1), "-f", "f32le",
                 "-ac", "1", "-ar", str(SR), "-"],
                capture_output=True, timeout=60, check=True,
            )
        except subprocess.TimeoutExpired as e:
            raise AsrError("Не удалось извлечь звук за 60 с.") from e
        except subprocess.CalledProcessError as e:
            raise AsrError("Не получилось извлечь звук из файла.") from e

        wav = np.frombuffer(decoded.stdout, dtype="<f4")
        dur = len(wav) / SR
        if dur > self.max_sec:
            raise AsrError(f"Запись длиннее {self.max_sec} с. Пришлите кусок покороче.")
        if dur < 0.5:
            raise AsrError("В файле нет звука или он слишком короткий.")
        sess = self._load()
        decoder = self._get_decoder()
        texts = []
        for a, b in split(wav):
            x = wav[int(a * SR):int(b * SR)]
            x = ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)
            lg = sess.run(None, {"input": x[None]})[0][0]
            max_logit = np.max(lg, axis=-1, keepdims=True)
            log_probs = lg - max_logit - np.log(np.exp(lg - max_logit).sum(axis=-1, keepdims=True))
            t = (re.sub(r"\s+", " ", decoder.decode(log_probs, beam_width=60)).strip()
                 if decoder is not None else decode_ids(lg.argmax(-1), self.inv, self.pad))
            if t:
                texts.append(t)
        return texts, dur

    def try_reserve(self) -> bool:
        """Reserve a bounded slot before downloading an upload (single asyncio event loop)."""
        if self._waiting >= self.queue_max:
            return False
        self._waiting += 1
        return True

    def release_reservation(self) -> None:
        self._waiting = max(0, self._waiting - 1)

    def queue_full(self) -> bool:
        return self._waiting >= self.queue_max

    async def transcribe_file(self, src: str, *, reserved: bool = False) -> Tuple[List[str], float]:
        if not reserved and not self.try_reserve():
            raise AsrError("Сейчас много запросов. Попробуйте через минуту.")
        try:
            async with self._sem:
                loop = asyncio.get_running_loop()
                # Do not cancel a running executor thread on timeout: Python cannot stop it,
                # and releasing the semaphore would allow a second inference to overlap.
                res = await loop.run_in_executor(None, self._run, src)
                self.last_used = time.time()
                return res
        finally:
            self.release_reservation()
