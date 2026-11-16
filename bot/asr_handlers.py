"""Роутер: голосовые, аудио и видео -> расшифровка. Подключается в main.py, только если ASR включён и модель есть."""
import asyncio, html, logging, math, os, tempfile
from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from .asr import AsrError, Recognizer
from .storage import Storage

log = logging.getLogger("keleni.asr")
NOTE = "<i>Автоматическая расшифровка может содержать ошибки; проверяйте по записи.</i>"


async def probe_duration(path: str) -> float | None:
    """Telegram не сообщает длительность файлов, присланных документом."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return None
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), 5)
        seconds = float(output)
        return seconds if proc.returncode == 0 and math.isfinite(seconds) and seconds > 0 else None
    except asyncio.TimeoutError:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()
    except ValueError:
        pass
    return None


def estimate_text(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, rest = divmod(5 * math.ceil(seconds / 5), 60)
    duration = f"{rest} с"
    if minutes:
        duration = f"{minutes} мин" + (f" {duration}" if rest else "")
    return f"\nПримерное время до результата — {duration}."


def build_asr_router(rec: Recognizer, admins: set, storage: Storage) -> Router:
    r = Router()
    MEDIA = F.voice | F.audio | F.video | F.video_note | (F.document & F.document.mime_type.regexp(r"^(audio|video)/"))

    @r.message(MEDIA)
    async def on_media(m: Message):
        f = m.voice or m.audio or m.video or m.video_note or m.document
        uid = m.from_user.id
        admin = uid in admins
        if f.file_size and f.file_size > rec.max_bytes:
            await m.answer(f"Файл больше {rec.max_bytes // 2**20} МБ. Пришлите покороче.")
            return
        dur = getattr(f, "duration", None)
        if dur and dur > rec.max_sec:
            await m.answer(f"Запись длиннее {rec.max_sec} с ({dur} с). Пришлите кусок покороче.")
            return
        if rec.queue_full():
            await m.answer("Сейчас много запросов. Попробуйте через минуту.")
            return
        if not admin:
            deny = rec.limits.check(uid)
            if deny:
                await m.answer(deny)
                return
        if not rec.try_reserve():
            if not admin:
                rec.limits.refund(uid)
            await m.answer("Сейчас много запросов. Попробуйте через минуту.")
            return
        ahead = rec.requests_ahead
        started_text = "🎙 Начинаю распознавать запись."
        accepted_text = "⏳ Запись принята и ждёт своей очереди." if ahead else started_text
        eta = rec.estimate_seconds(dur, f.file_size or 0, ahead=ahead)
        done = False
        reserved = True
        try:
            progress = await m.answer(accepted_text + estimate_text(eta))
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "in")
                await m.bot.download(f, destination=path)
                if not dur:
                    dur = await probe_duration(path)
                    if dur and dur > rec.max_sec:
                        raise AsrError(f"Запись длиннее {rec.max_sec} с. Пришлите кусок покороче.")

                async def announce_start():
                    try:
                        await progress.edit_text(started_text + estimate_text(rec.estimate_seconds(dur)))
                    except TelegramAPIError:
                        log.debug("не удалось обновить статус распознавания", exc_info=True)

                punctuation = bool(storage.settings(uid)["asr_punctuation"])
                reserved = False  # transcribe_file освобождает слот в своём finally
                texts, _ = await rec.transcribe_file(
                    path, reserved=True, punctuation=punctuation, duration_hint=dur,
                    on_start=announce_start if ahead or (eta is None and dur) else None,
                )
            done = True
        except AsrError as e:
            await m.answer(f"⚠️ {html.escape(str(e))}")
            return
        except Exception:
            log.exception("ошибка распознавания")
            await m.answer("⚠️ Не получилось расшифровать эту запись.")
            return
        finally:
            if reserved:
                rec.release_reservation()
            if not done and not admin:
                rec.limits.refund(uid)
        body = "\n".join(texts) or "(речь не распознана)"
        while len(body) > 3900:
            end = body.rfind("\n", 1, 3901)
            if end < 0:
                end = body.rfind(" ", 1, 3901)
            if end < 0:
                end = 3900
            await m.answer(html.escape(body[:end]))
            body = body[end:].lstrip()
        await m.answer(f"{html.escape(body)}\n\n{NOTE}")

    return r
