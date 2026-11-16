"""Роутер: голосовые, аудио и видео -> расшифровка. Подключается в main.py, только если ASR включён и модель есть."""
import asyncio, html, logging, os, tempfile
from aiogram import F, Router
from aiogram.types import Message

from .asr import AsrError, Recognizer
from .storage import Storage

log = logging.getLogger("keleni.asr")
NOTE = "<i>Автоматическая расшифровка может содержать ошибки; проверяйте по записи.</i>"


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
        done = False
        reserved = True
        try:
            await m.bot.send_chat_action(m.chat.id, "typing")
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "in")
                await m.bot.download(f, destination=path)
                punctuation = bool(storage.settings(uid)["asr_punctuation"])
                reserved = False  # transcribe_file освобождает слот в своём finally
                texts, secs = await rec.transcribe_file(path, reserved=True, punctuation=punctuation)
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
