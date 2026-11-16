"""Прогон бота без сети: подменённая сессия aiogram записывает все запросы.

    python tests/smoke.py
"""
import asyncio
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import (AnswerCallbackQuery, EditMessageReplyMarkup, EditMessageText, GetFile,
                             SendChatAction, SendMessage, SendVoice)
from aiogram.types import File, Message, Update

from bot.asr import Recognizer
from bot.asr_handlers import build_asr_router
from bot.corpus import Corpus
from bot.handlers import build_router
from bot.practice import dictation_score
from bot.storage import Storage

CHAT = {"id": 42, "type": "private"}
USER = {"id": 42, "is_bot": False, "first_name": "T"}


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.mid = 100

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, (AnswerCallbackQuery, SendChatAction)):
            return True
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="U", file_path="voice.ogg")
        if isinstance(method, (SendMessage, SendVoice, EditMessageText, EditMessageReplyMarkup)):
            self.mid += 1
            msg = {"message_id": self.mid, "date": int(time.time()), "chat": CHAT,
                   "text": getattr(method, "text", None) or "x"}
            if isinstance(method, SendVoice):
                msg["voice"] = {"file_id": f"F{self.mid}", "file_unique_id": f"U{self.mid}",
                                "duration": 1}
            return Message.model_validate(msg, context={"bot": bot})
        return True

    async def stream_content(self, *a, **k):
        yield b""

    async def close(self):
        pass


async def main():
    corpus = Corpus()
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "db.sqlite"
    with sqlite3.connect(db) as legacy:
        legacy.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, script TEXT, lang TEXT, "
                       "spoiler INTEGER, first_seen REAL, last_seen REAL)")
        legacy.execute("INSERT INTO users VALUES (42, 'cyr', 'ru', 1, 0, 0)")
    storage = Storage(db)
    assert storage.settings(42) == {"script": "cyr", "lang": "ru", "spoiler": 1, "asr_punctuation": 0}
    session = FakeSession()
    bot = Bot("1:TEST", session=session, default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher(storage=MemoryStorage())
    class FakeRecognizer(Recognizer):
        def _run(self, src, punctuation=None):
            return (["Кен ирв?"] if punctuation else ["кен ирв"]), 1.0

    rec = FakeRecognizer(tmp)
    dp.include_router(build_asr_router(rec, {42, 43}, storage))
    dp.include_router(build_router(corpus, storage, {42}))
    uid = [0]

    def upd(**kw):
        uid[0] += 1
        return Update.model_validate({"update_id": uid[0], **kw}, context={"bot": bot})

    async def text(t):
        session.calls.clear()
        await dp.feed_update(bot, upd(message={"message_id": uid[0], "date": int(time.time()),
                                                "chat": CHAT, "from": USER, "text": t}))
        return list(session.calls)

    async def click(data):
        session.calls.clear()
        await dp.feed_update(bot, upd(callback_query={
            "id": str(uid[0]), "from": USER, "chat_instance": "c", "data": data,
            "message": {"message_id": 1, "date": int(time.time()), "chat": CHAT, "text": "x"}}))
        return list(session.calls)

    async def voice(user):
        session.calls.clear()
        await dp.feed_update(bot, upd(message={
            "message_id": uid[0], "date": int(time.time()), "chat": CHAT, "from": user,
            "voice": {"file_id": "audio", "file_unique_id": "U", "duration": 1},
        }))
        messages = [call.text for call in session.calls if isinstance(call, SendMessage)]
        assert len(messages) == 2
        assert "Начинаю распознавать запись" in messages[0]
        assert "Примерное время до результата" in messages[0]
        return messages[-1]

    def show(calls, label):
        out = [c for c in calls if isinstance(c, (SendMessage, SendVoice, EditMessageText))]
        assert out, f"{label}: бот ничего не ответил ({[type(c).__name__ for c in calls]})"
        for c in out:
            body = getattr(c, "text", None) or getattr(c, "caption", "") or ""
            kbd = getattr(c, "reply_markup", None)
            btns = []
            if kbd is not None and hasattr(kbd, "inline_keyboard"):
                btns = [b.callback_data for row in kbd.inline_keyboard for b in row]
                for d in btns:
                    assert len(d.encode()) <= 64, d
            print(f"[{label}] {type(c).__name__}: {re.sub(r'\s+', ' ', body)[:150]!r} {btns[:8]}")
        return out

    show(await text("/start"), "start")
    show(await text("📚 Все записи"), "catalog")
    show(await click("g:nar:0"), "genre")
    show(await click("g:nar:5"), "genre p6")
    i = corpus.index["SFS_2008_Witch_flk"]
    show(await click(f"t:{i}"), "card")
    show(await click(f"f:{i}"), "full")
    show(await click(f"s:{i}:1"), "sent1")
    show(await click(f"s:{i}:1"), "sent1 again")    # теперь по file_id
    show(await click(f"sl:{i}:2"), "slow")
    show(await click(f"x:{i}:0"), "fulltext")
    j = corpus.index["LZB_2018_Bio_nar"]
    pages = show(await click(f"x:{j}:3"), "fulltext long p4")
    assert len(pages[0].text) < 4096
    k = corpus.index["SGC_2007_Kalmyk_nar"]
    show(await click(f"t:{k}"), "card unaligned")
    show(await click(f"s:{k}:1"), "sent unaligned")
    na = corpus.index["DBB_2007_Mother_flk"]
    show(await click(f"t:{na}"), "card no audio")
    show(await text("🎲 Случайная запись"), "random")
    show(await text("🎧 Случайное предложение"), "rsent")
    for q in ["волк", "мөрн", "морн", "tea", "бяяҗ", "zzzqqq"]:
        show(await text(q), f"search {q}")
    show(await click("sr:1"), "search p2")
    calls = show(await text("✍️ Диктант"), "dictation")
    show(await text("тер нег"), "dictation answer")
    show(await text("✍️ Диктант"), "dictation2")
    show(await text("/skip"), "skip")
    show(await text("✍️ Диктант"), "dictation3")
    show(await text("🔎 Поиск"), "menu during dictation")
    show(await text("тер"), "search after menu")
    q = show(await text("❓ Угадай перевод"), "quiz")
    qa = [b.callback_data for row in q[-1].reply_markup.inline_keyboard for b in row
          if b.callback_data.startswith("qa:")][0]
    show(await click(qa), "quiz answer")
    settings = show(await text("⚙️ Настройки"), "settings")[0]
    assert any(b.callback_data == "set:asr_punctuation:1" for row in settings.reply_markup.inline_keyboard for b in row)
    enabled = show(await click("set:asr_punctuation:1"), "enable punctuation")[0]
    assert any(b.callback_data == "set:asr_punctuation:0" and b.text.startswith("✅")
               for row in enabled.reply_markup.inline_keyboard for b in row)
    assert (await voice(USER)).startswith("Кен ирв?")
    assert (await voice({**USER, "id": 43})).startswith("кен ирв")
    assert Storage(db).settings(42)["asr_punctuation"] == 1
    show(await click("set:asr_punctuation:0"), "disable punctuation")
    assert (await voice(USER)).startswith("кен ирв")
    with_default = Storage(db, asr_punctuation_default=True)
    assert with_default.settings(42)["asr_punctuation"] == 0
    assert with_default.settings(43)["asr_punctuation"] == 1
    show(await click("set:script:both"), "set script")
    show(await click("set:lang:en"), "set lang")
    show(await click(f"s:{i}:3"), "sent with both/en")
    show(await click(f"er:{i}:3"), "report")
    show(await text("правильно так"), "report text")
    if any(t.id.startswith("catalog_") for t in corpus.texts.values()):
        assert all(corpus.index[key] == index for key, index in Corpus(additional=None).index.items())
        preview = next(t for t in corpus.texts.values() if t.preview_audio and t.has_automatic_transcript)
        p = corpus.index[preview.id]
        card = show(await click(f"t:{p}"), "TV preview card")[0]
        assert "Автоматическая расшифровка. Возможны ошибки." in card.text
        assert preview.source_url in card.text
        assert show(await click(f"f:{p}"), "TV preview")[0].caption.startswith("<b>")
        assert "Автоматическая расшифровка" in show(await click(f"x:{p}:0"), "TV transcript")[0].text
        full = next(t for t in corpus.texts.values() if t.id.startswith("catalog_") and t.full_audio)
        assert show(await click(f"f:{corpus.index[full.id]}"), "additional full recording")
    show(await text("ℹ️ О записях"), "about")
    show(await text("/stats"), "stats")

    sc, diff = dictation_score("Нег үкртә, тер үкрән сааһад", "нег укртя тер укрән")
    print(sc, diff)
    assert abs(sc - 0.8) < 1e-9
    print("OK")


asyncio.run(main())
