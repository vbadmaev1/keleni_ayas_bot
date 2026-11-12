"""Хэндлеры бота.

Колбэки (≤ 64 байт): текст кодируется номером в каталоге (corpus.index),
предложение — номером в тексте.
  g:<genre>:<page>     список текстов жанра
  t:<i>                карточка текста
  f:<i>                запись целиком
  x:<i>:<page>         текст с переводом, постранично
  s:<i>:<n>            предложение
  sl:<i>:<n>           предложение медленно
  r:t | r:s            случайный текст / предложение
  d                    новый диктант
  q                    новый тест;  qa:<i>:<n>:<выбор>:<верный>
  sr:<page>            страница результатов поиска
  set:<ключ>:<знач>    настройки
  er:<i>:<n>           сообщить об ошибке в тексте
"""

import asyncio
import html
import logging
from pathlib import Path
from typing import Dict, List, Optional

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (CallbackQuery, FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message,
                           ReplyKeyboardMarkup)

from . import texts as T
from .corpus import AUDIO, Corpus, Sentence, Text, highlight
from .practice import dictation_score, quiz_options
from .storage import Storage

log = logging.getLogger("keleni.handlers")

PAGE_TEXTS = 8
PAGE_RESULTS = 5
PAGE_CHARS = 3500


class Mode(StatesGroup):
    dictation = State()
    report = State()
    search = State()


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def plural(n: int, one: str, few: str, many: str) -> str:
    k = n % 100
    w = many if 11 <= k <= 14 else one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many
    return f"{n} {w}"


def kb(rows) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=t, callback_data=d) for t, d in row] for row in rows if row])


MENU = {T.BTN_RANDOM, T.BTN_CATALOG, T.BTN_SENTENCE, T.BTN_DICTATION, T.BTN_QUIZ,
        T.BTN_SEARCH, T.BTN_SETTINGS, T.BTN_ABOUT}


def main_menu() -> ReplyKeyboardMarkup:
    b = lambda t: KeyboardButton(text=t)  # noqa: E731
    return ReplyKeyboardMarkup(resize_keyboard=True, is_persistent=True, keyboard=[
        [b(T.BTN_RANDOM), b(T.BTN_CATALOG)],
        [b(T.BTN_SENTENCE), b(T.BTN_DICTATION), b(T.BTN_QUIZ)],
        [b(T.BTN_SEARCH), b(T.BTN_SETTINGS), b(T.BTN_ABOUT)],
    ])


def build_router(corpus: Corpus, storage: Storage, admins: set,
                 cache_dir: Optional[Path] = None) -> Router:
    r = Router()

    @r.message.outer_middleware()
    async def reset_mode(handler, event: Message, data):
        """Кнопка меню или команда (кроме /skip) отменяет диктант и правку."""
        text = event.text or ""
        if text in MENU or (text.startswith("/") and not text.startswith("/skip")):
            state: FSMContext = data.get("state")
            if state is not None:
                await state.clear()
        return await handler(event, data)
    last_search: Dict[int, tuple] = {}     # user_id -> (query, [Sentence])
    slow_lock = asyncio.Lock()

    # ---------------------------------------------------------------- утилиты
    def idx(t: Text) -> int:
        return corpus.index[t.id]

    def by_idx(i: str) -> Text:
        return corpus.texts[corpus.order[int(i)]]

    def kalmyk(s: Sentence, script: str) -> str:
        cyr, lat = html.escape(s.cyr), html.escape(s.lat)
        if script == "lat":
            return lat
        if script == "both":
            return f"{cyr}\n<code>{lat}</code>"
        return cyr

    def translation(s: Sentence, lang: str) -> str:
        if lang == "both":
            return f"{html.escape(s.ru)}\n{html.escape(s.en)}"
        return html.escape(s.en if lang == "en" else s.ru)

    def spoil(text: str, on: bool) -> str:
        return f"<tg-spoiler>{text}</tg-spoiler>" if on and text else text

    async def send_audio_file(msg: Message, path: Path, key: str, caption: str,
                              markup=None, duration: Optional[float] = None):
        """Голосовое по file_id, а если его ещё нет — загрузка с диска."""
        if len(caption) > 1000:
            caption = caption[:990] + "…"
        fid = storage.file_id(key)
        args = dict(caption=caption, reply_markup=markup)
        if duration:
            args["duration"] = int(round(duration))
        if fid:
            try:
                return await msg.answer_voice(fid, **args)
            except TelegramBadRequest:
                storage.drop_file_id(key)
        sent = await msg.answer_voice(FSInputFile(path), **args)
        if sent.voice:
            storage.set_file_id(key, sent.voice.file_id)
        return sent

    # ---------------------------------------------------------------- меню
    @r.message(CommandStart())
    async def start(m: Message, state: FSMContext):
        await state.clear()
        storage.touch(m.from_user.id)
        storage.log(m.from_user.id, "start")
        await m.answer(T.START, reply_markup=main_menu())

    @r.message(Command("about"))
    @r.message(F.text == T.BTN_ABOUT)
    async def about(m: Message):
        n_sent = sum(len(t.sentences) for t in corpus.texts.values())
        hours = sum(t.duration for t in corpus.texts.values()) / 3600
        await m.answer(T.ABOUT.format(texts=len(corpus.texts), sent=n_sent,
                                      hours=f"{hours:.1f}".replace(".", ",")),
                       disable_web_page_preview=True)

    # ---------------------------------------------------------------- каталог
    def catalog_root():
        rows = [[(f"{name} ({n})", f"g:{g}:0")] for g, name, n in corpus.genres()]
        return "Выберите раздел:", kb(rows)

    def genre_page(genre: str, page: int):
        items = corpus.by_genre(genre)
        pages = max(1, -(-len(items) // PAGE_TEXTS))
        page = max(0, min(page, pages - 1))
        rows = []
        for t in items[page * PAGE_TEXTS:(page + 1) * PAGE_TEXTS]:
            mark = "🔊" if t.audio else "📄"
            dur = f" · {fmt_dur(t.duration)}" if t.duration else ""
            rows.append([(f"{mark} {t.title_ru}{dur}", f"t:{idx(t)}")])
        nav = []
        if page > 0:
            nav.append(("◀️", f"g:{genre}:{page - 1}"))
        nav.append((f"{page + 1}/{pages}", "noop"))
        if page < pages - 1:
            nav.append(("▶️", f"g:{genre}:{page + 1}"))
        rows.append(nav)
        rows.append([("⬅️ Разделы", "cat")])
        name = items[0].genre_ru if items else genre
        return f"<b>{name}</b> — {len(items)}. 🔊 есть запись, 📄 только текст.", kb(rows)

    @r.message(Command("list"))
    @r.message(F.text == T.BTN_CATALOG)
    async def catalog(m: Message):
        text, markup = catalog_root()
        await m.answer(text, reply_markup=markup)

    @r.callback_query(F.data == "cat")
    async def cb_catalog(c: CallbackQuery):
        text, markup = catalog_root()
        await c.message.edit_text(text, reply_markup=markup)
        await c.answer()

    @r.callback_query(F.data.startswith("g:"))
    async def cb_genre(c: CallbackQuery):
        _, g, page = c.data.split(":")
        text, markup = genre_page(g, int(page))
        await c.message.edit_text(text, reply_markup=markup)
        await c.answer()

    @r.callback_query(F.data == "noop")
    async def cb_noop(c: CallbackQuery):
        await c.answer()

    def text_card(t: Text):
        lines = [f"<b>{html.escape(t.title_ru)}</b>"
                 + (f" <i>({html.escape(t.title_en)})</i>" if t.title_en else "")]
        who = t.speaker_names
        if who:
            lines.append(f"🗣 {html.escape(who)}")
        meta = [x for x in (t.genre_ru, f"{t.dialect} говор" if t.dialect else "",
                            t.place, t.year) if x]
        lines.append(" · ".join(meta))
        info = plural(len(t.sentences), "предложение", "предложения", "предложений")
        if t.duration:
            info = f"⏱ {fmt_dur(t.duration)} · " + info
        lines.append(info)
        if not t.audio:
            lines.append("📄 Записи нет, только текст.")
        elif not t.aligned:
            lines.append("По предложениям слушать нельзя: в корпусе нет разметки по времени.")
        if t.partial:
            covered = t.sentences[-1].end
            lines.append(f"ℹ️ Текст расшифрован только для первых {fmt_dur(covered)} записи.")
        first = t.sentences[0].cyr if t.sentences else ""
        if first:
            lines.append(f"\n<i>{html.escape(first[:200])}…</i>")
        i = idx(t)
        rows = []
        if t.full_audio:
            rows.append([("▶️ Вся запись", f"f:{i}")])
        if t.aligned:
            rows.append([("🎧 По предложениям", f"s:{i}:1")])
        rows.append([("📝 Текст с переводом", f"x:{i}:0")])
        rows.append([("⬅️ К списку", f"g:{t.genre}:{corpus.by_genre(t.genre).index(t) // PAGE_TEXTS}")])
        return "\n".join(lines), kb(rows)

    @r.callback_query(F.data.startswith("t:"))
    async def cb_text(c: CallbackQuery):
        t = by_idx(c.data.split(":")[1])
        text, markup = text_card(t)
        try:
            await c.message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest:
            await c.message.answer(text, reply_markup=markup)
        await c.answer()

    async def send_full(m: Message, t: Text, user_id: int):
        storage.log(user_id, "full", t.id)
        cap = f"<b>{html.escape(t.title_ru)}</b> · {html.escape(t.speaker_names)}"
        i = idx(t)
        rows = [[("📝 Текст с переводом", f"x:{i}:0")]]
        if t.aligned:
            rows[0].append(("🎧 По предложениям", f"s:{i}:1"))
        await m.bot.send_chat_action(m.chat.id, "upload_voice")
        await send_audio_file(m, t.full_audio, f"full:{t.id}", cap, kb(rows), t.duration)

    @r.callback_query(F.data.startswith("f:"))
    async def cb_full(c: CallbackQuery):
        t = by_idx(c.data.split(":")[1])
        await c.answer("Отправляю…")
        await send_full(c.message, t, c.from_user.id)

    # ---------------------------------------------------------------- текст целиком
    def text_pages(t: Text, script: str, lang: str) -> List[str]:
        pages, cur = [], ""
        for s in t.sentences:
            block = f"<b>{s.n}.</b> {kalmyk(s, script)}\n<i>{translation(s, lang)}</i>\n\n"
            if len(cur) + len(block) > PAGE_CHARS and cur:
                pages.append(cur)
                cur = ""
            cur += block
        if cur:
            pages.append(cur)
        return pages or ["(пусто)"]

    @r.callback_query(F.data.startswith("x:"))
    async def cb_fulltext(c: CallbackQuery):
        _, i, page = c.data.split(":")
        t = by_idx(i)
        st = storage.settings(c.from_user.id)
        pages = text_pages(t, st["script"], st["lang"])
        page = max(0, min(int(page), len(pages) - 1))
        head = f"<b>{html.escape(t.title_ru)}</b>"
        if len(pages) > 1:
            head += f" · стр. {page + 1}/{len(pages)}"
        rows = []
        if page < len(pages) - 1:
            rows.append([("Дальше ▶️", f"x:{i}:{page + 1}")])
        rows.append([("⬅️ К записи", f"t:{i}")])
        await c.message.answer(f"{head}\n\n{pages[page]}", reply_markup=kb(rows))
        await c.answer()

    # ---------------------------------------------------------------- предложения
    def sentence_caption(s: Sentence, t: Text, st: dict, spoiler: Optional[bool] = None) -> str:
        on = bool(st["spoiler"]) if spoiler is None else spoiler
        head = f"<b>{html.escape(t.title_ru)}</b> · {s.n}/{len(t.sentences)}"
        out = f"{head}\n{spoil(kalmyk(s, st['script']), on)}"
        tr = translation(s, st["lang"]).strip()
        if tr:
            out += f"\n\n{spoil('<i>' + tr + '</i>', on)}"
        return out

    def sentence_kb(t: Text, n: int):
        i, total = idx(t), len(t.sentences)
        nav = []
        if n > 1:
            nav.append(("◀️", f"s:{i}:{n - 1}"))
        nav.append(("🔁", f"s:{i}:{n}"))
        if n < total:
            nav.append(("▶️", f"s:{i}:{n + 1}"))
        return kb([nav,
                   [("🐢 0.75×", f"sl:{i}:{n}"), ("⚠️ Ошибка", f"er:{i}:{n}")],
                   [("📝 Весь текст", f"x:{i}:0"), ("⬅️ К записи", f"t:{i}")]])

    async def send_sentence(m: Message, t: Text, n: int, user_id: int, slow: bool = False):
        s = t.sentences[n - 1]
        st = storage.settings(user_id)
        clip = t.clip(n)
        cap = sentence_caption(s, t, st)
        if not clip:
            await m.answer(cap + "\n\n(для этого предложения нет отдельного аудио)",
                           reply_markup=sentence_kb(t, n))
            return
        storage.log(user_id, "slow" if slow else "sentence", f"{t.id}:{n}")
        if slow:
            clip = await slow_version(clip)
            if clip is None:
                await m.answer("Не получилось замедлить запись 😔")
                return
            cap = "🐢 " + cap
        key = f"{'slow' if slow else 'sent'}:{t.id}:{n}"
        await send_audio_file(m, clip, key, cap, sentence_kb(t, n),
                              s.duration / (0.75 if slow else 1))

    async def slow_version(clip: Path) -> Optional[Path]:
        out = (cache_dir or AUDIO) / "slow" / clip.parent.name / clip.name
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        async with slow_lock:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-v", "error", "-i", str(clip), "-af", "atempo=0.75",
                "-c:a", "libopus", "-b:a", "32k", "-application", "voip", str(out),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, err = await proc.communicate()
        if proc.returncode:
            log.error("ffmpeg slow failed: %s", err.decode()[-300:])
            return None
        return out

    @r.callback_query(F.data.startswith("s:"))
    async def cb_sentence(c: CallbackQuery):
        _, i, n = c.data.split(":")
        await c.answer()
        await send_sentence(c.message, by_idx(i), int(n), c.from_user.id)

    @r.callback_query(F.data.startswith("sl:"))
    async def cb_slow(c: CallbackQuery):
        _, i, n = c.data.split(":")
        await c.answer("Замедляю…")
        await send_sentence(c.message, by_idx(i), int(n), c.from_user.id, slow=True)

    # ---------------------------------------------------------------- случайное
    @r.message(Command("random"))
    @r.message(F.text == T.BTN_RANDOM)
    async def random_text(m: Message):
        t = corpus.random_text()
        text, markup = text_card(t)
        await m.answer(text, reply_markup=markup)
        await send_full(m, t, m.from_user.id)

    @r.message(Command("sentence"))
    @r.message(F.text == T.BTN_SENTENCE)
    async def random_sentence(m: Message):
        s = corpus.random_sentence()
        await send_sentence(m, corpus.texts[s.text_id], s.n, m.from_user.id)

    # ---------------------------------------------------------------- диктант
    async def new_dictation(m: Message, state: FSMContext, user_id: int):
        s = corpus.random_sentence(max_len=6.0, min_words=3, max_words=8)
        t = corpus.texts[s.text_id]
        await state.set_state(Mode.dictation)
        await state.update_data(text_id=t.id, n=s.n)
        storage.log(user_id, "dictation", f"{t.id}:{s.n}")
        i = idx(t)
        await m.answer(T.DICTATION_TASK)
        await send_audio_file(m, t.clip(s.n), f"sent:{t.id}:{s.n}", "",
                              kb([[("🐢 Медленнее", f"sl:{i}:{s.n}")]]), s.duration)

    @r.message(Command("dictation"))
    @r.message(F.text == T.BTN_DICTATION)
    async def dictation(m: Message, state: FSMContext):
        await new_dictation(m, state, m.from_user.id)

    @r.callback_query(F.data == "d")
    async def cb_dictation(c: CallbackQuery, state: FSMContext):
        await c.answer()
        await new_dictation(c.message, state, c.from_user.id)

    @r.message(StateFilter(Mode.dictation), Command("skip"))
    async def dictation_skip(m: Message, state: FSMContext):
        data = await state.get_data()
        await state.clear()
        s = corpus.sentence(data["text_id"], data["n"])
        t = corpus.texts[s.text_id]
        st = storage.settings(m.from_user.id)
        await m.answer(f"Правильно:\n{html.escape(s.cyr)}\n<i>{translation(s, st['lang'])}</i>",
                       reply_markup=kb([[("✍️ Ещё", "d"), ("📖 Контекст", f"s:{idx(t)}:{s.n}")]]))

    # ---------------------------------------------------------------- тест
    async def new_quiz(m: Message, user_id: int):
        s = corpus.random_sentence(max_len=12.0, min_words=3)
        t = corpus.texts[s.text_id]
        st = storage.settings(user_id)
        lang = "en" if st["lang"] == "en" else "ru"
        opts, right = quiz_options(corpus, s, lang)
        i = idx(t)
        storage.log(user_id, "quiz", f"{t.id}:{s.n}")
        body = "❓ <b>Что здесь говорят?</b>\n\n" + "\n".join(
            f"{k + 1}. {html.escape(o)}" for k, o in enumerate(opts))
        await send_audio_file(m, t.clip(s.n), f"sent:{t.id}:{s.n}", "", None, s.duration)
        await m.answer(body, reply_markup=kb([
            [(str(k + 1), f"qa:{i}:{s.n}:{k}:{right}") for k in range(len(opts))],
            [("🐢 Медленнее", f"sl:{i}:{s.n}")]]))

    @r.message(Command("quiz"))
    @r.message(F.text == T.BTN_QUIZ)
    async def quiz(m: Message):
        await new_quiz(m, m.from_user.id)

    @r.callback_query(F.data == "q")
    async def cb_quiz(c: CallbackQuery):
        await c.answer()
        await new_quiz(c.message, c.from_user.id)

    @r.callback_query(F.data.startswith("qa:"))
    async def cb_quiz_answer(c: CallbackQuery):
        _, i, n, pick, right = c.data.split(":")
        t = by_idx(i)
        s = t.sentences[int(n) - 1]
        ok = pick == right
        storage.answer(c.from_user.id, "quiz", t.id, int(n), 1.0 if ok else 0.0)
        verdict = "✅ Верно!" if ok else f"❌ Нет, правильный ответ — {int(right) + 1}."
        await c.answer("Верно!" if ok else "Мимо")
        try:
            await c.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
        await c.message.answer(
            f"{verdict}\n\n{html.escape(s.cyr)}\n<i>{html.escape(s.ru)}</i>",
            reply_markup=kb([[("❓ Ещё", "q"), ("📖 Контекст", f"s:{i}:{n}")]]))

    # ---------------------------------------------------------------- поиск
    @r.message(Command("search"))
    @r.message(F.text == T.BTN_SEARCH)
    async def search_prompt(m: Message, state: FSMContext):
        await state.set_state(Mode.search)
        await m.answer(T.SEARCH_PROMPT)

    def mark(text: str, q: str) -> str:
        a, b = highlight(text, q)
        if a < 0:
            return html.escape(text)
        return html.escape(text[:a]) + "<b>" + html.escape(text[a:b]) + "</b>" + html.escape(text[b:])

    def results_page(user_id: int, page: int):
        q, res = last_search[user_id]
        st = storage.settings(user_id)
        pages = max(1, -(-len(res) // PAGE_RESULTS))
        page = max(0, min(page, pages - 1))
        chunk = res[page * PAGE_RESULTS:(page + 1) * PAGE_RESULTS]
        more = "+" if len(res) >= 200 else ""
        lines = [f"🔎 «{html.escape(q)}» — найдено {len(res)}{more}\n"]
        buttons = []
        for k, s in enumerate(chunk, page * PAGE_RESULTS + 1):
            t = corpus.texts[s.text_id]
            kal = mark(s.lat, q) if st["script"] == "lat" else mark(s.cyr, q)
            tr = mark(s.en if st["lang"] == "en" else s.ru, q)
            icon = "🔊" if t.clip(s.n) else "📄"
            lines.append(f"<b>{k}.</b> {icon} <i>{html.escape(t.title_ru)}</i>, #{s.n}\n{kal}\n— {tr}\n")
            buttons.append((f"{icon} {k}", f"s:{idx(t)}:{s.n}"))
        nav = []
        if page > 0:
            nav.append(("◀️", f"sr:{page - 1}"))
        if pages > 1:
            nav.append((f"{page + 1}/{pages}", "noop"))
        if page < pages - 1:
            nav.append(("▶️", f"sr:{page + 1}"))
        return "\n".join(lines), kb([buttons, nav])

    async def do_search(m: Message, q: str):
        q = q.strip()[:100]
        res = corpus.search(q)
        storage.log(m.from_user.id, "search", q)
        if not res:
            await m.answer(T.NO_RESULTS.format(q=html.escape(q)))
            return
        last_search[m.from_user.id] = (q, res)
        text, markup = results_page(m.from_user.id, 0)
        await m.answer(text, reply_markup=markup)

    @r.callback_query(F.data.startswith("sr:"))
    async def cb_results(c: CallbackQuery):
        if c.from_user.id not in last_search:
            await c.answer("Поиск устарел, повторите запрос", show_alert=True)
            return
        text, markup = results_page(c.from_user.id, int(c.data.split(":")[1]))
        try:
            await c.message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest:
            pass
        await c.answer()

    # ---------------------------------------------------------------- ошибки в тексте
    @r.callback_query(F.data.startswith("er:"))
    async def cb_report(c: CallbackQuery, state: FSMContext):
        _, i, n = c.data.split(":")
        t = by_idx(i)
        storage.log(c.from_user.id, "report", f"{t.id}:{n}")
        await state.set_state(Mode.report)
        await state.update_data(text_id=t.id, n=int(n))
        await c.answer()
        await c.message.answer(T.REPORT_ASK)

    @r.message(StateFilter(Mode.report), Command("skip"))
    async def report_skip(m: Message, state: FSMContext):
        await state.clear()
        await m.answer(T.REPORT_THANKS)

    @r.message(StateFilter(Mode.report), F.text, ~F.text.startswith("/"))
    async def report_text(m: Message, state: FSMContext):
        data = await state.get_data()
        await state.clear()
        storage.log(m.from_user.id, "correction", f"{data['text_id']}:{data['n']}\t{m.text}")
        await m.answer(T.REPORT_THANKS)
        for a in admins:
            try:
                await m.bot.send_message(
                    a, f"⚠️ Правка {data['text_id']} #{data['n']} от {m.from_user.id}:\n"
                       f"{html.escape(m.text)}")
            except Exception:
                pass

    # ---------------------------------------------------------------- настройки
    SCRIPTS = {"cyr": "Кириллица", "lat": "Латиница (транскрипция)", "both": "Обе"}
    LANGS = {"ru": "Русский", "en": "English", "both": "Оба"}

    def settings_view(user_id: int):
        st = storage.settings(user_id)
        rows = [[(("• " if st["script"] == k else "") + v, f"set:script:{k}")
                 for k, v in SCRIPTS.items()],
                [(("• " if st["lang"] == k else "") + v, f"set:lang:{k}") for k, v in LANGS.items()],
                [(("✅" if st["spoiler"] else "⬜") + " Прятать текст под спойлер",
                  f"set:spoiler:{0 if st['spoiler'] else 1}")]]
        stats = storage.user_stats(user_id)
        q = stats.get("quiz")
        extra = ""
        if q:
            extra = f"\n\nТест: {q[0]} ответов, верных {q[1] * 100:.0f}%."
        d = stats.get("dictation")
        if d:
            extra += f"\nДиктант: {d[0]} раз, в среднем {d[1] * 100:.0f}% слов."
        text = ("⚙️ <b>Настройки</b>\n\nПервая строка — как показывать калмыцкий текст, "
                "вторая — язык перевода." + extra)
        return text, kb(rows)

    @r.message(Command("settings"))
    @r.message(F.text == T.BTN_SETTINGS)
    async def settings(m: Message):
        storage.touch(m.from_user.id)
        text, markup = settings_view(m.from_user.id)
        await m.answer(text, reply_markup=markup)

    @r.callback_query(F.data.startswith("set:"))
    async def cb_set(c: CallbackQuery):
        _, key, val = c.data.split(":")
        storage.set_setting(c.from_user.id, key, int(val) if key == "spoiler" else val)
        text, markup = settings_view(c.from_user.id)
        try:
            await c.message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest:
            pass
        await c.answer("Сохранено")

    # ---------------------------------------------------------------- админ
    @r.message(Command("stats"))
    async def stats(m: Message):
        if m.from_user.id not in admins:
            return
        s = storage.global_stats()
        ev = "\n".join(f"{k}: {n}" for k, n in s["events"])
        await m.answer(f"Пользователей: {s['users']} (за неделю {s['week']})\n\n{ev}")

    # ---------------------------------------------------------------- свободный текст
    @r.message(StateFilter(Mode.dictation), F.text, ~F.text.startswith("/"))
    async def dictation_answer(m: Message, state: FSMContext):
        data = await state.get_data()
        await state.clear()
        s = corpus.sentence(data["text_id"], data["n"])
        t = corpus.texts[s.text_id]
        score, diff = dictation_score(s.cyr, m.text)
        storage.answer(m.from_user.id, "dictation", t.id, s.n, score)
        st = storage.settings(m.from_user.id)
        emoji = "🎉" if score >= 0.99 else "👍" if score >= 0.6 else "🙂"
        await m.answer(
            f"{emoji} Верно {score * 100:.0f}% слов.\n\n{diff}\n\n<i>{translation(s, st['lang'])}</i>\n\n"
            "<s>зачёркнуто</s> — ваше, <b>жирным</b> — как в записи, <u>подчёркнуто</u> — пропущено.",
            reply_markup=kb([[("✍️ Ещё", "d"), ("📖 Контекст", f"s:{idx(t)}:{s.n}")]]))

    @r.message(F.text, ~F.text.startswith("/"))
    async def free_text(m: Message, state: FSMContext):
        await state.clear()
        await do_search(m, m.text)

    return r
