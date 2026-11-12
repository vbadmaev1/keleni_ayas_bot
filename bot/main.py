"""Точка входа:  python -m bot.main  (long polling)."""

import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from . import texts
from .config import Config
from .corpus import Corpus
from .handlers import build_router
from .storage import Storage


async def run() -> None:
    cfg = Config.load()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    # в stdout (консоль хостинга) и в файл DATA_DIR/bot.log (5 файлов по 2 МБ)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  RotatingFileHandler(cfg.data_dir / "bot.log", maxBytes=2_000_000,
                                      backupCount=5, encoding="utf-8")])
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    corpus = Corpus()
    storage = Storage(cfg.data_dir / "bot.sqlite")
    logging.getLogger("keleni").info("корпус: %d текстов", len(corpus.texts))

    bot = Bot(cfg.token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(build_router(corpus, storage, cfg.admins, cfg.data_dir))
    await bot.set_my_commands([BotCommand(command=k, description=v)
                               for k, v in texts.COMMANDS.items()])
    me = await bot.get_me()
    logging.getLogger("keleni").info("бот запущен: @%s, данные в %s", me.username, cfg.data_dir)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run())
