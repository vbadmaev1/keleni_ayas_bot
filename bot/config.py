"""Конфигурация. Всё читается из переменных окружения (как в todorxoi-bot):

  BOT_TOKEN   токен бота (обязательно)
  ADMIN_IDS   id админов через запятую: /stats и правки текста
  DATA_DIR    папка для базы, логов и кеша замедленных записей (по умолчанию ./state);
              на хостинге должна лежать на постоянном диске, иначе всё
              пропадёт при пересборке
  LOG_LEVEL   INFO по умолчанию
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:  # .env нужен только для локального запуска
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    token: str
    admins: set
    data_dir: Path
    log_level: str

    @classmethod
    def load(cls) -> "Config":
        token = os.getenv("BOT_TOKEN", "").strip()
        if not token:
            raise SystemExit("BOT_TOKEN не задан")
        admins = set()
        for piece in os.getenv("ADMIN_IDS", "").replace(";", ",").split(","):
            piece = piece.strip()
            if piece.lstrip("-").isdigit():
                admins.add(int(piece))
        data_dir = Path(os.getenv("DATA_DIR") or ROOT / "state").expanduser().resolve()
        return cls(token=token, admins=admins, data_dir=data_dir,
                   log_level=os.getenv("LOG_LEVEL", "INFO").upper())
