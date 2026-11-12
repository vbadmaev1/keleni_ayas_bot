"""SQLite: file_id загруженных аудио, настройки и статистика пользователей.

file_id — главное: Telegram хранит загруженный файл у себя, и повторная
отправка по file_id мгновенная, без повторной загрузки с диска.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULTS = {"script": "cyr", "lang": "ru", "spoiler": 1}


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        # то же, что в events, но построчным JSON: удобно смотреть и выгружать
        self.jsonl = path.parent / "events.jsonl"
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS files (key TEXT PRIMARY KEY, file_id TEXT);
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY, script TEXT, lang TEXT, spoiler INTEGER,
                first_seen REAL, last_seen REAL);
            CREATE TABLE IF NOT EXISTS events (
                ts REAL, user_id INTEGER, kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS answers (
                ts REAL, user_id INTEGER, mode TEXT, text_id TEXT, n INTEGER, score REAL);
            """)
            self.db.commit()

    # --- file_id -------------------------------------------------------------
    def file_id(self, key: str) -> Optional[str]:
        with self.lock:
            row = self.db.execute("SELECT file_id FROM files WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_file_id(self, key: str, file_id: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?)", (key, file_id))
            self.db.commit()

    def drop_file_id(self, key: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM files WHERE key=?", (key,))
            self.db.commit()

    # --- пользователи --------------------------------------------------------
    def settings(self, user_id: int) -> dict:
        with self.lock:
            row = self.db.execute("SELECT script, lang, spoiler FROM users WHERE user_id=?",
                                  (user_id,)).fetchone()
        if not row:
            return dict(DEFAULTS)
        return {"script": row[0] or DEFAULTS["script"], "lang": row[1] or DEFAULTS["lang"],
                "spoiler": DEFAULTS["spoiler"] if row[2] is None else row[2]}

    def set_setting(self, user_id: int, key: str, value) -> None:
        assert key in DEFAULTS
        self.touch(user_id)
        with self.lock:
            self.db.execute(f"UPDATE users SET {key}=? WHERE user_id=?", (value, user_id))
            self.db.commit()

    def touch(self, user_id: int) -> None:
        now = time.time()
        with self.lock:
            self.db.execute(
                "INSERT INTO users (user_id, script, lang, spoiler, first_seen, last_seen) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET last_seen=excluded.last_seen",
                (user_id, DEFAULTS["script"], DEFAULTS["lang"], DEFAULTS["spoiler"], now, now))
            self.db.commit()

    def log(self, user_id: int, kind: str, payload: str = "") -> None:
        now = time.time()
        with self.lock:
            try:
                with open(self.jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": round(now, 3), "user": user_id, "kind": kind,
                                        "payload": payload}, ensure_ascii=False) + "\n")
            except OSError:
                pass
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)",
                            (now, user_id, kind, payload))
            self.db.commit()

    def answer(self, user_id: int, mode: str, text_id: str, n: int, score: float) -> None:
        with self.lock:
            self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?)",
                            (time.time(), user_id, mode, text_id, n, score))
            self.db.commit()

    def user_stats(self, user_id: int) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT mode, COUNT(*), AVG(score) FROM answers WHERE user_id=? GROUP BY mode",
                (user_id,)).fetchall()
        return {m: (n, avg or 0.0) for m, n, avg in rows}

    def global_stats(self) -> dict:
        with self.lock:
            users = self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            week = self.db.execute("SELECT COUNT(*) FROM users WHERE last_seen>?",
                                   (time.time() - 7 * 86400,)).fetchone()[0]
            kinds = self.db.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind "
                                    "ORDER BY 2 DESC").fetchall()
        return {"users": users, "week": week, "events": kinds}
