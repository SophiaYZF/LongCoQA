from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from longcoqa.utils.hashing import stable_json_hash


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                request_hash TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                response_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        os.chmod(self.path, 0o600)

    @staticmethod
    def request_hash(request: dict[str, Any]) -> str:
        return stable_json_hash(request)

    def get(self, request_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM responses WHERE request_hash = ?", (request_hash,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        request_hash: str,
        provider: str,
        model: str,
        response: dict[str, Any],
    ) -> None:
        payload = json.dumps(response, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO responses(request_hash, provider, model, response_json)
                VALUES (?, ?, ?, ?)
                """,
                (request_hash, provider, model, payload),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
