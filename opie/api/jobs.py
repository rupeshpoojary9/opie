"""SQLite-backed job store for batch + real-time job status.

Batch jobs run in a background thread; each product's result is written as it completes,
so `GET /v1/jobs/{id}` reflects real-time progress. Results persist across restarts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from opie.schemas import ProductIntelligence


@dataclass
class JobItem:
    product_id: str
    image_bytes: bytes


class JobStore:
    def __init__(self, db_path: str | Path = "opie_jobs.sqlite") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT, total INTEGER, done INTEGER,
                created_at REAL, updated_at REAL, error TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS results (
                job_id TEXT, product_id TEXT, result_json TEXT,
                PRIMARY KEY (job_id, product_id))""")

    # --- lifecycle --------------------------------------------------------

    def create(self, total: int) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                      (job_id, "queued", total, 0, now, now, None))
        return job_id

    def _touch(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols}, updated_at=? WHERE id=?",
                      (*fields.values(), time.time(), job_id))

    def add_result(self, job_id: str, product: ProductIntelligence) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)",
                      (job_id, product.product_id, product.model_dump_json()))
            c.execute("UPDATE jobs SET done = done + 1, updated_at=? WHERE id=?",
                      (time.time(), job_id))

    def status(self, job_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            results = c.execute(
                "SELECT product_id, result_json FROM results WHERE job_id=?", (job_id,)
            ).fetchall()
        return {
            "job_id": row["id"],
            "status": row["status"],
            "total": row["total"],
            "done": row["done"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
            "results": [json.loads(r["result_json"]) for r in results],
        }

    # --- background execution --------------------------------------------

    def run_async(self, job_id: str, items: list[JobItem],
                  runner: Callable[[str, bytes], ProductIntelligence]) -> None:
        def worker() -> None:
            self._touch(job_id, status="running")
            try:
                for item in items:
                    product = runner(item.product_id, item.image_bytes)
                    self.add_result(job_id, product)
                self._touch(job_id, status="succeeded")
            except Exception as exc:  # pragma: no cover - defensive
                self._touch(job_id, status="failed", error=str(exc))

        threading.Thread(target=worker, daemon=True).start()
