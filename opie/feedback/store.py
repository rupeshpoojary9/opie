"""SQLite capture store: every extraction run, field observation, review-queue entry,
and correction is persisted so the flywheel is auditable and resumable."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from opie.feedback.types import CorrectionCase, FieldObservation


class FeedbackStore:
    def __init__(self, db_path: str | Path = "opie_feedback.sqlite") -> None:
        self.db_path = str(db_path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, round_index INTEGER, mode TEXT,
                product_count INTEGER, created_at REAL);
            CREATE TABLE IF NOT EXISTS observations (
                run_id TEXT, product_id TEXT, kind TEXT, name TEXT,
                extracted_value TEXT, raw_confidence REAL, calibrated_confidence REAL,
                corrected INTEGER, validator_flagged INTEGER, queued INTEGER,
                gt_value TEXT, is_correct INTEGER);
            CREATE TABLE IF NOT EXISTS corrections (
                round_index INTEGER, product_id TEXT, kind TEXT, name TEXT,
                extracted_value TEXT, correct_value TEXT);
            """)

    def start_run(self, run_id: str, round_index: int, mode: str, product_count: int) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                      (run_id, round_index, mode, product_count, time.time()))

    def save_observations(self, run_id: str, obs: list[FieldObservation]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, o.product_id, o.kind, o.name, _s(o.extracted_value), o.raw_confidence,
                  o.calibrated_confidence, int(o.corrected), int(o.validator_flagged),
                  int(o.queued), _s(o.gt_value), _b(o.is_correct)) for o in obs])

    def save_corrections(self, cases: list[CorrectionCase]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT INTO corrections VALUES (?,?,?,?,?,?)",
                [(x.round_index, x.product_id, x.kind, x.name, _s(x.extracted_value),
                  _s(x.correct_value)) for x in cases])

    def review_queue(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT product_id, kind, name, extracted_value, calibrated_confidence "
                "FROM observations WHERE run_id=? AND queued=1", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict:
        with self._conn() as c:
            runs = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            corr = c.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        return {"runs": runs, "observations": obs, "corrections": corr}


def _s(v) -> Optional[str]:
    return None if v is None else str(v)


def _b(v) -> Optional[int]:
    return None if v is None else int(bool(v))
