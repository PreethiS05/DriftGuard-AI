"""Local SQLite store for analysis summaries (no raw data is stored)."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .pipeline import AnalysisResult
from .reports import summary_dict

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "DRIFTGUARD_HISTORY_DB",
        Path(__file__).resolve().parent.parent / "data" / "history.db",
    )
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    reference_name TEXT NOT NULL,
    current_name TEXT NOT NULL,
    reference_rows INTEGER NOT NULL,
    current_rows INTEGER NOT NULL,
    n_features INTEGER NOT NULL,
    n_drifted INTEGER NOT NULL,
    n_insufficient INTEGER NOT NULL,
    n_quality_issues INTEGER NOT NULL,
    correction TEXT NOT NULL,
    alpha REAL NOT NULL,
    drifted_features TEXT NOT NULL,
    summary_json TEXT NOT NULL
)
"""


class HistoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, result: AnalysisResult) -> None:
        s = result.summary()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.analysis_id,
                    result.timestamp,
                    result.reference_meta.name,
                    result.current_meta.name,
                    result.reference_meta.n_rows,
                    result.current_meta.n_rows,
                    s["n_features_analyzed"],
                    s["n_drifted"],
                    s["n_insufficient"],
                    s["n_quality_issues"],
                    result.config.correction,
                    result.config.alpha,
                    json.dumps(s["drifted_features"]),
                    json.dumps(summary_dict(result)),
                ),
            )

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT analysis_id, timestamp_utc, reference_name, current_name, reference_rows, "
                "current_rows, n_features, n_drifted, n_insufficient, n_quality_issues, correction, "
                "alpha, drifted_features FROM analyses ORDER BY timestamp_utc DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["drifted_features"] = json.loads(d["drifted_features"])
            out.append(d)
        return out

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT summary_json FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def delete(self, analysis_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
        return cur.rowcount > 0

    def clear(self) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("DELETE FROM analyses")
        return cur.rowcount
