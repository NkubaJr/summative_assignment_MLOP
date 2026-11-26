import datetime
import os
import sqlite3
from contextlib import contextmanager

from src import config


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                used_in_retrain INTEGER NOT NULL DEFAULT 0
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS retrain_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                n_new_images INTEGER NOT NULL DEFAULT 0,
                accuracy REAL,
                f1_macro REAL,
                precision_macro REAL,
                recall_macro REAL,
                notes TEXT
            )"""
        )


def add_upload(filename, label, path):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO uploads (filename, label, path, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, label, path, _now()),
        )


def pending_uploads():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM uploads WHERE used_in_retrain = 0").fetchall()
        return [dict(r) for r in rows]


def count_pending():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads WHERE used_in_retrain = 0").fetchone()[0]


def pending_by_label():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM uploads WHERE used_in_retrain = 0 GROUP BY label"
        ).fetchall()
        return {r["label"]: r["n"] for r in rows}


def total_uploads():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def mark_uploads_used(ids):
    if not ids:
        return
    with get_conn() as conn:
        conn.executemany("UPDATE uploads SET used_in_retrain = 1 WHERE id = ?", [(i,) for i in ids])


def running_run(max_age_minutes=60):
    # A run counts as active only if it started recently; older ones are treated as interrupted
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM retrain_runs WHERE status = 'running'").fetchall()
    now = datetime.datetime.now(datetime.timezone.utc)
    for r in rows:
        started = datetime.datetime.fromisoformat(r["started_at"])
        if (now - started).total_seconds() < max_age_minutes * 60:
            return dict(r)
    return None


def start_run(n_new_images):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs (started_at, status, n_new_images) VALUES (?, 'running', ?)",
            (_now(), n_new_images),
        )
        return cur.lastrowid


def finish_run(run_id, status, metrics=None, notes=""):
    metrics = metrics or {}
    with get_conn() as conn:
        conn.execute(
            """UPDATE retrain_runs SET finished_at = ?, status = ?, accuracy = ?, f1_macro = ?,
               precision_macro = ?, recall_macro = ?, notes = ? WHERE id = ?""",
            (
                _now(), status, metrics.get("accuracy"), metrics.get("f1_macro"),
                metrics.get("precision_macro"), metrics.get("recall_macro"), notes, run_id,
            ),
        )


def list_runs(limit=10):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
