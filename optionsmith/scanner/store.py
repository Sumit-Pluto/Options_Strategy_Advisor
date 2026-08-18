"""Scanner persistence — SQLite, stdlib only.

Three things outlive a process and therefore need a disk:

  * the BLACKLIST, so a name you switched off stays off across restarts
  * ATM IV HISTORY, which is the only way to ever compute an IV percentile.
    No broker sells it and it cannot be backfilled from a live feed — it is
    built one session at a time, so the recording starts the day the scanner
    does, not the day the feature is wanted.
  * SCAN RESULTS, so the screen survives a refresh and so the scanner's own
    picks can be judged later against what actually happened.

SQLite and not Redis: Redis is a cache, and an eviction loses IV history that
takes a year of trading days to rebuild.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import threading
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "scanner.db"
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS blacklist (
    symbol TEXT PRIMARY KEY,
    added  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- one row per symbol per session. UNIQUE(symbol, day) makes a re-scan on the
-- same day update rather than double-count, so an intraday rescan cannot skew
-- the percentile by stuffing the sample with correlated points.
CREATE TABLE IF NOT EXISTS iv_history (
    symbol  TEXT NOT NULL,
    day     TEXT NOT NULL,
    atm_iv  REAL NOT NULL,
    skew    REAL,
    smile   REAL,
    spot    REAL,
    dte     INTEGER,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS scan_runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started   TEXT NOT NULL,
    finished  TEXT,
    source    TEXT,
    scanned   INTEGER DEFAULT 0,
    matched   INTEGER DEFAULT 0,
    pop_min   REAL,
    pop_max   REAL
);
CREATE TABLE IF NOT EXISTS scan_results (
    run_id  INTEGER NOT NULL,
    symbol  TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES scan_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_results_run ON scan_results(run_id);
"""


def db_path() -> Path:
    return Path(os.environ.get("OPTIONSMITH_DB", str(_DEFAULT)))


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p), timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    # WAL + a real busy timeout: the scan runs in a PROCESS pool, so several
    # workers write IV history at once. The default rollback journal serialises
    # them into "database is locked" instead of queueing.
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    c.execute("PRAGMA synchronous=NORMAL")
    c.executescript(SCHEMA)
    return c


# ── blacklist ──────────────────────────────────────────────────────────
def blacklist() -> set[str]:
    with connect() as c:
        return {r["symbol"] for r in c.execute("SELECT symbol FROM blacklist")}


def set_blacklisted(symbol: str, on: bool) -> None:
    with _LOCK, connect() as c:
        if on:
            c.execute("INSERT OR IGNORE INTO blacklist VALUES (?,?)",
                      (symbol.upper(), dt.date.today().isoformat()))
        else:
            c.execute("DELETE FROM blacklist WHERE symbol=?", (symbol.upper(),))


# ── settings ───────────────────────────────────────────────────────────
def get_setting(key: str, default=None):
    with connect() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def set_setting(key: str, value) -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT INTO settings VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value)))


# ── IV history (the long-lead dependency) ──────────────────────────────
def record_iv(symbol: str, atm_iv: float, *, skew: float | None = None,
              smile: float | None = None, spot: float | None = None,
              dte: int | None = None, day: str | None = None) -> None:
    """Append today's ATM IV. Free during a scan — the number is already computed."""
    if not atm_iv or atm_iv <= 0:
        return
    with _LOCK, connect() as c:
        c.execute("INSERT INTO iv_history VALUES (?,?,?,?,?,?,?) "
                  "ON CONFLICT(symbol, day) DO UPDATE SET "
                  "atm_iv=excluded.atm_iv, skew=excluded.skew, "
                  "smile=excluded.smile, spot=excluded.spot, dte=excluded.dte",
                  (symbol.upper(), day or dt.date.today().isoformat(),
                   float(atm_iv), skew, smile, spot, dte))


def iv_percentile(symbol: str, atm_iv: float,
                  min_sessions: int = 60) -> tuple[float | None, int]:
    """(percentile, sessions) of today's ATM IV in this symbol's own history.

    PERCENTILE, not IV Rank: the rank form is (iv-min)/(max-min), which one
    spike distorts for a year. The percentile is the share of past sessions
    that were LOWER, so a single outlier moves it by one sample.

    Returns (None, n) until `min_sessions` exist — an honest "not yet" beats a
    percentile drawn from three days, which would drive the vol regime and
    therefore every credit-vs-debit decision off noise.
    """
    with connect() as c:
        rows = [r["atm_iv"] for r in c.execute(
            "SELECT atm_iv FROM iv_history WHERE symbol=?", (symbol.upper(),))]
    n = len(rows)
    if n < min_sessions or not atm_iv:
        return None, n
    below = sum(1 for x in rows if x < atm_iv)
    return 100.0 * below / n, n


def iv_coverage() -> dict:
    """How far along the IV history is — the UI shows this so the missing vol
    channel is visible rather than silently absent."""
    with connect() as c:
        r = c.execute("SELECT COUNT(DISTINCT symbol) s, COUNT(*) n, "
                      "MIN(day) f, MAX(day) l FROM iv_history").fetchone()
    return {"symbols": r["s"] or 0, "rows": r["n"] or 0,
            "first": r["f"], "last": r["l"]}


# ── scan runs ──────────────────────────────────────────────────────────
def start_run(source: str, pop_min: float, pop_max: float) -> int:
    with _LOCK, connect() as c:
        cur = c.execute(
            "INSERT INTO scan_runs (started, source, pop_min, pop_max) "
            "VALUES (?,?,?,?)",
            (dt.datetime.now().isoformat(timespec="seconds"), source,
             pop_min, pop_max))
        return int(cur.lastrowid)


def save_result(run_id: int, symbol: str, payload: dict) -> None:
    with _LOCK, connect() as c:
        c.execute("INSERT INTO scan_results VALUES (?,?,?)",
                  (run_id, symbol.upper(), json.dumps(payload)))


def finish_run(run_id: int, scanned: int, matched: int) -> None:
    with _LOCK, connect() as c:
        c.execute("UPDATE scan_runs SET finished=?, scanned=?, matched=? "
                  "WHERE id=?",
                  (dt.datetime.now().isoformat(timespec="seconds"),
                   scanned, matched, run_id))
