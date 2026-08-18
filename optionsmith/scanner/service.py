"""Background scan runner — start, watch, stop.

A universe scan is slow for a reason that no amount of local optimisation
fixes: the broker has no bulk-quote endpoint, so a chain costs one round-trip
per leg. The runner therefore streams — every symbol's result is published the
moment it lands, so the screen fills progressively instead of showing nothing
for minutes and then everything.

Cancellation is cooperative and checked between symbols. Killing a scan
mid-flight must not leave a half-written run in the database, so the run is
closed out on every exit path.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import store, universe
from .engine import ScanConfig, scan_symbol

# The Gateway serialises per-leg broker calls, so more threads mostly buys
# queueing. Kept low for live and lifted for synthetic, where nothing is shared.
LIVE_WORKERS = 4
SYNTHETIC_WORKERS = 8


class ScanRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self.run_id: int | None = None
        self.cfg: ScanConfig | None = None
        self.total = 0
        self.scanned = 0
        self.matched = 0
        self.current = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.results: list[dict] = []
        self.errors: list[dict] = []

    # ---------------- control ----------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: ScanConfig) -> int:
        with self._lock:
            if self.running:
                raise RuntimeError("a scan is already running")
            self._cancel.clear()
            self._reset()
            self.cfg = cfg
            symbols = universe.scan_list(cfg.include_indices)
            self.total = len(symbols)
            self.started_at = time.time()
            self.run_id = store.start_run(cfg.source, cfg.pop_min, cfg.pop_max)
            self._thread = threading.Thread(
                target=self._run, args=(symbols, cfg), daemon=True,
                name="optionsmith-scan")
            self._thread.start()
            return self.run_id

    def stop(self) -> None:
        self._cancel.set()

    # ---------------- worker ----------------
    def _run(self, symbols: list[str], cfg: ScanConfig) -> None:
        workers = SYNTHETIC_WORKERS if cfg.source == "synthetic" else LIVE_WORKERS
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(scan_symbol, s, cfg): s for s in symbols}
                for fut in as_completed(futures):
                    if self._cancel.is_set():
                        for f in futures:
                            f.cancel()
                        break
                    sym = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:                       # noqa: BLE001
                        self.scanned += 1
                        self.errors.append({"symbol": sym, "error": str(e)})
                        continue
                    self._publish(res)
        finally:
            self.finished_at = time.time()
            if self.run_id is not None:
                store.finish_run(self.run_id, self.scanned, self.matched)

    def _publish(self, res) -> None:
        self.scanned += 1
        self.current = res.symbol
        if not res.ok:
            self.errors.append({"symbol": res.symbol, "error": res.error})
            return
        for opp in res.opportunities:
            opp["iv_percentile"] = res.iv_percentile
            opp["iv_sessions"] = res.iv_sessions
            opp["smile_spread"] = res.smile_spread
            self.results.append(opp)
            if self.run_id is not None:
                store.save_result(self.run_id, res.symbol, opp)
        self.matched += len(res.opportunities)
        # best first, by the EV-free screen score
        self.results.sort(key=lambda x: -x.get("screen_score", 0.0))

    # ---------------- read ----------------
    def status(self, limit: int = 100) -> dict:
        done = self.total and self.scanned >= self.total
        elapsed = ((self.finished_at or time.time()) - self.started_at
                   if self.started_at else 0.0)
        rate = (self.scanned / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.scanned)
        return {
            "running": self.running,
            "run_id": self.run_id,
            "total": self.total,
            "scanned": self.scanned,
            "matched": self.matched,
            "current": self.current,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(remaining / rate, 1) if rate > 0 and self.running else None,
            "done": bool(done and not self.running),
            "cancelled": self._cancel.is_set(),
            "results": self.results[:limit],
            "errors": self.errors[:25],
            "error_count": len(self.errors),
            "config": self.cfg.to_dict() if self.cfg else None,
            "iv_coverage": store.iv_coverage(),
        }


RUNNER = ScanRunner()
