"""OptionSmith web dashboard — FastAPI + a single self-contained page.

    python -m optionsmith ui --port 8899

Endpoints
    GET  /api/catalogue                 the strategy library
    GET  /api/gateway/status            is live data available right now
    GET  /api/gateway/search?q=         underlying lookup
    GET  /api/gateway/expiries?symbol=  listed expiries for an underlying
    POST /api/advise                    {chain?|live|demo params, view...} -> report
    POST /api/build                     {key, chain...} -> one priced structure
    POST /api/payoff                    {legs, chain...} -> payoff curve points

Live mode routes through the trading Gateway; the dashboard never sees broker
credentials and cannot place an order.
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..advisor import advise, build_named
from ..chain.loaders import _from_dict, from_gateway, synthetic
from ..core.models import Leg
from ..core.payoff import evaluate, payoff_at
from ..gateway.client import GatewayClient, GatewayError
from ..scanner import universe as scan_universe
from ..scanner.engine import ScanConfig
from ..scanner.service import RUNNER
from ..scanner import store as scan_store
from ..strategies.library import CATALOG

STATIC = Path(__file__).resolve().parent / "static"


class ChainSpec(BaseModel):
    """A full chain payload, a live Gateway fetch, or synthetic parameters."""
    chain: dict | None = None
    live: bool = False
    exchange: str = "NSE"
    expiry: str = ""
    strikes: int = 15
    symbol: str = "DEMO"
    spot: float = 1400.0
    dte: int = 30
    lot_size: int = 250
    base_iv: float = 0.28
    wall_call: float | None = None
    wall_put: float | None = None


class AdviseReq(ChainSpec):
    view: str | None = None
    vol: str | None = None
    move: float | None = None
    iv_percentile: float | None = None
    top_n: int = 5
    max_legs: int = 4
    max_loss: float = 25_000.0


class BuildReq(ChainSpec):
    key: str


class ScanReq(BaseModel):
    """What the scanner needs. Everything else has a sane default — the point
    of the screen is one slider, not a form."""
    pop_min: float = 50.0
    pop_max: float = 100.0
    source: str = "live"            # live | synthetic
    strikes: int = 8
    max_loss: float = 25_000.0
    max_legs: int = 4
    top_per_symbol: int = 3
    include_indices: bool = False


class BlacklistReq(BaseModel):
    symbol: str
    blacklisted: bool


class PayoffReq(ChainSpec):
    legs: list[dict]


def _chain(spec: ChainSpec):
    if spec.chain:
        return _from_dict(spec.chain, source="api")
    if spec.live:
        return from_gateway(spec.symbol, exchange=spec.exchange,
                            expiry=spec.expiry or "", count=spec.strikes)
    return synthetic(spec.symbol, spot=spec.spot, dte=spec.dte,
                     lot_size=spec.lot_size, base_iv=spec.base_iv,
                     wall_call=spec.wall_call, wall_put=spec.wall_put)


def _gateway() -> GatewayClient:
    """A fresh client per request.

    Deliberately not a module-level singleton: the token is short-lived and the
    handlers run in FastAPI's threadpool, so a shared client would need its own
    locking around re-login for no measurable gain — a chain fetch is seconds of
    broker I/O, next to which the TCP setup does not register.
    """
    try:
        return GatewayClient()
    except GatewayError as e:
        raise HTTPException(503, str(e))


def create_app() -> FastAPI:
    app = FastAPI(title="OptionSmith", version="1.1.0")

    @app.get("/")
    def root():
        """The scanner IS the product. The single-chain advisor stays reachable
        at /advisor for drilling into one name, but nobody should have to fill
        in a form to find out what the market is offering."""
        return FileResponse(STATIC / "scanner.html")

    @app.get("/advisor")
    def advisor_page():
        return FileResponse(STATIC / "index.html")

    # ---------------- scanner ----------------
    @app.post("/api/scan/start")
    def scan_start(req: ScanReq):
        if req.pop_min > req.pop_max:
            raise HTTPException(400, "pop_min cannot exceed pop_max")
        cfg = ScanConfig(pop_min=req.pop_min, pop_max=req.pop_max,
                         source=req.source, strikes=req.strikes,
                         max_loss=req.max_loss, max_legs=req.max_legs,
                         top_per_symbol=req.top_per_symbol,
                         include_indices=req.include_indices)
        try:
            run_id = RUNNER.start(cfg)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"run_id": run_id, "total": RUNNER.total}

    @app.get("/api/scan/status")
    def scan_status(limit: int = 100):
        return RUNNER.status(limit=limit)

    @app.post("/api/scan/stop")
    def scan_stop():
        RUNNER.stop()
        return {"stopping": True}

    @app.get("/api/scan/universe")
    def scan_universe_list():
        return {"symbols": scan_universe.with_status(),
                "active": len(scan_universe.scan_list())}

    @app.post("/api/scan/blacklist")
    def scan_blacklist(req: BlacklistReq):
        scan_store.set_blacklisted(req.symbol, req.blacklisted)
        return {"symbol": req.symbol.upper(),
                "blacklisted": req.blacklisted,
                "active": len(scan_universe.scan_list())}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "optionsmith"}

    @app.get("/api/catalogue")
    def catalogue():
        return [{"key": r.key, "name": r.name, "views": list(r.views),
                 "family": r.family, "caveat": r.caveat} for r in CATALOG]

    @app.get("/api/gateway/status")
    def gw_status():
        """Whether live mode will work, and if not, why — so the page can say
        'broker not connected' instead of failing on the first chain fetch."""
        try:
            c = GatewayClient()
        except GatewayError as e:
            return {"available": False, "connected": False, "reason": str(e)}
        try:
            with c:
                st = c.status()
            return {"available": True, "connected": bool(st.get("connected")),
                    "reason": "" if st.get("connected")
                              else "gateway reachable but the broker session is "
                                   "down — connect it in the Gateway UI"}
        except GatewayError as e:
            return {"available": False, "connected": False, "reason": str(e)}

    @app.get("/api/gateway/search")
    def gw_search(q: str, exchange: str = ""):
        with _gateway() as c:
            try:
                return {"results": c.search(q, exchange)}
            except GatewayError as e:
                raise HTTPException(502, str(e))

    @app.get("/api/gateway/expiries")
    def gw_expiries(symbol: str, exchange: str = "NSE"):
        with _gateway() as c:
            try:
                return {"expiries": c.expiries(symbol, exchange)}
            except GatewayError as e:
                raise HTTPException(502, str(e))

    @app.post("/api/advise")
    def api_advise(req: AdviseReq):
        with _fail_as_http():
            ch = _chain(req)
            rep = advise(ch, user_direction=req.view, user_volatility=req.vol,
                         user_target_move_pct=req.move,
                         iv_percentile=req.iv_percentile, top_n=req.top_n,
                         max_legs=req.max_legs, max_loss_rupees=req.max_loss)
            out = rep.to_dict()
            out["chain_preview"] = _preview(ch)
            return out

    @app.post("/api/build")
    def api_build(req: BuildReq):
        with _fail_as_http():
            ch = _chain(req)
            res = build_named(ch, req.key)
            d = res.to_dict()
            d["curve"] = _curve(res.legs, ch)
            return d

    @app.post("/api/payoff")
    def api_payoff(req: PayoffReq):
        with _fail_as_http():
            ch = _chain(req)
            legs = [Leg(float(l["strike"]),
                        str(l.get("right", "CE")).upper().startswith("C"),
                        1 if str(l.get("action", "BUY")).upper() == "BUY" else -1,
                        int(l.get("qty", 1))) for l in req.legs]
            res = evaluate(legs, ch, name="custom structure", is_custom=True,
                           snap_strikes=True)   # hand-typed strikes -> nearest listed
            d = res.to_dict()
            d["curve"] = _curve(res.legs, ch)
            return d

    return app


@contextmanager
def _fail_as_http():
    """Map failures to a status the page can act on.

    A data-supply failure is not a bad request: answering 400 for "the broker
    session is down" tells the user to fix their inputs, which cannot help. 502
    keeps the two apart, and the Gateway's own wording is passed through intact
    because it already names the thing to fix.
    """
    try:
        yield
    except HTTPException:
        raise
    except GatewayError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))


def _curve(legs, ch, n: int = 121) -> dict:
    lo, hi = ch.spot * 0.82, ch.spot * 1.18
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return {"x": [round(x, 2) for x in xs],
            "y": [round(payoff_at(legs, x) * ch.lot_size, 1) for x in xs],
            "spot": ch.spot,
            "strikes": sorted({l.strike for l in legs})}


def _preview(ch) -> dict:
    def side(q) -> dict:
        if not q:
            return {}
        return {"ltp": round(q.mid, 2), "oi": q.oi,
                "iv": round(q.iv * 100, 1) if q.iv else None,
                "bid": round(q.bid, 2) or None, "ask": round(q.ask, 2) or None,
                # None, not 0: with no prior snapshot the change is unknown, and
                # a table showing 0% would assert that OI held flat.
                "d_oi": q.d_oi if q.prev_oi else None,
                "d_oi_pct": round(q.d_oi_pct, 1) if q.prev_oi else None,
                "volume": q.volume}

    rows = []
    for k in ch.strikes:
        c, p = ch.get(k, True), ch.get(k, False)
        ce, pe = side(c), side(p)
        rows.append({"strike": k,
                     # flat keys the existing page reads, kept as-is
                     "ce_ltp": ce.get("ltp"), "ce_oi": ce.get("oi"),
                     "ce_iv": ce.get("iv"),
                     "pe_ltp": pe.get("ltp"), "pe_oi": pe.get("oi"),
                     "pe_iv": pe.get("iv"),
                     "ce": ce, "pe": pe})
    booked = sum(1 for q in ch.quotes if q.bid > 0 and q.ask > 0)
    return {"symbol": ch.symbol, "spot": ch.spot, "atm": ch.atm,
            "expiry": ch.expiry.isoformat(), "dte": ch.days_to_expiry,
            "lot_size": ch.lot_size, "source": ch.source,
            "contracts": len(ch.quotes), "booked": booked,
            "liquid": len(ch.liquid()), "rows": rows}


def run(port: int = 8899, host: str = "127.0.0.1") -> None:
    import uvicorn
    print(f"OptionSmith dashboard -> http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
