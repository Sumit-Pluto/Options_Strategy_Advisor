"""Chain loaders — JSON, CSV, synthetic, and live (Dhan / NSE).

The advisor is source-agnostic: everything downstream consumes a `Chain`.
`synthetic()` means the whole module is testable with no broker, no login
and no network.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

from ..core.mathx import bs_price, implied_vol
from ..core.models import Chain, OptionQuote


# ── file loaders ───────────────────────────────────────────────────────
def from_json(path: str | Path) -> Chain:
    """JSON schema:
    {"symbol","spot","expiry":"YYYY-MM-DD","lot_size","asof":"YYYY-MM-DD",
     "quotes":[{"strike","right":"CE|PE","ltp","bid","ask","oi","prev_oi",
                "volume","iv"}]}
    """
    d = json.loads(Path(path).read_text())
    return _from_dict(d, source=f"json:{Path(path).name}")


def _from_dict(d: dict, source: str) -> Chain:
    quotes = []
    for q in d["quotes"]:
        iv = q.get("iv")
        if iv is not None and iv > 1.5:       # given in percent
            iv = iv / 100.0
        quotes.append(OptionQuote(
            strike=float(q["strike"]),
            is_call=str(q.get("right", q.get("type", "CE"))).upper().startswith("C"),
            ltp=float(q.get("ltp", q.get("last", 0)) or 0),
            bid=float(q.get("bid", 0) or 0), ask=float(q.get("ask", 0) or 0),
            oi=int(q.get("oi", 0) or 0), prev_oi=int(q.get("prev_oi", 0) or 0),
            volume=int(q.get("volume", 0) or 0), iv=iv))
    asof = d.get("asof")
    return Chain(symbol=d["symbol"], spot=float(d["spot"]),
                 expiry=dt.date.fromisoformat(d["expiry"]),
                 lot_size=int(d.get("lot_size", 1)), quotes=quotes,
                 asof=dt.date.fromisoformat(asof) if asof else dt.date.today(),
                 source=source)


def from_csv(path: str | Path, symbol: str, spot: float, expiry: str,
             lot_size: int, asof: str | None = None) -> Chain:
    """CSV columns: strike,right,ltp[,bid,ask,oi,prev_oi,volume,iv]"""
    quotes = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            iv = row.get("iv")
            iv = float(iv) if iv not in (None, "") else None
            if iv is not None and iv > 1.5:
                iv /= 100.0
            f = lambda k, d=0: float(row.get(k) or d)          # noqa: E731
            quotes.append(OptionQuote(
                strike=f("strike"),
                is_call=row["right"].upper().startswith("C"),
                ltp=f("ltp"), bid=f("bid"), ask=f("ask"),
                oi=int(f("oi")), prev_oi=int(f("prev_oi")),
                volume=int(f("volume")), iv=iv))
    return Chain(symbol=symbol, spot=spot, expiry=dt.date.fromisoformat(expiry),
                 lot_size=lot_size, quotes=quotes,
                 asof=dt.date.fromisoformat(asof) if asof else dt.date.today(),
                 source=f"csv:{Path(path).name}")


# ── synthetic (no network — the module's test harness) ─────────────────
def synthetic(symbol: str = "DEMO", spot: float = 1000.0, dte: int = 30,
              lot_size: int = 250, n_strikes: int = 21, step: float | None = None,
              base_iv: float = 0.28, skew: float = 0.08, smile: float = 0.35,
              wall_call: float | None = None, wall_put: float | None = None,
              seed: int = 7) -> Chain:
    """A realistic chain: BS-fair prices, a volatility smile with put skew,
    a bid/ask spread that widens for OTM legs, and OI shaped like a real
    chain (peaks at round strikes; optional planted walls)."""
    rnd = random.Random(seed)
    step = step or max(round(spot * 0.01 / 5) * 5, 2.5)
    atm = round(spot / step) * step
    t = max(dte, 1) / 365.0
    quotes: list[OptionQuote] = []
    half = n_strikes // 2
    for i in range(-half, half + 1):
        k = atm + i * step
        if k <= 0:
            continue
        m = math.log(k / spot)
        iv = base_iv + smile * m * m - skew * m          # put skew + smile
        iv = max(0.06, iv)
        for is_call in (True, False):
            px = bs_price(is_call, spot, k, t, iv)
            if px < 0.05:
                continue
            spread = max(0.05, px * (0.01 + 0.04 * min(abs(i) / max(half, 1), 1)))
            bid, ask = max(0.05, px - spread / 2), px + spread / 2
            # Real chains do NOT peak at ATM: call OI builds ABOVE spot
            # (resistance/covered calls), put OI BELOW (protection/puts sold
            # at support). Centre each side's hump accordingly.
            centre = 2.0 if is_call else -2.0
            dist = abs(i - centre)
            base_oi = int(4000 * math.exp(-0.06 * dist * dist) + 200)
            if wall_call and is_call and abs(k - wall_call) < 1e-6:
                base_oi *= 4
            if wall_put and not is_call and abs(k - wall_put) < 1e-6:
                base_oi *= 4
            oi = int(base_oi * rnd.uniform(0.75, 1.25))
            prev = int(oi * rnd.uniform(0.75, 1.1))
            quotes.append(OptionQuote(
                strike=k, is_call=is_call, ltp=round(px, 2),
                bid=round(bid, 2), ask=round(ask, 2), oi=oi, prev_oi=prev,
                volume=int(oi * rnd.uniform(0.1, 0.6)), iv=iv))
    return Chain(symbol=symbol, spot=spot,
                 expiry=dt.date.today() + dt.timedelta(days=dte),
                 lot_size=lot_size, quotes=quotes, asof=dt.date.today(),
                 source="synthetic")


# ── live: Dhan option chain ────────────────────────────────────────────
def _env(path: str | Path | None = None) -> dict:
    """Read a .env for DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN."""
    cands = [Path(path)] if path else []
    cands += [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for fp in cands:
        try:
            out = {}
            for ln in fp.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    out[k.strip()] = v.strip()
            if out:
                return out
        except Exception:
            continue
    return dict(os.environ)


def from_dhan(security_id: int, symbol: str, expiry: str, lot_size: int,
              segment: str = "NSE_FNO", env_path: str | None = None) -> Chain:
    """Live option chain via DhanHQ v2 `/optionchain`.

    Requires DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN. Note Dhan rate-limits this
    endpoint hard (≈1 request / 3 s) — the caller should cache.
    """
    import requests                       # optional dependency, only for live
    env = _env(env_path)
    hdrs = {"access-token": env["DHAN_ACCESS_TOKEN"],
            "client-id": env["DHAN_CLIENT_ID"],
            "Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post("https://api.dhan.co/v2/optionchain", headers=hdrs, json={
        "UnderlyingScrip": int(security_id), "UnderlyingSeg": segment,
        "Expiry": expiry}, timeout=30)
    r.raise_for_status()
    js = r.json()
    data = js.get("data", js)
    spot = float(data.get("last_price") or data.get("ltp") or 0)
    oc = data.get("oc") or data.get("optionChain") or {}
    quotes = []
    for k_str, node in oc.items():
        k = float(k_str)
        for key, is_call in (("ce", True), ("pe", False)):
            leg = node.get(key) or {}
            if not leg:
                continue
            iv = leg.get("implied_volatility")
            iv = (float(iv) / 100.0) if iv else None
            quotes.append(OptionQuote(
                strike=k, is_call=is_call,
                ltp=float(leg.get("last_price") or 0),
                bid=float((leg.get("top_bid_price") or 0)),
                ask=float((leg.get("top_ask_price") or 0)),
                oi=int(leg.get("oi") or 0), prev_oi=int(leg.get("previous_oi") or 0),
                volume=int(leg.get("volume") or 0), iv=iv))
    return Chain(symbol=symbol, spot=spot, expiry=dt.date.fromisoformat(expiry),
                 lot_size=lot_size, quotes=quotes, asof=dt.date.today(),
                 source="dhan")


# ── live: trading Gateway (Shoonya via gateway_system) ─────────────────
def from_gateway_payload(d: dict, symbol: str | None = None) -> Chain:
    """Build a Chain from a Gateway `/api/option-chain` response.

    Split out from `from_gateway` so a captured payload can be replayed offline
    — the mapping is where a field-name mistake would silently mis-price a whole
    chain, so it must be testable without a broker session.

    Legs the Gateway could not quote (`quoted` false) are dropped rather than
    admitted at zero: a strike present in the chain is one the advisor may build
    a structure on, and a zero-priced leg reads as free optionality.
    """
    spot = float(d.get("spot") or 0)
    expiry_iso = d.get("expiry_iso") or ""
    if not expiry_iso:
        raise ValueError(f"gateway returned no usable expiry for "
                         f"{d.get('symbol', symbol)} (expiry={d.get('expiry')!r})")
    if spot <= 0:
        raise ValueError(
            f"gateway could not resolve the underlying price for "
            f"{d.get('symbol', symbol)} — every strike would be judged against a "
            f"spot of 0, so no read of this chain would mean anything")

    quotes: list[OptionQuote] = []
    for row in d.get("chain", []):
        for key, is_call in (("CE", True), ("PE", False)):
            leg = row.get(key)
            if not leg or not leg.get("quoted"):
                continue
            ltp = float(leg.get("ltp") or 0)
            bid = float(leg.get("bid") or 0)
            ask = float(leg.get("ask") or 0)
            if ltp <= 0 and bid <= 0 and ask <= 0:
                continue
            quotes.append(OptionQuote(
                strike=float(leg.get("strike") or row.get("strike") or 0),
                is_call=is_call, ltp=ltp, bid=bid, ask=ask,
                oi=int(leg.get("oi_num") or 0),
                prev_oi=int(leg.get("prev_oi") or 0),
                volume=int(leg.get("volume") or 0),
                iv=None))            # Shoonya sends no IV — inverted on load

    lot = int(d.get("lot_size") or 1)
    return Chain(symbol=d.get("symbol") or symbol or "", spot=spot,
                 expiry=dt.date.fromisoformat(expiry_iso), lot_size=lot,
                 quotes=quotes, asof=dt.date.today(),
                 source=f"gateway:{d.get('exchange', '')}")


def from_gateway(symbol: str, exchange: str = "NSE", *, expiry: str = "",
                 count: int = 15, client=None, use_cache: bool = True) -> Chain:
    """Live option chain from the Gateway (broker session lives there, not here).

        chain = from_gateway("RELIANCE")            # nearest expiry, ±15 strikes
        chain = from_gateway("NIFTY", expiry="25-AUG-2026")

    `count` is strikes on EACH side of ATM. Wider is not free: the Gateway pays
    one broker round-trip per leg, so ±15 is ~60 quotes.
    """
    from ..gateway.client import GatewayClient
    own = client is None
    client = client or GatewayClient()
    try:
        d = client.option_chain(symbol, exchange, expiry=expiry, count=count,
                                use_cache=use_cache)
    finally:
        if own:
            client.close()
    if d.get("error"):
        raise ValueError(f"gateway: {d['error']}")
    return from_gateway_payload(d, symbol)


# ── post-load enrichment ───────────────────────────────────────────────
# Bounds on a believable carry. An Indian stock future's basis is a financing
# rate net of dividends: negative around a big dividend, well above repo when
# the stock is hard to borrow. Outside this band the estimate is not a carry,
# it is a stale or crossed quote, and the fixed default is the safer answer.
_CARRY_MIN, _CARRY_MAX = -0.35, 0.60
_CARRY_MIN_ESTIMATES = 3


def calibrate_carry(chain: Chain) -> Chain:
    """Set `chain.carry_rate` from the chain's own put-call parity.

    C - P + K is the forward at every strike, so the near-the-money strikes give
    several independent estimates and the median shrugs off one stale leg. This
    is the market's own answer and needs no rate assumption.

    It must run BEFORE any IV is inverted: inverting against the wrong forward
    is what produces the phantom call-over-put skew.

    A chain that arrives with an explicit carry (or that cannot supply enough
    paired strikes) is left alone.
    """
    if chain.carry_rate is not None or chain.spot <= 0:
        return chain
    t = chain.t_years
    if t <= 0:
        return chain

    fwd, _n = chain.implied_forward(min_estimates=_CARRY_MIN_ESTIMATES)
    if not fwd or fwd <= 0:
        return chain
    r = math.log(fwd / chain.spot) / t
    if _CARRY_MIN <= r <= _CARRY_MAX:
        chain.carry_rate = r
    return chain


def fill_missing_ivs(chain: Chain) -> Chain:
    """Calibrate the carry, then invert BS for any quote that arrived without
    an IV. The order matters — see `calibrate_carry`."""
    calibrate_carry(chain)
    t = chain.t_years
    r = chain.r
    for q in chain.quotes:
        if q.iv is None or q.iv <= 0:
            q.iv = implied_vol(q.is_call, q.mid, chain.spot, q.strike, t, r)
    return chain
