"""Gateway integration tests — no network, no broker, no pytest.

    python tests/test_gateway.py

The payload→Chain mapping is the one place where a wrong field name produces no
error at all, just a chain that prices differently: read `lp` where `bid` was
meant and every structure shows edge it cannot capture. So the fixture below is
a verbatim-shaped Gateway response and the assertions pin the exact fields.

The fixture is deliberately awkward on purpose — an unquoted leg, a leg with no
book, a leg with no prior OI snapshot — because those are the normal state of a
real chain's wings, not edge cases.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionsmith import advise, from_gateway_payload                  # noqa: E402
from optionsmith.chain.loaders import fill_missing_ivs                # noqa: E402
from optionsmith.gateway.config import GatewaySettings                # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _leg(strike, token, ltp, bid, ask, oi, prev_oi, vol=1000, quoted=True):
    return {"token": token, "tsym": f"T{token}", "exch": "NFO",
            "lp": str(ltp), "oi": str(oi), "v": str(vol),
            "strike": strike, "ltp": ltp, "bid": bid, "ask": ask,
            "bid_qty": 500, "ask_qty": 500, "oi_num": oi, "prev_oi": prev_oi,
            "volume": vol, "prev_close": ltp, "lot_size": 500,
            "tick_size": 0.05, "quoted": quoted}


def payload(**over) -> dict:
    """A Gateway /api/option-chain response, shaped exactly as the server sends
    it (spot 1320, RELIANCE-like, 5 strikes each side of a 20-point grid)."""
    rows = []
    for i, k in enumerate([1260.0, 1280.0, 1300.0, 1320.0, 1340.0, 1360.0, 1380.0]):
        moneyness = k - 1320.0
        ce_px = max(0.05, 30.0 - moneyness * 0.45)
        pe_px = max(0.05, 30.0 + moneyness * 0.45)
        rows.append({
            "strike": k,
            "CE": _leg(k, f"1{i}0", ce_px, round(ce_px - 0.15, 2),
                       round(ce_px + 0.15, 2), 5000 + i * 100, 4800 + i * 100),
            "PE": _leg(k, f"2{i}0", pe_px, round(pe_px - 0.15, 2),
                       round(pe_px + 0.15, 2), 6000 + i * 100, 6100 + i * 100),
        })
    # the wings as they really arrive: one leg never quoted, one with no book
    rows[0]["CE"] = _leg(1260.0, "199", 0.0, 0.0, 0.0, 0, 0, 0, quoted=False)
    rows[-1]["CE"] = _leg(1380.0, "160", 0.85, 0.0, 0.0, 900, 0, 12)
    d = {
        "symbol": "RELIANCE", "exchange": "NFO",
        "expiry": "25-AUG-2026", "expiry_iso": "2026-08-25",
        "expiries": ["25-AUG-2026", "29-SEP-2026"],
        "expiries_iso": ["2026-08-25", "2026-09-29"],
        "spot": 1320.0, "lot_size": 500,
        "underlying_exchange": "NSE", "underlying_token": "2885",
        "asof": "2026-08-17T15:15:00", "cached": False, "truncated": False,
        "quality": {"legs": 14, "quoted": 13, "prev_oi_known": 12},
        "chain": rows, "error": "",
    }
    d.update(over)
    return d


# ── 1. mapping ─────────────────────────────────────────────────────────
def test_mapping() -> None:
    print("\n[1] gateway payload -> Chain")
    ch = from_gateway_payload(payload())

    check("spot taken from the payload, not inferred", ch.spot == 1320.0)
    check("expiry parsed from expiry_iso",
          ch.expiry == dt.date(2026, 8, 25), str(ch.expiry))
    check("lot size carried through", ch.lot_size == 500)
    check("source records the origin", ch.source.startswith("gateway:"))

    # 7 strikes x 2 sides = 14, minus the one leg the gateway could not quote
    check("unquoted leg dropped, not zero-priced", len(ch.quotes) == 13,
          f"{len(ch.quotes)} quotes")
    check("no zero-priced contract survived",
          all(q.ltp > 0 or q.bid > 0 or q.ask > 0 for q in ch.quotes))

    atm_ce = ch.get(1320.0, True)
    check("bid/ask mapped from the book, not from lp",
          atm_ce.bid == 29.85 and atm_ce.ask == 30.15,
          f"bid={atm_ce.bid} ask={atm_ce.ask}")
    check("buy fills at ask, sell at bid", atm_ce.exec_price(+1) == 30.15
          and atm_ce.exec_price(-1) == 29.85)
    check("oi read from oi_num (the int), not the display string",
          atm_ce.oi == 5300, str(atm_ce.oi))
    check("prev_oi carried through for the OI delta", atm_ce.prev_oi == 5100)
    check("volume mapped", atm_ce.volume == 1000)

    nobook = ch.get(1380.0, True)
    check("leg with no book keeps its LTP", nobook.ltp == 0.85)
    check("leg with no book falls back to LTP for execution",
          nobook.exec_price(+1) == 0.85 and nobook.exec_price(-1) == 0.85)
    check("leg with no book reports no spread", nobook.spread_pct == 0.0)


# ── 2. refusal to guess ────────────────────────────────────────────────
def test_refusals() -> None:
    print("\n[2] refuses to build a chain it cannot price")
    for label, over in (("spot of 0", {"spot": 0.0}),
                        ("missing expiry_iso", {"expiry_iso": ""})):
        try:
            from_gateway_payload(payload(**over))
            check(f"{label} rejected", False, "no error raised")
        except ValueError:
            check(f"{label} rejected", True)


# ── 3. unknown vs zero OI change ───────────────────────────────────────
def test_oi_delta() -> None:
    print("\n[3] OI change distinguishes unknown from unchanged")
    ch = from_gateway_payload(payload())
    known = ch.get(1320.0, True)
    unknown = ch.get(1380.0, True)      # prev_oi 0 = never snapshotted
    check("known prev_oi yields a real delta", known.d_oi == 200, str(known.d_oi))
    check("known prev_oi yields a real percentage",
          abs(known.d_oi_pct - 200 / 5100 * 100) < 1e-9)
    check("absent prev_oi reports 0%, never a fabricated build",
          unknown.d_oi_pct == 0.0)

    m = compute_metrics_safe(ch)
    check("metrics survive a chain with partial OI history", m is not None)


def compute_metrics_safe(ch):
    from optionsmith import compute_metrics
    return compute_metrics(fill_missing_ivs(ch))


# ── 4. IV backfill ─────────────────────────────────────────────────────
def test_iv_backfill() -> None:
    print("\n[4] IV inverted on load (the broker sends none)")
    ch = from_gateway_payload(payload())
    check("no IV arrives from the gateway",
          all(q.iv is None for q in ch.quotes))
    fill_missing_ivs(ch)
    priced = [q for q in ch.quotes if q.iv and q.iv > 0]
    check("IV inverted for the quotes that support it", len(priced) >= 8,
          f"{len(priced)} of {len(ch.quotes)}")
    atm = ch.get(1320.0, True)
    check("ATM IV is a plausible decimal, not a percent",
          0.01 < (atm.iv or 0) < 3.0, str(atm.iv))


# ── 5. the advisor runs end to end on gateway data ─────────────────────
def test_advise_on_gateway_chain() -> None:
    print("\n[5] advisor end-to-end on a gateway chain")
    ch = from_gateway_payload(payload())
    rep = advise(ch, user_direction="neutral", iv_percentile=85.0, top_n=3)
    check("report names the gateway symbol", rep.symbol == "RELIANCE")
    check("report carries the real lot size", rep.lot_size == 500)
    check("report is JSON-serialisable", bool(json.dumps(rep.to_dict())))
    check("partial-book warning raised for the wings",
          any("two-sided book" in w for w in rep.warnings), str(rep.warnings))


# ── 6. settings ────────────────────────────────────────────────────────
def test_settings() -> None:
    print("\n[6] gateway settings")
    import os
    keep = {k: os.environ.get(k) for k in
            ("GATEWAY_BASE_URL", "GATEWAY_CLIENT_ID", "GATEWAY_CLIENT_SECRET")}
    try:
        os.environ["GATEWAY_BASE_URL"] = "http://gw.example:8000/"
        os.environ["GATEWAY_CLIENT_ID"] = "svc_x"
        os.environ["GATEWAY_CLIENT_SECRET"] = "s3cret"
        s = GatewaySettings.load()
        check("trailing slash stripped from base url",
              s.base_url == "http://gw.example:8000", s.base_url)
        check("credentials detected as configured", s.configured)
        os.environ["GATEWAY_CLIENT_SECRET"] = ""
        check("half-set credentials count as unconfigured",
              not GatewaySettings.load().configured)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    print("OptionSmith gateway suite")
    for fn in (test_mapping, test_refusals, test_oi_delta, test_iv_backfill,
               test_advise_on_gateway_chain, test_settings):
        fn()
    print(f"\n{'='*46}\n  {PASS} passed, {FAIL} failed\n{'='*46}")
    sys.exit(1 if FAIL else 0)
