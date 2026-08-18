"""Chain analytics — what the options market is actually saying.

Every metric is computed from the chain alone (no history feed required),
and each one is reported with the caveat that matters for reading it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.mathx import RISK_FREE, bs_gamma
from ..core.models import Chain


@dataclass
class ChainMetrics:
    symbol: str
    spot: float
    atm: float
    dte: int
    call_wall: float | None = None          # max call OI = resistance
    put_wall: float | None = None           # max put OI = support
    call_wall_oi: int = 0
    put_wall_oi: int = 0
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    max_pain: float | None = None
    atm_iv: float | None = None
    iv_skew_25d: float | None = None        # IV(25d put) - IV(25d call)
    total_call_oi: int = 0
    total_put_oi: int = 0
    call_concentration: float = 0.0         # top-2 strikes' share of call OI
    put_concentration: float = 0.0
    net_gex: float = 0.0                    # dealer-gamma proxy
    gamma_flip: float | None = None
    oi_builds: list[dict] = field(default_factory=list)
    oi_unwinds: list[dict] = field(default_factory=list)
    liquid_strikes: int = 0
    forward: float | None = None            # implied by put-call parity
    basis_pct: float | None = None          # (forward - spot) / spot * 100
    carry_implied: float | None = None      # annualised rate the chain implies
    smile_spread: float | None = None       # max-min liquid IV, in vol points
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        for k in ("pcr_oi", "pcr_volume", "call_concentration",
                  "put_concentration"):
            d[k] = round(d[k], 3)
        d["atm_iv_pct"] = round(self.atm_iv * 100, 1) if self.atm_iv else None
        d["iv_skew_25d_pts"] = (round(self.iv_skew_25d * 100, 1)
                                if self.iv_skew_25d else None)
        d["net_gex"] = round(self.net_gex, 1)
        d["forward"] = round(self.forward, 2) if self.forward else None
        d["basis_pct"] = round(self.basis_pct, 3) if self.basis_pct is not None else None
        d["carry_implied_pct"] = (round(self.carry_implied * 100, 2)
                                  if self.carry_implied is not None else None)
        d["smile_spread"] = (round(self.smile_spread, 1)
                             if self.smile_spread is not None else None)
        return d




def liquid_smile_spread(chain: Chain) -> float | None:
    """Max-min IV across the LIQUID contracts, in vol points. None if too thin.

    Shared with the ranker rather than left inline in `compute()`: this is the
    number that says how much of a multi-strike structure's EV is smile
    artifact rather than edge, and the generator needs it without paying for a
    full metrics pass. Illiquid wings invert to absurd IVs and would make every
    chain look extreme, so they are excluded.
    """
    ivs = [q.iv for q in chain.liquid() if q.iv and q.iv > 0]
    if len(ivs) < 4:
        return None
    return (max(ivs) - min(ivs)) * 100.0


def _max_pain(chain: Chain) -> float | None:
    """Strike minimising total in-the-money value of all open contracts."""
    ks = chain.strikes
    if not ks:
        return None
    best, best_pain = None, float("inf")
    for k in ks:
        pain = 0.0
        for q in chain.quotes:
            if q.oi <= 0:
                continue
            itm = max(0.0, k - q.strike) if q.is_call else max(0.0, q.strike - k)
            pain += itm * q.oi
        if pain < best_pain:
            best, best_pain = k, pain
    return best


def _concentration(quotes) -> float:
    tot = sum(q.oi for q in quotes)
    if tot <= 0:
        return 0.0
    top2 = sum(sorted((q.oi for q in quotes), reverse=True)[:2])
    return top2 / tot


def _skew_25d(chain: Chain) -> float | None:
    """IV(25-delta put) − IV(25-delta call). Positive = puts bid (fear)."""
    from ..core.mathx import bs_delta
    t = chain.t_years
    best_c = best_p = None
    for q in chain.quotes:
        if not q.iv:
            continue
        d = bs_delta(q.is_call, chain.spot, q.strike, t, q.iv, chain.r)
        target = 0.25 if q.is_call else -0.25
        err = abs(d - target)
        if q.is_call and (best_c is None or err < best_c[0]):
            best_c = (err, q.iv)
        if not q.is_call and (best_p is None or err < best_p[0]):
            best_p = (err, q.iv)
    if best_c and best_p:
        return best_p[1] - best_c[1]
    return None


def _gex(chain: Chain) -> tuple[float, float | None]:
    """Dealer gamma exposure profile and the flip level.

    CONVENTION (stated, because it is the assumption that invalidates most
    GEX work): dealers are assumed SHORT calls / LONG puts against retail —
    i.e. call OI contributes positive gamma, put OI negative. Without signed
    dealer inventory this is a heuristic, so GEX is used for CONTEXT
    (pin/whip risk), never as a direction signal.
    """
    t = chain.t_years
    per_strike: dict[float, float] = {}
    for q in chain.quotes:
        if not q.iv or q.oi <= 0:
            continue
        g = bs_gamma(chain.spot, q.strike, t, q.iv, chain.r)
        val = g * q.oi * chain.lot_size * chain.spot * chain.spot * 0.01
        per_strike[q.strike] = per_strike.get(q.strike, 0.0) + \
            (val if q.is_call else -val)
    net = sum(per_strike.values())
    flip = None
    ks = sorted(per_strike)
    run = 0.0
    for k in ks:                     # cumulative sign change = flip level
        prev = run
        run += per_strike[k]
        if prev < 0 <= run or prev > 0 >= run:
            flip = k
    return net, flip


def compute(chain: Chain, build_pct: float = 20.0,
            unwind_pct: float = -30.0, min_oi: int = 100) -> ChainMetrics:
    calls, puts = chain.calls(), chain.puts()
    m = ChainMetrics(symbol=chain.symbol, spot=chain.spot, atm=chain.atm,
                     dte=chain.days_to_expiry)

    if calls:
        cw = max(calls, key=lambda q: q.oi)
        m.call_wall, m.call_wall_oi = cw.strike, cw.oi
    if puts:
        pw = max(puts, key=lambda q: q.oi)
        m.put_wall, m.put_wall_oi = pw.strike, pw.oi

    m.total_call_oi = sum(q.oi for q in calls)
    m.total_put_oi = sum(q.oi for q in puts)
    m.pcr_oi = (m.total_put_oi / m.total_call_oi) if m.total_call_oi else 0.0
    cv, pv = sum(q.volume for q in calls), sum(q.volume for q in puts)
    m.pcr_volume = (pv / cv) if cv else 0.0
    m.call_concentration = _concentration(calls)
    m.put_concentration = _concentration(puts)
    m.max_pain = _max_pain(chain)

    atm_c, atm_p = chain.get(chain.atm, True), chain.get(chain.atm, False)
    ivs = [q.iv for q in (atm_c, atm_p) if q and q.iv]
    m.atm_iv = (sum(ivs) / len(ivs)) if ivs else None
    m.iv_skew_25d = _skew_25d(chain)
    m.net_gex, m.gamma_flip = _gex(chain)

    for q in chain.quotes:
        if q.prev_oi < min_oi:
            continue
        rec = {"strike": q.strike, "right": q.right, "d_oi": q.d_oi,
               "d_oi_pct": round(q.d_oi_pct, 1), "oi": q.oi}
        if q.d_oi_pct >= build_pct:
            m.oi_builds.append(rec)
        elif q.d_oi_pct <= unwind_pct:
            m.oi_unwinds.append(rec)
    m.oi_builds.sort(key=lambda r: -abs(r["d_oi"]))
    m.oi_unwinds.sort(key=lambda r: -abs(r["d_oi"]))
    m.liquid_strikes = len({q.strike for q in chain.liquid()})

    if m.liquid_strikes < 5:
        m.notes.append("thin chain — few liquid strikes; treat all reads as weak")
    if m.dte <= 2:
        m.notes.append("expiry week — pinning and settlement effects dominate")
    fwd, n_est = chain.implied_forward()
    t = chain.t_years
    if fwd and t > 0:
        m.forward = fwd
        m.basis_pct = (fwd - chain.spot) / chain.spot * 100.0
        m.carry_implied = math.log(fwd / chain.spot) / t
        # The chain is priced off its forward, and everything here prices off
        # spot with drift `chain.r`. Those agree once the carry is calibrated;
        # when they do not — an uncalibrated chain, or a basis outside the
        # believable band — the gap lands on IVs, skew and EV in opposite
        # directions for calls and puts, so it has to be said out loud.
        model_fwd = chain.spot * math.exp(chain.r * t)
        drift_pct = abs(fwd - model_fwd) / chain.spot * 100.0
        if drift_pct > 0.10:
            m.notes.append(
                f"chain implies a forward of {fwd:,.2f} ({m.basis_pct:+.2f}% over "
                f"spot, carry {m.carry_implied * 100:.1f}%/yr from {n_est} strikes) "
                f"but pricing here uses {model_fwd:,.2f} at {chain.r * 100:.1f}% — "
                f"IVs, skew and EV carry that {drift_pct:.2f}% reference error")
        elif abs(m.basis_pct) > 0.05:
            m.notes.append(
                f"priced off the chain's own forward {fwd:,.2f} "
                f"({m.basis_pct:+.2f}% over cash spot, carry "
                f"{m.carry_implied * 100:.1f}%/yr) — not the cash price")

    # Smile steepness across the LIQUID contracts (the ones a structure would
    # actually be built from). Illiquid wings invert to absurd IVs and would
    # make every chain look extreme.
    m.smile_spread = liquid_smile_spread(chain)
    if m.smile_spread is not None:
        if m.smile_spread > 5.0:
            m.notes.append(
                f"smile spans {m.smile_spread:.1f} vol points across liquid "
                f"strikes — structures are priced leg-by-leg off the smile but "
                f"valued at one blended sigma, so EV and the ranking overstate "
                f"multi-strike shapes (see _sigma_for). Compare RR and POP, not EV")

    # A live chain is not all-or-nothing on the book: liquid strikes quote two
    # sides and the far wings quote none, so reporting only the "no bid/ask at
    # all" case would stay silent on a chain where half the legs are priced at a
    # stale LTP nobody is showing.
    booked = sum(1 for q in chain.quotes if q.bid > 0 and q.ask > 0)
    total = len(chain.quotes)
    if total and not booked:
        m.notes.append("no bid/ask in feed — execution prices fall back to LTP")
    elif total and booked < total:
        m.notes.append(
            f"{total - booked}/{total} contracts have no two-sided book — those "
            f"legs are priced at LTP, which is not an executable price")
    return m
