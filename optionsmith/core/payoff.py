"""Exact payoff mathematics + realistic probability of profit.

Everything a strategy claim needs to be honest:

  * payoff curve at expiry, computed on the exact kink points (no grid error)
  * max profit / max loss per LOT in rupees, with unbounded sides detected
    analytically from the net slope beyond the outermost strikes
  * breakevens found exactly by linear interpolation between kinks
  * POP with three realism upgrades over the textbook number:
      1. FAT TAILS — terminal log-return is Student-t (df=5 default), which
         assigns real probability to the crash moves that blow through short
         wings; the normal assigns ~0 and prints 99.9% for wide condors.
      2. FRICTION — a "win" is payoff > costs, not payoff > 0.
      3. SMILE — sigma is the mean of the LEGS' own IVs when available, so
         wings are not priced with a single low ATM vol.
    The textbook figure is still returned as pop_classic_pct so the optimism
    gap stays visible.
"""
from __future__ import annotations

import math

from .mathx import (RISK_FREE, bs_delta, bs_gamma, bs_theta, bs_vega, t_cdf,
                    norm_cdf)
from .models import Chain, Leg, StrategyResult

TAIL_DF = 5.0                 # Student-t df; equity daily returns fit ~3-6
COST_PER_LEG = 40.0           # rupees per leg round trip (brokerage+STT+slippage)


# ── payoff geometry ────────────────────────────────────────────────────
def net_premium(legs: list[Leg]) -> float:
    """Per-share net premium. Positive = net debit paid."""
    return sum(l.side * l.qty * l.price for l in legs)


def payoff_at(legs: list[Leg], spot: float) -> float:
    """Per-share expiry P&L of the whole structure."""
    return sum(l.payoff_at(spot) for l in legs)


def kink_points(legs: list[Leg], spot: float) -> list[float]:
    """Strikes plus probe points outside them — where the curve can bend."""
    ks = sorted({l.strike for l in legs})
    if not ks:
        return [spot]
    span = max(ks[-1] - ks[0], spot * 0.35) + spot * 0.1
    return [max(0.01, ks[0] - span)] + ks + [ks[-1] + span]


def upside_slope(legs: list[Leg]) -> float:
    """d(payoff)/dS above every strike — only calls contribute there."""
    return sum(l.side * l.qty for l in legs if l.is_call)


def extremes(legs: list[Leg], spot: float, lot: int) -> tuple[float, float]:
    """(max_profit, max_loss) in rupees per lot; inf only when truly unbounded.

    The downside is NEVER unbounded: the underlying cannot go below zero, so
    the payoff at S=0 is the terminal downside value. Only the upside can run
    to infinity (a net-short call position). Getting this wrong marks every
    put-containing structure as infinite-risk — which silently deleted every
    generated structure with a put leg in testing.
    """
    pts = kink_points(legs, spot) + [0.0]          # S=0 is a real, reachable node
    vals = [payoff_at(legs, p) for p in pts]
    up = upside_slope(legs)
    hi, lo = max(vals) * lot, min(vals) * lot
    max_profit = float("inf") if up > 0 else hi
    max_loss = float("inf") if up < 0 else abs(min(0.0, lo))
    if max_loss == 0 and max_profit != float("inf") and hi <= 0:
        max_loss = abs(hi)                          # structure that only loses
    return max_profit, max_loss


def breakevens(legs: list[Leg], spot: float) -> list[float]:
    """Exact zero-crossings — linear interpolation between kinks is exact
    because the payoff is piecewise-linear in the underlying."""
    pts = kink_points(legs, spot)
    out: list[float] = []
    for a, b in zip(pts, pts[1:]):
        fa, fb = payoff_at(legs, a), payoff_at(legs, b)
        if fa == 0:
            out.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            out.append(a + (b - a) * (-fa) / (fb - fa))
    return sorted({round(x, 4) for x in out if x > 0})


# ── terminal distribution ──────────────────────────────────────────────
def _sigma_for(legs: list[Leg], fallback: float) -> float:
    """One sigma for the whole structure: the mean of its legs' own IVs.

    KNOWN BIAS, and it is not small on a real chain. The legs are PRICED at
    their individual IVs off the smile, but the payoff is then integrated under
    a single-sigma lognormal. Any structure spanning strikes with different IVs
    therefore books the smile's curvature as edge, and the wider the span the
    larger the phantom EV.

    Measured on a live 8-DTE RELIANCE chain (smile spread 15.9 vol points),
    neutral view and no volatility thesis — where the model's own invariant says
    EV should be about -(friction + spread):

        real smile   ratio backspread   EV +4,169
        flat smile   (same chain)       EV   -164   <- the correct answer

    Removing it needs the terminal density built from the smile itself
    (Breeden-Litzenberger on the call curve) rather than from one sigma. Until
    then `ChainMetrics` warns whenever a chain's smile is steep enough for this
    to dominate the ranking.
    """
    ivs = [l.iv for l in legs if l.iv and l.iv > 0]
    return (sum(ivs) / len(ivs)) if ivs else fallback


def prob_below(price: float, spot: float, sigma: float, t: float,
               fat: bool = True, r: float = RISK_FREE) -> float:
    """P(S_T <= price). Student-t (variance- and median-matched) or lognormal."""
    if price <= 0 or spot <= 0 or sigma <= 0 or t <= 0:
        return 0.0 if price < spot else 1.0
    v = sigma * math.sqrt(t)
    mu = math.log(spot) + (r - 0.5 * sigma * sigma) * t     # lognormal median
    z = (math.log(price) - mu) / v
    if not fat or TAIL_DF <= 2:
        return norm_cdf(z)
    # scale the t so its VARIANCE matches the lognormal's (t has var df/(df-2))
    scale = math.sqrt((TAIL_DF - 2.0) / TAIL_DF)
    return t_cdf(z * scale, TAIL_DF)


def prob_of_profit(legs: list[Leg], spot: float, sigma: float, t: float,
                   lot: int, friction: float, fat: bool = True,
                   r: float = RISK_FREE) -> float:
    """P(payoff per lot > friction) — integrates the profitable regions.

    The drift MUST match the measure the options are priced under (the
    forward, spot*e^rt). Using zero drift against forward-priced options
    hands every put a free ~r*t of phantom edge — visible in testing as
    "long put" topping a strictly neutral view. The caller's directional
    opinion enters through the tilt, not through the drift.
    """
    thresh = friction / max(lot, 1)                     # per share
    pts = kink_points(legs, spot)
    # find all crossings of the threshold line, then sum probability mass of
    # the segments where payoff > threshold
    xs: list[float] = []
    for a, b in zip(pts, pts[1:]):
        fa = payoff_at(legs, a) - thresh
        fb = payoff_at(legs, b) - thresh
        if (fa < 0 < fb) or (fb < 0 < fa):
            xs.append(a + (b - a) * (-fa) / (fb - fa))
    edges = [0.0] + sorted(x for x in xs if x > 0) + [float("inf")]
    total = 0.0
    for a, b in zip(edges, edges[1:]):
        probe = (a + b) / 2 if b != float("inf") else max(a * 1.5, spot * 2)
        if payoff_at(legs, probe) - thresh > 0:
            pa = prob_below(a, spot, sigma, t, fat, r) if a > 0 else 0.0
            pb = 1.0 if b == float("inf") else prob_below(b, spot, sigma, t, fat, r)
            total += max(0.0, pb - pa)
    return max(0.0, min(1.0, total)) * 100.0


def payoff_moments(legs: list[Leg], spot: float, sigma: float, t: float,
                   lot: int, tilt: float = 0.0, n: int = 401,
                   r: float = RISK_FREE) -> tuple[float, float]:
    """(mean, standard deviation) of the per-lot expiry payoff under the view.

    The std is what stops the optimiser degenerating into "buy the furthest
    OTM call": maximising EV per rupee of MAX LOSS always prefers maximum
    leverage, because a cheap lottery ticket has a tiny denominator. Ranking
    on EV per unit of payoff VOLATILITY is the risk-adjusted question a
    professional actually asks.
    """
    if spot <= 0 or sigma <= 0 or t <= 0:
        return 0.0, 0.0
    center = spot * (1.0 + tilt)
    width = max(sigma * math.sqrt(t), 0.02)
    lo, hi = center * math.exp(-5 * width), center * math.exp(5 * width)
    step = (hi - lo) / (n - 1)
    m1 = m2 = 0.0
    prev = prob_below(lo, center, sigma, t, fat=False, r=r)
    for i in range(1, n):
        x = lo + i * step
        cdf = prob_below(x, center, sigma, t, fat=False, r=r)
        w = cdf - prev
        prev = cdf
        p = payoff_at(legs, x - step / 2) * lot
        m1 += w * p
        m2 += w * p * p
    var = max(0.0, m2 - m1 * m1)
    return m1, math.sqrt(var)


def expected_value(legs: list[Leg], spot: float, sigma: float, t: float,
                   lot: int, tilt: float = 0.0, n: int = 401,
                   r: float = RISK_FREE) -> float:
    """E[payoff] per lot under the view-tilted terminal density.

    Two deliberate choices, both of which change the answer materially:

      * LOGNORMAL, not Student-t. A log-t has NO finite mean (E[e^X] diverges),
        so integrating the fat-tailed density produces an EV dominated by the
        truncation point — it made every long call look like a 2x edge in
        testing. Fat tails stay where they are well-defined: the POP.
      * ZERO DRIFT plus the view tilt. The median is spot*(1+tilt), so with a
        neutral view a fairly-priced option has EV ~ 0 and the only edge that
        can show up is the one the USER's view puts there. Using the
        risk-free drift instead would manufacture a permanent bullish bias.
    """
    if spot <= 0 or sigma <= 0 or t <= 0:
        return 0.0
    center = spot * (1.0 + tilt)
    width = max(sigma * math.sqrt(t), 0.02)
    lo, hi = center * math.exp(-5 * width), center * math.exp(5 * width)
    step = (hi - lo) / (n - 1)
    total = 0.0
    prev_cdf = prob_below(lo, center, sigma, t, fat=False, r=r)
    for i in range(1, n):
        x = lo + i * step
        cdf = prob_below(x, center, sigma, t, fat=False, r=r)
        w = cdf - prev_cdf
        prev_cdf = cdf
        total += w * payoff_at(legs, x - step / 2)
    # tail masses beyond the integration window, valued at the terminal slopes
    total += prev_cdf_tail(legs, lo, hi, spot, sigma, t, center, r)
    return total * lot


def prev_cdf_tail(legs: list[Leg], lo: float, hi: float, spot: float,
                  sigma: float, t: float, center: float,
                  r: float = RISK_FREE) -> float:
    """Value the truncated tails at their (linear) terminal payoff so wide
    short structures are not credited with mass they never keep."""
    p_lo = prob_below(lo, center, sigma, t, fat=False, r=r)
    p_hi = 1.0 - prob_below(hi, center, sigma, t, fat=False, r=r)
    return p_lo * payoff_at(legs, lo * 0.9) + p_hi * payoff_at(legs, hi * 1.1)


# ── greeks & margin ────────────────────────────────────────────────────
def position_greeks(legs: list[Leg], spot: float, t: float, lot: int,
                    fallback_iv: float, r: float = RISK_FREE) -> dict:
    d = g = v = th = 0.0
    for l in legs:
        iv = l.iv or fallback_iv
        mult = l.side * l.qty * lot
        d += mult * bs_delta(l.is_call, spot, l.strike, t, iv, r)
        g += mult * bs_gamma(spot, l.strike, t, iv, r)
        v += mult * bs_vega(spot, l.strike, t, iv, r) / 100.0   # per vol point
        th += mult * bs_theta(l.is_call, spot, l.strike, t, iv, r) / 365.0
    return {"delta": d, "gamma": g, "vega": v, "theta": th}


def margin_estimate(legs: list[Leg], spot: float, lot: int,
                    max_loss: float) -> float:
    """Rough SPAN+exposure proxy — an estimate; the broker is the truth.

    Cover logic (the first version had this backwards and billed every
    vertical spread as if it were naked): a short option is offset by ANY
    long of the SAME RIGHT once quantities match — the strikes only decide
    WHERE the loss caps, not whether it caps. So count unmatched shorts per
    right and charge those: ~15% of spot notional for naked calls, ~15% of
    the strike value for naked (cash-secured) puts.
    """
    naked_margin = 0.0
    for is_call in (True, False):
        longs = sum(l.qty for l in legs if l.side > 0 and l.is_call == is_call)
        shorts = [l for l in legs if l.side < 0 and l.is_call == is_call]
        uncovered = max(0, sum(l.qty for l in shorts) - longs)
        if not uncovered:
            continue
        if is_call:
            naked_margin += uncovered * 0.15 * spot * lot
        else:
            k = max((l.strike for l in shorts), default=spot)
            naked_margin += uncovered * 0.15 * k * lot
    if naked_margin:
        return naked_margin
    return max_loss if max_loss != float("inf") else 0.0


# ── the one entry point everything else uses ───────────────────────────
def evaluate(legs: list[Leg], chain: Chain, name: str = "custom structure",
             tilt: float = 0.0, use_exec_prices: bool = True,
             rationale: str = "", tags: list[str] | None = None,
             is_custom: bool = False, vol_mult: float = 1.0,
             source: str = "library", snap_strikes: bool = False
             ) -> StrategyResult:
    """Price the legs off the chain and produce a complete StrategyResult."""
    lot = chain.lot_size
    t = chain.t_years
    atm_q = chain.get(chain.atm, True)
    fallback_iv = (atm_q.iv if atm_q and atm_q.iv else 0.30)

    priced: list[Leg] = []
    for l in legs:
        q = chain.get(l.strike, l.is_call)
        if q is None and snap_strikes:
            # hand-typed strikes rarely land exactly on the chain's grid —
            # snap to the nearest listed strike of the same right
            avail = [x.strike for x in chain.quotes if x.is_call == l.is_call]
            if avail:
                k = min(avail, key=lambda x: abs(x - l.strike))
                q = chain.get(k, l.is_call)
                l = Leg(k, l.is_call, l.side, l.qty, l.price, l.iv)
        if q is None:
            continue
        px = q.exec_price(l.side) if use_exec_prices else q.mid
        priced.append(Leg(l.strike, l.is_call, l.side, l.qty, px,
                          q.iv or fallback_iv))
    if len(priced) != len(legs) or not priced:
        raise ValueError("one or more legs are not present in the chain")

    friction = COST_PER_LEG * sum(l.qty for l in priced)
    mp, ml = extremes(priced, chain.spot, lot)
    sigma = _sigma_for(priced, fallback_iv)
    # legs are PRICED at market IV, but valued under the vol we EXPECT to be
    # realised (vol_mult < 1 when IV is rich) — that asymmetry is the edge
    eval_sigma = max(1e-4, sigma * vol_mult)
    # The drift must be the carry the CHAIN is priced under, not a constant.
    # Getting it wrong does not merely shift EV: it moves the whole terminal
    # density off the forward, and ITM options then look mispriced against
    # intrinsic — which the optimiser dutifully "arbitrages".
    r = chain.r
    center = chain.spot * (1.0 + tilt)          # POP under the same view
    pop = prob_of_profit(priced, center, eval_sigma, t, lot, friction,
                         fat=True, r=r)
    pop_c = prob_of_profit(priced, chain.spot, fallback_iv, t, lot, 0.0,
                           fat=False, r=r)
    ev_raw, ev_std = payoff_moments(priced, chain.spot, eval_sigma, t, lot,
                                    tilt, r=r)
    ev = ev_raw - friction
    rr = (mp / ml) if (ml > 0 and mp != float("inf")) else (
        float("inf") if mp == float("inf") and ml > 0 else 0.0)

    return StrategyResult(
        name=name, legs=priced, lot_size=lot,
        net_premium=net_premium(priced),
        max_profit=mp, max_loss=ml,
        breakevens=breakevens(priced, chain.spot),
        pop_pct=pop, pop_classic_pct=pop_c,
        expected_value=ev, payoff_std=ev_std,
        rr_ratio=(0.0 if rr == float("inf") else rr),
        friction=friction,
        margin_estimate=margin_estimate(priced, chain.spot, lot, ml),
        greeks=position_greeks(priced, chain.spot, t, lot, fallback_iv, r),
        tags=tags or [], rationale=rationale, is_custom=is_custom,
        source=source)
