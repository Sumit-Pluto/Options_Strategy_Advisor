"""The classic option-strategy library — 28 named structures.

Each entry knows: which view it expresses, how to lay its strikes on a real
chain, and what its practical caveat is. `build_all()` instantiates every
strategy whose strikes exist on the given chain, so the advisor can rank the
textbook menu honestly against generated custom structures.

Strike selection is delta-aware where that is how the structure is actually
traded (e.g. 25Δ strangles), otherwise step-based around ATM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..analytics.metrics import call_wall, max_pain, put_wall
from ..core.mathx import bs_delta
from ..core.models import Chain, Leg

# view tags used for filtering
BULL, BEAR, NEUTRAL, VOL_UP, VOL_DOWN = ("bullish", "bearish", "neutral",
                                         "vol_up", "vol_down")


@dataclass
class Recipe:
    key: str
    name: str
    views: tuple[str, ...]
    build: Callable[[Chain], list[Leg] | None]
    caveat: str = ""
    family: str = "vertical"


# ── strike helpers ─────────────────────────────────────────────────────
def _k(chain: Chain, n: int, is_call: bool = True) -> float | None:
    """Strike n steps from ATM (n>0 = OTM for calls / ITM for puts)."""
    ks = sorted({q.strike for q in chain.quotes if q.is_call == is_call})
    if not ks:
        return None
    atm = min(ks, key=lambda k: abs(k - chain.spot))
    i = ks.index(atm) + n
    return ks[i] if 0 <= i < len(ks) else None


DELTA_TOLERANCE = 0.05      # how far a "25-delta" strike may actually miss


def _by_delta(chain: Chain, target: float, is_call: bool,
              tol: float = DELTA_TOLERANCE) -> float | None:
    """Strike whose delta is closest to `target` (e.g. 0.25 / -0.25).

    Returns None when even the closest listed strike misses by more than `tol`,
    which makes the recipe fail to build and drop out of the menu. Without that
    guard the miss is SILENT and the structure renames itself: measured on a
    +/-5 strike window at 60 DTE, the "20-delta short strangle" was really a
    39-delta strangle — twice the intended delta, far more premium and far more
    assignment risk, still carrying the 20-delta label. Not building is the
    honest outcome; a mislabelled structure is worse than a missing one.

    Delta is taken at the chain's own carry, not the fixed default: the IVs
    were inverted against that forward, so selecting on a different one
    reintroduces the very inconsistency `calibrate_carry` exists to remove.
    """
    t = chain.t_years
    best, err = None, 1e9
    for q in chain.quotes:
        if q.is_call != is_call or not q.iv:
            continue
        d = bs_delta(q.is_call, chain.spot, q.strike, t, q.iv, chain.r)
        e = abs(d - target)
        if e < err:
            best, err = q.strike, e
    return best if err <= tol else None


def _from_anchor(chain: Chain, anchor: float | None, n: int,
                 is_call: bool) -> float | None:
    """Strike `n` steps from a STRUCTURAL anchor (a wall, max pain) rather than
    from ATM.

    This is the difference between "sell the 2nd strike out" and "sell where
    the market says the wall is". The offset recipes place strikes at a fixed
    distance from spot regardless of what the chain says; these place them at
    the level the open interest actually marks.
    """
    if anchor is None:
        return None
    ks = sorted({q.strike for q in chain.quotes if q.is_call == is_call})
    if not ks:
        return None
    i = min(range(len(ks)), key=lambda j: abs(ks[j] - anchor)) + n
    return ks[i] if 0 <= i < len(ks) else None


def _walls(chain: Chain) -> tuple[float | None, float | None]:
    """(call wall, put wall) — but only when they bracket spot sanely.

    A wall on the wrong side of spot is not a barrier, it is deep-ITM
    inventory. Anchoring to it would write a short IN the money, so the
    structure refuses to build instead."""
    cw, pw = call_wall(chain), put_wall(chain)
    if cw is None or pw is None or pw >= cw:
        return None, None
    return cw, pw


def _legs(*specs) -> list[Leg]:
    """(strike, is_call, side, qty) tuples -> Legs; None strike aborts."""
    out = []
    for strike, is_call, side, qty in specs:
        if strike is None:
            return []
        out.append(Leg(strike, is_call, side, qty))
    return out


# ── single leg ─────────────────────────────────────────────────────────
def _long_call(c):    return _legs((_k(c, 1), True, 1, 1))
def _long_put(c):     return _legs((_k(c, -1, False), False, 1, 1))
def _short_call(c):   return _legs((_k(c, 2), True, -1, 1))
def _short_put(c):    return _legs((_k(c, -2, False), False, -1, 1))
def _covered_call(c): return _legs((_k(c, 2), True, -1, 1))


# ── verticals ──────────────────────────────────────────────────────────
def _bull_call_spread(c):
    return _legs((_k(c, 0), True, 1, 1), (_k(c, 2), True, -1, 1))


def _bear_call_spread(c):
    return _legs((_k(c, 1), True, -1, 1), (_k(c, 3), True, 1, 1))


def _bull_put_spread(c):
    return _legs((_k(c, -1, False), False, -1, 1), (_k(c, -3, False), False, 1, 1))


def _bear_put_spread(c):
    return _legs((_k(c, 0, False), False, 1, 1), (_k(c, -2, False), False, -1, 1))


# ── volatility structures ──────────────────────────────────────────────
def _long_straddle(c):
    k = _k(c, 0)
    return _legs((k, True, 1, 1), (k, False, 1, 1))


def _short_straddle(c):
    k = _k(c, 0)
    return _legs((k, True, -1, 1), (k, False, -1, 1))


def _long_strangle(c):
    return _legs((_by_delta(c, 0.25, True), True, 1, 1),
                 (_by_delta(c, -0.25, False), False, 1, 1))


def _short_strangle(c):
    return _legs((_by_delta(c, 0.20, True), True, -1, 1),
                 (_by_delta(c, -0.20, False), False, -1, 1))


# ── wings ──────────────────────────────────────────────────────────────
def _iron_condor(c):
    return _legs((_k(c, -4, False), False, 1, 1), (_k(c, -2, False), False, -1, 1),
                 (_k(c, 2), True, -1, 1), (_k(c, 4), True, 1, 1))


def _reverse_iron_condor(c):
    return _legs((_k(c, -4, False), False, -1, 1), (_k(c, -2, False), False, 1, 1),
                 (_k(c, 2), True, 1, 1), (_k(c, 4), True, -1, 1))


def _iron_butterfly(c):
    k = _k(c, 0)
    return _legs((_k(c, -3, False), False, 1, 1), (k, False, -1, 1),
                 (k, True, -1, 1), (_k(c, 3), True, 1, 1))


def _call_butterfly(c):
    return _legs((_k(c, 0), True, 1, 1), (_k(c, 2), True, -1, 2),
                 (_k(c, 4), True, 1, 1))


def _put_butterfly(c):
    return _legs((_k(c, 0, False), False, 1, 1), (_k(c, -2, False), False, -1, 2),
                 (_k(c, -4, False), False, 1, 1))


def _broken_wing_call_fly(c):
    """Skipped-strike fly: cheaper (often a credit), risk shifted to one side."""
    return _legs((_k(c, 0), True, 1, 1), (_k(c, 2), True, -1, 2),
                 (_k(c, 5), True, 1, 1))


def _call_condor(c):
    return _legs((_k(c, 0), True, 1, 1), (_k(c, 1), True, -1, 1),
                 (_k(c, 3), True, -1, 1), (_k(c, 4), True, 1, 1))


# ── ratio / directional-with-a-twist ───────────────────────────────────
def _call_ratio_spread(c):
    return _legs((_k(c, 0), True, 1, 1), (_k(c, 2), True, -1, 2))


def _put_ratio_spread(c):
    return _legs((_k(c, 0, False), False, 1, 1), (_k(c, -2, False), False, -1, 2))


def _call_backspread(c):
    return _legs((_k(c, 0), True, -1, 1), (_k(c, 2), True, 1, 2))


def _put_backspread(c):
    return _legs((_k(c, 0, False), False, -1, 1), (_k(c, -2, False), False, 1, 2))


def _risk_reversal(c):
    return _legs((_by_delta(c, -0.25, False), False, -1, 1),
                 (_by_delta(c, 0.25, True), True, 1, 1))


def _collar_synthetic(c):
    """Long stock proxy is out of scope; the option-only equivalent is the
    combo: long ATM call + short ATM put (synthetic long)."""
    k = _k(c, 0)
    return _legs((k, True, 1, 1), (k, False, -1, 1))


def _jade_lizard(c):
    """Short put + short call spread — no upside risk when the credit exceeds
    the call-spread width."""
    return _legs((_k(c, -2, False), False, -1, 1),
                 (_k(c, 2), True, -1, 1), (_k(c, 4), True, 1, 1))


def _strap(c):
    k = _k(c, 0)
    return _legs((k, True, 1, 2), (k, False, 1, 1))


def _strip(c):
    k = _k(c, 0)
    return _legs((k, True, 1, 1), (k, False, 1, 2))


def _guts(c):
    return _legs((_k(c, -2), True, 1, 1), (_k(c, 2, False), False, 1, 1))


def _box(c):
    """Arbitrage check structure — value must equal the strike width."""
    k1, k2 = _k(c, 0), _k(c, 2)
    return _legs((k1, True, 1, 1), (k2, True, -1, 1),
                 (k2, False, 1, 1), (k1, False, -1, 1))


# ── wall-anchored: strikes placed where the CHAIN says, not at fixed offsets ──
def _wall_iron_condor(c):
    """Shorts AT the walls, wings one strike beyond.

    The range trade the metrics are actually pointing at: if the market has
    written its resistance at 1470 and support at 1330, that is where the
    short strikes belong — not at ATM+/-2, which ignores the read entirely."""
    cw, pw = _walls(c)
    return _legs((_from_anchor(c, pw, -1, False), False, 1, 1),
                 (_from_anchor(c, pw, 0, False), False, -1, 1),
                 (_from_anchor(c, cw, 0, True), True, -1, 1),
                 (_from_anchor(c, cw, 1, True), True, 1, 1))


def _wall_bull_put_spread(c):
    """Sell the put wall, buy two strikes below it."""
    _, pw = _walls(c)
    return _legs((_from_anchor(c, pw, 0, False), False, -1, 1),
                 (_from_anchor(c, pw, -2, False), False, 1, 1))


def _wall_bear_call_spread(c):
    """Sell the call wall, buy two strikes above it."""
    cw, _ = _walls(c)
    return _legs((_from_anchor(c, cw, 0, True), True, -1, 1),
                 (_from_anchor(c, cw, 2, True), True, 1, 1))


def _wall_short_strangle(c):
    """Naked at both walls — the highest-POP expression of a range view, and
    the one that loses the most when the range fails. Undefined risk."""
    cw, pw = _walls(c)
    return _legs((_from_anchor(c, cw, 0, True), True, -1, 1),
                 (_from_anchor(c, pw, 0, False), False, -1, 1))


def _pin_butterfly(c):
    """Body at MAX PAIN — a butterfly is a bet on where price finishes, and
    max pain is the chain's own estimate of exactly that."""
    mp = max_pain(c)
    return _legs((_from_anchor(c, mp, -2, True), True, 1, 1),
                 (_from_anchor(c, mp, 0, True), True, -1, 2),
                 (_from_anchor(c, mp, 2, True), True, 1, 1))


def _wall_jade_lizard(c):
    """Short put at support, short call spread at resistance."""
    cw, pw = _walls(c)
    return _legs((_from_anchor(c, pw, 0, False), False, -1, 1),
                 (_from_anchor(c, cw, 0, True), True, -1, 1),
                 (_from_anchor(c, cw, 2, True), True, 1, 1))


CATALOG: list[Recipe] = [
    Recipe("long_call", "long call", (BULL, VOL_UP), _long_call,
           "Unlimited upside, but theta bleeds daily and you need the move to "
           "beat the premium — most long calls expire worthless.", "single"),
    Recipe("long_put", "long put", (BEAR, VOL_UP), _long_put,
           "Same theta bleed as a long call; puts also carry richer IV (skew), "
           "so you overpay in calm markets.", "single"),
    Recipe("short_call", "short call (naked)", (BEAR, VOL_DOWN), _short_call,
           "UNLIMITED loss and heavy margin. Only for accounts that can hold "
           "the stock; most brokers restrict it.", "single"),
    Recipe("short_put", "short put (cash-secured)", (BULL, VOL_DOWN), _short_put,
           "Loss to zero on the stock; treat it as a commitment to own the "
           "shares at the strike.", "single"),
    Recipe("bull_call_spread", "bull call spread", (BULL,), _bull_call_spread,
           "Defined risk, capped reward — the workhorse bullish debit trade.",
           "vertical"),
    Recipe("bear_call_spread", "bear call spread", (BEAR, VOL_DOWN),
           _bear_call_spread,
           "Credit trade; wins on time and on the stock not rallying.",
           "vertical"),
    Recipe("bull_put_spread", "bull put spread", (BULL, VOL_DOWN),
           _bull_put_spread,
           "Credit trade; the classic 'sell fear' structure below support.",
           "vertical"),
    Recipe("bear_put_spread", "bear put spread", (BEAR,), _bear_put_spread,
           "Defined-risk bearish debit; cheaper than a naked long put.",
           "vertical"),
    Recipe("long_straddle", "long straddle", (VOL_UP,), _long_straddle,
           "Needs a move LARGER than the combined premium — buying it before "
           "a known event usually means buying inflated IV.", "volatility"),
    Recipe("short_straddle", "short straddle", (NEUTRAL, VOL_DOWN),
           _short_straddle,
           "Unlimited risk both ways; the single most dangerous common "
           "structure. Margin-heavy.", "volatility"),
    Recipe("long_strangle", "long strangle (25Δ)", (VOL_UP,), _long_strangle,
           "Cheaper than a straddle, needs an even bigger move.", "volatility"),
    Recipe("short_strangle", "short strangle (20Δ)", (NEUTRAL, VOL_DOWN),
           _short_strangle,
           "Unlimited risk; high win rate that hides rare large losses.",
           "volatility"),
    Recipe("iron_condor", "iron condor", (NEUTRAL, VOL_DOWN), _iron_condor,
           "Defined-risk range trade. Wins ~70-80% of the time and loses "
           "multiples of the credit when it fails — check the RR, not the POP.",
           "wing"),
    Recipe("reverse_iron_condor", "reverse iron condor", (VOL_UP,),
           _reverse_iron_condor,
           "Defined-risk breakout play; pays only on a decisive move.", "wing"),
    Recipe("iron_butterfly", "iron butterfly", (NEUTRAL, VOL_DOWN),
           _iron_butterfly,
           "Bigger credit than a condor, much narrower profit zone — a pin bet.",
           "wing"),
    Recipe("call_butterfly", "long call butterfly", (NEUTRAL,), _call_butterfly,
           "Cheap lottery on the stock finishing at the body strike.", "wing"),
    Recipe("put_butterfly", "long put butterfly", (NEUTRAL,), _put_butterfly,
           "Mirror of the call fly; use the side with better liquidity.", "wing"),
    Recipe("bw_call_fly", "broken-wing call butterfly", (BULL, NEUTRAL),
           _broken_wing_call_fly,
           "Often a credit with no downside risk, but the skipped wing leaves "
           "a larger loss zone above — know where it is.", "wing"),
    Recipe("call_condor", "long call condor", (NEUTRAL,), _call_condor,
           "All-call version of the range trade; four legs of friction.", "wing"),
    Recipe("call_ratio_spread", "call ratio spread (1x2)", (NEUTRAL, VOL_DOWN),
           _call_ratio_spread,
           "Extra short call means UNLIMITED upside risk above the wing.",
           "ratio"),
    Recipe("put_ratio_spread", "put ratio spread (1x2)", (NEUTRAL, VOL_DOWN),
           _put_ratio_spread,
           "Large loss if the stock collapses through the short strikes.",
           "ratio"),
    Recipe("call_backspread", "call ratio backspread", (BULL, VOL_UP),
           _call_backspread,
           "Pays on a violent rally; loses most at the long strike on expiry.",
           "ratio"),
    Recipe("put_backspread", "put ratio backspread", (BEAR, VOL_UP),
           _put_backspread,
           "Crash insurance that can be entered for a credit.", "ratio"),
    Recipe("risk_reversal", "risk reversal (25Δ)", (BULL,), _risk_reversal,
           "Sell the put to fund the call — a leveraged bullish bet with "
           "stock-like downside.", "combo"),
    Recipe("synthetic_long", "synthetic long (combo)", (BULL,),
           _collar_synthetic,
           "Replicates the stock; full downside, near-zero premium outlay.",
           "combo"),
    Recipe("jade_lizard", "jade lizard", (NEUTRAL, BULL, VOL_DOWN), _jade_lizard,
           "No upside risk WHEN the credit exceeds the call-spread width — "
           "the engine checks that for you.", "combo"),
    Recipe("strap", "strap (2 calls + 1 put)", (VOL_UP, BULL), _strap,
           "Straddle tilted bullish; costs more than either single leg.",
           "volatility"),
    Recipe("strip", "strip (1 call + 2 puts)", (VOL_UP, BEAR), _strip,
           "Straddle tilted bearish.", "volatility"),
    Recipe("guts", "long guts", (VOL_UP,), _guts,
           "ITM strangle — more intrinsic, wider spreads, rarely optimal vs a "
           "plain strangle.", "volatility"),
    # --- wall-anchored variants: same shapes, structural strikes -------
    Recipe("wall_iron_condor", "iron condor @ walls", (NEUTRAL, VOL_DOWN),
           _wall_iron_condor,
           "Shorts sit at the OI walls instead of a fixed offset, so the "
           "profit zone is the range the market itself has written. Higher POP "
           "than the ATM-offset condor — and a smaller credit for it.", "wall"),
    Recipe("wall_bull_put_spread", "bull put spread @ put wall",
           (BULL, NEUTRAL, VOL_DOWN), _wall_bull_put_spread,
           "Sells the strike with the most put OI below spot — the level the "
           "market is defending. The wall holding is the whole thesis.", "wall"),
    Recipe("wall_bear_call_spread", "bear call spread @ call wall",
           (BEAR, NEUTRAL, VOL_DOWN), _wall_bear_call_spread,
           "Sells resistance. If the call wall is written by hedged supply it "
           "holds; if it is a breakout target it does not.", "wall"),
    Recipe("wall_short_strangle", "short strangle @ walls",
           (NEUTRAL, VOL_DOWN), _wall_short_strangle,
           "The highest-POP way to express a range and the one that loses most "
           "when the range fails. UNLIMITED risk both sides.", "wall"),
    Recipe("pin_butterfly", "butterfly @ max pain", (NEUTRAL,), _pin_butterfly,
           "Body on the chain's own pin estimate. Max pain is a magnet only "
           "in the final week, and it is an artefact of OI, not a forecast.",
           "wall"),
    Recipe("wall_jade_lizard", "jade lizard @ walls", (NEUTRAL, BULL, VOL_DOWN),
           _wall_jade_lizard,
           "Short put at support, short call spread at resistance — no upside "
           "risk when the credit exceeds the call-spread width.", "wall"),
    Recipe("box", "box spread", (NEUTRAL,), _box,
           "Pure arbitrage check: fair value = strike width. A price far from "
           "that is a data error far more often than free money.", "arb"),
]

BY_KEY = {r.key: r for r in CATALOG}


def build_all(chain: Chain, views: tuple[str, ...] | None = None
              ) -> list[tuple[Recipe, list[Leg]]]:
    """Instantiate every recipe whose strikes exist on this chain."""
    out = []
    for r in CATALOG:
        if views and not set(r.views) & set(views):
            continue
        try:
            legs = r.build(chain)
        except Exception:
            legs = []
        if legs and all(chain.get(l.strike, l.is_call) for l in legs):
            out.append((r, legs))
    return out
