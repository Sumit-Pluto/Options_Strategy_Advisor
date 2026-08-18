"""Scan one symbol, and rank across symbols.

Two rules here that differ from the single-chain advisor, both of them
consequences of scanning MANY chains instead of one:

  * SCREENING POP IS UNTILTED. `advise()` shifts the terminal density by the
    view before integrating, so its POP is conditional on a directional
    opinion — and in a universe scan every symbol infers its OWN opinion. A
    "POP > 60%" filter over view-tilted numbers compares quantities that each
    carry a different assumption. The screen therefore re-computes POP at zero
    tilt so one slider means one thing across 180 names.

  * CROSS-SYMBOL RANKING NEVER USES EV. EV is biased by the smile (legs priced
    off the smile, valued at one sigma), and the bias grows with smile
    steepness. Sorting a universe by EV therefore surfaces the steepest-smile
    chains first — it ranks names by how wrong the model is on them. The screen
    ranks on POP x RR, ordinary trader expectancy, which touches neither the
    smile nor the biased integral. EV is still reported per structure, because
    within one chain it is informative; it just does not decide the order.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..advisor import advise
from ..analytics.metrics import liquid_smile_spread
from ..chain.loaders import fill_missing_ivs, from_gateway, synthetic
from ..core.models import Chain
from ..core.payoff import COST_PER_LEG, _sigma_for, prob_of_profit
from . import store

RR_CAP = 3.0


@dataclass
class ScanConfig:
    pop_min: float = 50.0
    pop_max: float = 100.0
    source: str = "live"              # live | synthetic
    strikes: int = 8                  # +/- around ATM; 8 covers all 30 recipes
    exchange: str = "NSE"
    max_loss: float = 25_000.0
    max_legs: int = 4
    top_per_symbol: int = 3
    include_indices: bool = False
    min_dte: int = 5                  # below this, roll to the next expiry
    record_iv: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SymbolResult:
    symbol: str
    ok: bool = True
    error: str = ""
    spot: float = 0.0
    dte: int = 0
    atm_iv_pct: float | None = None
    iv_percentile: float | None = None
    iv_sessions: int = 0
    smile_spread: float | None = None
    opportunities: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _untilted_pop(res_legs, chain: Chain, fallback_iv: float) -> float:
    """POP with no directional tilt — the comparable number for a screen."""
    friction = COST_PER_LEG * sum(l.qty for l in res_legs)
    sigma = _sigma_for(res_legs, fallback_iv)
    return prob_of_profit(res_legs, chain.spot, sigma, chain.t_years,
                          chain.lot_size, friction, fat=True, r=chain.r)


def screen_score(pop_pct: float, rr_ratio) -> float:
    """Cross-symbol ranking key: P(win) x reward:risk. Deliberately EV-free.

    rr_ratio arrives as None for unbounded upside (see StrategyResult.to_dict),
    which is the BEST payoff shape, not the worst — it takes the cap.
    """
    rr = RR_CAP if rr_ratio is None else min(float(rr_ratio), RR_CAP)
    return (pop_pct / 100.0) * rr


def load_chain(symbol: str, cfg: ScanConfig) -> Chain:
    if cfg.source == "synthetic":
        # deterministic per symbol so a demo scan is reproducible
        seed = abs(hash(symbol)) % 997
        spot = 500 + (seed % 40) * 55
        return synthetic(symbol, spot=float(spot), dte=30, lot_size=250,
                         n_strikes=2 * cfg.strikes + 1, seed=seed)
    return from_gateway(symbol, exchange=cfg.exchange, count=cfg.strikes)


def scan_symbol(symbol: str, cfg: ScanConfig) -> SymbolResult:
    """One symbol end to end. Never raises — a dead symbol must not stop a scan."""
    out = SymbolResult(symbol=symbol)
    try:
        chain = load_chain(symbol, cfg)
        fill_missing_ivs(chain)
        out.spot, out.dte = chain.spot, chain.days_to_expiry
        out.smile_spread = liquid_smile_spread(chain)

        atm_q = chain.get(chain.atm, True)
        atm_iv = atm_q.iv if atm_q and atm_q.iv else None
        if atm_iv:
            out.atm_iv_pct = round(atm_iv * 100, 1)
            if cfg.record_iv:
                # free: the number is already computed, and this is the only
                # way an IV percentile ever comes to exist
                store.record_iv(symbol, atm_iv, smile=out.smile_spread,
                                spot=chain.spot, dte=chain.days_to_expiry)
            pct, n = store.iv_percentile(symbol, atm_iv)
            out.iv_percentile, out.iv_sessions = pct, n

        if out.dte < cfg.min_dte:
            out.warnings.append(
                f"{out.dte}d to expiry — pinning dominates; roll to next month")

        # Pre-gate the advisor LOOSER than the screen asks, then filter exactly
        # on the untilted number. Gating at 0 lets the advisor's EV-ranked
        # top-N fill with low-POP shapes before the screen ever sees the
        # high-POP ones — a "POP > 70%" scan would then come back empty on a
        # chain full of 80% condors. Gating at exactly pop_min instead drops
        # structures whose TILTED POP dips just under the untilted one.
        pre_pop = max(0.0, cfg.pop_min - 10.0)
        rep = advise(chain, iv_percentile=out.iv_percentile,
                     top_n=max(cfg.top_per_symbol * 8, 24),
                     max_legs=cfg.max_legs, max_loss_rupees=cfg.max_loss,
                     min_pop=pre_pop)
        out.warnings.extend(rep.warnings)

        fallback_iv = atm_iv or 0.30
        kept: list[dict] = []
        for r, d in zip(_results_of(rep), rep.recommendations):
            pop = _untilted_pop(r, chain, fallback_iv) if r else d["pop_pct"]
            if not (cfg.pop_min <= pop <= cfg.pop_max):
                continue
            row = dict(d)
            row["pop_screen_pct"] = round(pop, 1)
            row["screen_score"] = round(screen_score(pop, d["rr_ratio"]), 4)
            row["symbol"] = symbol
            row["spot"] = chain.spot
            row["dte"] = chain.days_to_expiry
            kept.append(row)
        kept.sort(key=lambda x: -x["screen_score"])
        out.opportunities = kept[:cfg.top_per_symbol]
    except Exception as e:                       # noqa: BLE001 - report, never abort
        out.ok = False
        out.error = f"{type(e).__name__}: {e}"
    return out


def _results_of(rep) -> list:
    """The advisor returns dicts; the untilted POP needs the Leg objects back.

    advise() does not hand back its StrategyResult objects, so rebuild the legs
    from the dict. Cheap, and it keeps the advisor's public shape unchanged.
    """
    from ..core.models import Leg
    out = []
    for d in rep.recommendations:
        try:
            out.append([Leg(float(l["strike"]),
                            l["right"] == "CE",
                            1 if l["action"] == "BUY" else -1,
                            int(l["qty"]), float(l["price"]),
                            (l["iv"] / 100.0) if l.get("iv") else None)
                        for l in d["legs"]])
        except Exception:
            out.append(None)
    return out
