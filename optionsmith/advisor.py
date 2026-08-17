"""The Strategy Advisor — one call in, a complete recommendation out.

    from optionsmith import advise, synthetic
    report = advise(synthetic("RELIANCE", spot=1400))

Pipeline (mirrors the module docs):

    chain -> metrics -> view -> [classic library | custom generator]
          -> unified ranking -> report{metrics, view, recommendations, menu}

Ranking is view-conditional expected value per rupee of risk, so a classic
and a generated structure compete on identical terms; the report always
shows both so the user can see whether the custom shape actually beat the
textbook one.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .analytics import metrics as metrics_mod
from .analytics.view import MarketView, from_user, infer
from .chain.loaders import fill_missing_ivs
from .core.models import Chain, StrategyResult
from .core.payoff import evaluate
from .strategies import library
from .strategies.generator import generate, score_of

VIEW_TAGS = {
    "bullish": ("bullish",), "bearish": ("bearish",), "neutral": ("neutral",),
}


@dataclass
class AdviceReport:
    symbol: str
    spot: float
    expiry: str
    dte: int
    lot_size: int
    generated_at: str
    metrics: dict
    view: dict
    recommendations: list[dict] = field(default_factory=list)
    classic_menu: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _view_tags(v: MarketView) -> tuple[str, ...]:
    tags: list[str] = [v.direction]
    if v.volatility == "iv_rich":
        tags.append("vol_down")
    elif v.volatility == "iv_cheap":
        tags.append("vol_up")
    if v.range_conviction == "strong":
        tags.append("neutral")
    return tuple(dict.fromkeys(tags))


def build_menu(chain: Chain, view: MarketView,
               restrict_to_view: bool = True,
               allow_undefined_risk: bool = True) -> list[StrategyResult]:
    """Price every classic structure the chain can express.

    Undefined-risk classics (naked short call, short straddle/strangle) stay
    in the MENU for reference but are excluded from the ranked
    recommendations unless the caller explicitly allows them.
    """
    tags = _view_tags(view) if restrict_to_view else None
    out: list[StrategyResult] = []
    for recipe, legs in library.build_all(chain, tags):
        try:
            res = evaluate(legs, chain, name=recipe.name, tilt=view.tilt,
                           rationale=recipe.caveat, vol_mult=view.vol_multiplier,
                           tags=[recipe.family, *recipe.views], source="library")
        except Exception:
            continue
        res.is_custom = False
        res.score = score_of(res)          # identical scoring to generated
        out.append(res)
    out.sort(key=lambda r: -r.score)
    return out


def advise(chain: Chain, *, view: MarketView | None = None,
           user_direction: str | None = None,
           user_volatility: str | None = None,
           user_target_move_pct: float | None = None,
           iv_percentile: float | None = None,
           top_n: int = 5, max_legs: int = 4,
           max_loss_rupees: float = 25_000.0,
           min_pop: float = 25.0, min_rr: float = 0.15,
           include_menu: bool = True,
           restrict_menu_to_view: bool = True) -> AdviceReport:
    """Analyse a chain and recommend structures.

    A user view (direction / volatility / target move) always overrides the
    inferred one — the advisor is a calculator for YOUR thesis first, and a
    read of the chain's own evidence second.
    """
    fill_missing_ivs(chain)
    m = metrics_mod.compute(chain)

    v = view or infer(m, iv_percentile=iv_percentile)
    if any(x is not None for x in (user_direction, user_volatility,
                                   user_target_move_pct)):
        v = from_user(user_direction, user_volatility, user_target_move_pct,
                      base=v)

    custom = generate(chain, v, top_n=top_n, max_legs=max_legs,
                      max_loss_rupees=max_loss_rupees, min_pop=min_pop,
                      min_rr=min_rr)
    menu = build_menu(chain, v, restrict_menu_to_view) if include_menu else []

    # unified ranking: classics and customs compete on identical terms,
    # under the same risk gates the generator applies to itself
    eligible = [r for r in menu
                if (r.max_loss != float("inf")) and r.pop_pct >= min_pop
                and r.max_loss <= max_loss_rupees]
    merged: list[StrategyResult] = sorted(custom + eligible,
                                          key=lambda r: -r.score)[:top_n]

    warnings = list(m.notes)
    if not custom:
        warnings.append("generator found no structure inside the risk budget "
                        "— widen max_loss or relax the POP/RR floors")
    if v.volatility == "normal" and iv_percentile is None:
        warnings.append("no IV percentile supplied: premium richness is "
                        "unjudged, so credit-vs-debit preference is neutral")
    unbounded = [r.name for r in merged if r.max_loss == float("inf")]
    if unbounded:
        warnings.append(f"undefined-risk structures present: {unbounded}")

    return AdviceReport(
        symbol=chain.symbol, spot=chain.spot, expiry=chain.expiry.isoformat(),
        dte=chain.days_to_expiry, lot_size=chain.lot_size,
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        metrics=m.to_dict(), view=v.to_dict(),
        recommendations=[r.to_dict() for r in merged],
        classic_menu=[r.to_dict() for r in menu],
        warnings=warnings)


def build_named(chain: Chain, key: str,
                view: MarketView | None = None) -> StrategyResult:
    """Price ONE named classic structure (e.g. 'iron_condor') on this chain."""
    recipe = library.BY_KEY.get(key)
    if recipe is None:
        raise KeyError(f"unknown strategy '{key}'. "
                       f"Known: {', '.join(sorted(library.BY_KEY))}")
    fill_missing_ivs(chain)
    legs = recipe.build(chain)
    if not legs:
        raise ValueError(f"'{key}' cannot be built on this chain "
                         f"(missing strikes)")
    tilt = view.tilt if view else 0.0
    res = evaluate(legs, chain, name=recipe.name, tilt=tilt,
                   rationale=recipe.caveat,
                   vol_mult=view.vol_multiplier if view else 1.0,
                   tags=[recipe.family, *recipe.views], source="library")
    res.is_custom = False
    return res
