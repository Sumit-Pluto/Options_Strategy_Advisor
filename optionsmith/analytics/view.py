"""Market view — turns chain metrics into a tradeable stance.

Output is deliberately small and explicit:
    direction  bullish | bearish | neutral       (+ bias in [-1, 1])
    volatility iv_rich | iv_cheap | normal
    range      strong | weak                     (do the walls hold?)
Each carries the reasons that produced it, so a user can disagree with the
machine on the evidence rather than on faith. A user-supplied view always
overrides the inferred one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import ChainMetrics


@dataclass
class MarketView:
    bias: float = 0.0                 # -1 (max bearish) .. +1 (max bullish)
    direction: str = "neutral"
    volatility: str = "normal"        # iv_rich | iv_cheap | normal
    range_conviction: str = "weak"    # strong | weak
    target_move_pct: float = 0.0      # expected move used to tilt the density
    reasons: list[str] = field(default_factory=list)
    user_supplied: bool = False

    @property
    def tilt(self) -> float:
        """Fractional shift applied to the terminal-price median."""
        return self.target_move_pct / 100.0

    @property
    def vol_multiplier(self) -> float:
        """Expected REALISED vol as a fraction of IMPLIED vol.

        This is where volatility edge actually enters: if IV sits high in its
        own history, the market is likely over-pricing future movement, so
        the density used to value a structure should be NARROWER than the one
        the options are priced off (and wider when IV is cheap). Without this
        the advisor cannot tell a premium seller from a premium buyer — the
        IV regime would change nothing at all.
        """
        return {"iv_rich": 0.85, "iv_cheap": 1.15}.get(self.volatility, 1.0)

    def to_dict(self) -> dict:
        return {"bias": round(self.bias, 3), "direction": self.direction,
                "volatility": self.volatility,
                "range_conviction": self.range_conviction,
                "target_move_pct": round(self.target_move_pct, 2),
                "reasons": self.reasons, "user_supplied": self.user_supplied}


def infer(m: ChainMetrics, iv_percentile: float | None = None) -> MarketView:
    """Build a view from the chain's own evidence.

    Honest about mechanism: PCR extremes are read CONTRARIAN (crowded side),
    walls are read as magnets/barriers, and IV richness is judged against a
    supplied percentile when available — absolute IV levels mean nothing
    without the stock's own history.
    """
    v = MarketView()
    bias = 0.0

    # --- walls: headroom asymmetry ------------------------------------
    if m.call_wall and m.put_wall and m.spot > 0:
        up = (m.call_wall - m.spot) / m.spot
        dn = (m.spot - m.put_wall) / m.spot
        if up + dn > 0:
            asym = (up - dn) / (up + dn)          # +ve = more room upward
            bias += 0.35 * asym
            v.reasons.append(
                f"room to call wall {m.call_wall:g} is {up*100:.1f}% vs "
                f"{dn*100:.1f}% to put wall {m.put_wall:g}")

    # --- max pain pull -------------------------------------------------
    if m.max_pain and m.spot:
        pull = (m.max_pain - m.spot) / m.spot
        if abs(pull) > 0.005:
            bias += 0.20 * max(-1.0, min(1.0, pull * 20))
            v.reasons.append(
                f"max pain {m.max_pain:g} sits {pull*100:+.1f}% from spot "
                f"(expiry magnet, strongest in the final week)")

    # --- PCR extremes, read contrarian ---------------------------------
    if m.pcr_oi:
        if m.pcr_oi > 1.3:
            bias += 0.15
            v.reasons.append(f"PCR(OI) {m.pcr_oi:.2f} — put-heavy crowd, "
                             f"mild contrarian bullish tilt")
        elif 0 < m.pcr_oi < 0.6:
            bias -= 0.15
            v.reasons.append(f"PCR(OI) {m.pcr_oi:.2f} — call-heavy crowd, "
                             f"mild contrarian bearish tilt")

    # --- fresh OI builds -----------------------------------------------
    ce_build = sum(r["d_oi"] for r in m.oi_builds if r["right"] == "CE"
                   and r["strike"] >= m.spot)
    pe_build = sum(r["d_oi"] for r in m.oi_builds if r["right"] == "PE"
                   and r["strike"] <= m.spot)
    if ce_build or pe_build:
        tot = ce_build + pe_build
        if tot > 0:
            # OTM call writing is bearish pressure; OTM put writing bullish
            net = (pe_build - ce_build) / tot
            bias += 0.25 * net
            v.reasons.append(
                f"fresh OI: {ce_build:,} OTM call vs {pe_build:,} OTM put "
                f"contracts — {'put writers' if net > 0 else 'call writers'} "
                f"more aggressive")

    # --- skew ------------------------------------------------------------
    if m.iv_skew_25d is not None:
        if m.iv_skew_25d > 0.05:
            bias -= 0.10
            v.reasons.append(f"25Δ skew {m.iv_skew_25d*100:+.1f} vol pts — "
                             f"puts bid, downside protection in demand")
        elif m.iv_skew_25d < -0.02:
            bias += 0.10
            v.reasons.append(f"25Δ skew {m.iv_skew_25d*100:+.1f} vol pts — "
                             f"calls bid (unusual; upside chase)")

    v.bias = max(-1.0, min(1.0, bias))
    v.direction = ("bullish" if v.bias > 0.15 else
                   "bearish" if v.bias < -0.15 else "neutral")

    # --- volatility regime ---------------------------------------------
    if iv_percentile is not None:
        if iv_percentile >= 70:
            v.volatility = "iv_rich"
            v.reasons.append(f"ATM IV at {iv_percentile:.0f}th percentile of "
                             f"its own history — premium selling favoured")
        elif iv_percentile <= 30:
            v.volatility = "iv_cheap"
            v.reasons.append(f"ATM IV at {iv_percentile:.0f}th percentile — "
                             f"premium buying favoured")
    elif m.atm_iv:
        v.reasons.append(f"ATM IV {m.atm_iv*100:.1f}% — no own-history "
                         f"percentile supplied, so richness is UNJUDGED "
                         f"(absolute IV levels are not comparable across names)")

    # --- range conviction ------------------------------------------------
    strong = (m.call_concentration > 0.35 and m.put_concentration > 0.35)
    if strong and not m.oi_unwinds:
        v.range_conviction = "strong"
        v.reasons.append(
            f"OI concentrated at the walls (top-2 strikes hold "
            f"{m.call_concentration*100:.0f}% call / "
            f"{m.put_concentration*100:.0f}% put OI) and no major unwind — "
            f"range-bound structures favoured")
    elif m.oi_unwinds:
        v.reasons.append(
            f"{len(m.oi_unwinds)} strike(s) unwinding >30% — walls are "
            f"weakening; do not lean on the range")

    # expected move for the density tilt: bias scaled by a ~1 sigma move
    if m.atm_iv and m.dte:
        one_sigma = m.atm_iv * (max(m.dte, 1) / 365.0) ** 0.5 * 100
        v.target_move_pct = v.bias * one_sigma * 0.6
    return v


def from_user(direction: str | None = None, volatility: str | None = None,
              target_move_pct: float | None = None,
              base: MarketView | None = None) -> MarketView:
    """Override the inferred view with the user's own stance."""
    v = base or MarketView()
    v.user_supplied = True
    if direction:
        v.direction = direction
        v.bias = {"bullish": 0.6, "bearish": -0.6, "neutral": 0.0}.get(
            direction, v.bias)
        # the inferred tilt belonged to the inferred bias — recompute it, or a
        # user who says "neutral" keeps the machine's directional lean
        if target_move_pct is None:
            v.target_move_pct = v.bias * abs(v.target_move_pct) / 0.6 \
                if v.target_move_pct else v.bias * 3.0
        v.reasons.insert(0, f"user view: {direction}")
    if volatility:
        v.volatility = volatility
        v.reasons.insert(0, f"user volatility view: {volatility}")
    if target_move_pct is not None:
        v.target_move_pct = target_move_pct
        v.reasons.insert(0, f"user target move: {target_move_pct:+.1f}%")
    return v
