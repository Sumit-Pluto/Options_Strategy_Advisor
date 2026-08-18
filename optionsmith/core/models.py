"""Core data types — Leg, OptionQuote, Chain, StrategyResult."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .mathx import RISK_FREE


@dataclass
class OptionQuote:
    """One tradable option contract from the chain."""
    strike: float
    is_call: bool
    ltp: float                      # last / mid reference price per share
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0                     # open interest (contracts)
    prev_oi: int = 0
    volume: int = 0
    iv: float | None = None         # decimal (0.32 = 32%), inverted if absent

    @property
    def right(self) -> str:
        return "CE" if self.is_call else "PE"

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0 and self.ask >= self.bid:
            return 0.5 * (self.bid + self.ask)
        return self.ltp

    @property
    def spread_pct(self) -> float:
        """Round-trip spread as % of mid — the liquidity gate everyone skips."""
        if self.bid > 0 and self.ask > self.bid and self.mid > 0:
            return (self.ask - self.bid) / self.mid * 100.0
        return 0.0

    def exec_price(self, side: int) -> float:
        """EXECUTABLE price: buy at ask, sell at bid (falls back to ltp)."""
        if side > 0:
            return self.ask if self.ask > 0 else self.ltp
        return self.bid if self.bid > 0 else self.ltp

    @property
    def d_oi(self) -> int:
        return int(self.oi - self.prev_oi)

    @property
    def d_oi_pct(self) -> float:
        return (self.d_oi / self.prev_oi * 100.0) if self.prev_oi > 0 else 0.0


@dataclass
class Chain:
    """A single-expiry option chain plus its underlying context."""
    symbol: str
    spot: float
    expiry: dt.date
    lot_size: int
    quotes: list[OptionQuote] = field(default_factory=list)
    asof: dt.date | None = None
    source: str = "unknown"
    carry_rate: float | None = None
    """Annualised carry the chain itself implies, from put-call parity.

    Every model here prices and integrates off `spot` with a drift of `r`, which
    makes its forward `spot*e^(r*t)`. That is the measure the options are
    actually quoted under — so if `r` is wrong, the forward is wrong, and the
    error does not cancel: it pushes call IVs up and put IVs down (a skew that
    is not there), and it makes ITM options look mispriced against intrinsic.

    Indian stock and index options are quoted off the FUTURE, whose basis over
    cash reflects financing and dividends, not the repo rate. Measured on live
    RELIANCE the chain implied ~13%/yr against the assumed 7% — enough that the
    advisor's top three structures were all the same artefact.

    None means "not calibrated" and everything falls back to RISK_FREE, so a
    synthetic or hand-written chain behaves exactly as before.
    """

    # -------------------------------------------------- derived accessors
    @property
    def r(self) -> float:
        """The carry to price and integrate at. Calibrated when the chain could
        tell us, the fixed default when it could not."""
        return RISK_FREE if self.carry_rate is None else self.carry_rate

    @property
    def days_to_expiry(self) -> int:
        base = self.asof or dt.date.today()
        return max(0, (self.expiry - base).days)

    @property
    def t_years(self) -> float:
        return max(self.days_to_expiry, 0.5) / 365.0

    @property
    def strikes(self) -> list[float]:
        return sorted({q.strike for q in self.quotes})

    @property
    def strike_step(self) -> float:
        ks = self.strikes
        if len(ks) < 2:
            return max(1.0, round(self.spot * 0.01))
        diffs = sorted(round(b - a, 4) for a, b in zip(ks, ks[1:]))
        return diffs[len(diffs) // 2] or 1.0

    @property
    def atm(self) -> float:
        return min(self.strikes, key=lambda k: abs(k - self.spot)) \
            if self.strikes else self.spot

    def get(self, strike: float, is_call: bool) -> OptionQuote | None:
        for q in self.quotes:
            if abs(q.strike - strike) < 1e-6 and q.is_call == is_call:
                return q
        return None

    def calls(self) -> list[OptionQuote]:
        return sorted([q for q in self.quotes if q.is_call], key=lambda q: q.strike)

    def puts(self) -> list[OptionQuote]:
        return sorted([q for q in self.quotes if not q.is_call], key=lambda q: q.strike)

    def implied_forward(self, band: float = 0.10,
                        min_estimates: int = 3) -> tuple[float | None, int]:
        """(forward, n) implied by this chain's own put-call parity.

        C - P + K is the forward at every strike, so each paired near-the-money
        strike is an independent estimate and the median tolerates one stale
        leg. Strikes further than `band` from spot are skipped: out there one
        side trades at the tick floor, where C - P is rounding, not information.

        Returns (None, n) when too few strikes qualify — better to say nothing
        than to publish a forward derived from two crossed quotes.
        """
        if self.spot <= 0:
            return None, 0
        ests = []
        for k in self.strikes:
            if abs(k - self.spot) > band * self.spot:
                continue
            c, p = self.get(k, True), self.get(k, False)
            if not (c and p) or c.mid <= 0.05 or p.mid <= 0.05:
                continue
            ests.append(c.mid - p.mid + k)
        if len(ests) < min_estimates:
            return None, len(ests)
        ests.sort()
        return ests[len(ests) // 2], len(ests)

    def liquid(self, min_oi: int = 100, max_spread_pct: float = 25.0,
               min_price: float = 0.5) -> list[OptionQuote]:
        """Contracts a retail order can actually get filled in."""
        out = []
        for q in self.quotes:
            if q.oi < min_oi or q.mid < min_price:
                continue
            if q.spread_pct and q.spread_pct > max_spread_pct:
                continue
            out.append(q)
        return out


@dataclass
class Leg:
    """One leg of a structure. side=+1 buy, -1 sell. qty in LOTS."""
    strike: float
    is_call: bool
    side: int
    qty: int = 1
    price: float = 0.0              # per-share entry price actually used
    iv: float | None = None

    @property
    def right(self) -> str:
        return "CE" if self.is_call else "PE"

    @property
    def label(self) -> str:
        s = "BUY " if self.side > 0 else "SELL"
        q = f" x{self.qty}" if self.qty != 1 else ""
        return f"{s} {self.strike:g} {self.right}{q} @ {self.price:.2f}"

    def payoff_at(self, spot: float) -> float:
        """Per-share expiry P&L of this leg (premium included)."""
        intrinsic = max(0.0, spot - self.strike) if self.is_call \
            else max(0.0, self.strike - spot)
        return self.side * self.qty * (intrinsic - self.price)


@dataclass
class StrategyResult:
    """A fully priced, fully evaluated structure — the module's output unit."""
    name: str
    legs: list[Leg]
    lot_size: int
    net_premium: float              # per share; <0 = credit received
    max_profit: float               # rupees per lot (inf -> None)
    max_loss: float                 # rupees per lot (positive number)
    breakevens: list[float]
    pop_pct: float                  # fat-tailed, friction-aware
    pop_classic_pct: float          # frictionless lognormal (optimism gap)
    expected_value: float           # rupees per lot under the view
    rr_ratio: float
    friction: float                 # rupees per lot, round trip
    payoff_std: float = 0.0         # std of the per-lot payoff (risk-adjusted rank)
    margin_estimate: float = 0.0
    greeks: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    is_custom: bool = False   # shape not in the textbook catalogue
    source: str = "library"   # library | generated

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_custom": self.is_custom,
            "source": self.source,
            "legs": [{"action": "BUY" if l.side > 0 else "SELL",
                      "strike": l.strike, "right": l.right, "qty": l.qty,
                      "price": round(l.price, 2), "iv": (round(l.iv * 100, 1)
                                                         if l.iv else None)}
                     for l in self.legs],
            "lot_size": self.lot_size,
            "net_premium_per_share": round(self.net_premium, 2),
            "net_premium_per_lot": round(self.net_premium * self.lot_size, 0),
            "credit_or_debit": "CREDIT" if self.net_premium < 0 else "DEBIT",
            "max_profit": (None if self.max_profit == float("inf")
                           else round(self.max_profit, 0)),
            # None = UNBOUNDED. Never emit float('inf'): it is not valid JSON
            # and 500s the API on the first naked short in the menu.
            "max_loss": (None if self.max_loss == float("inf")
                         else round(self.max_loss, 0)),
            "undefined_risk": self.max_loss == float("inf"),
            "breakevens": [round(b, 2) for b in self.breakevens],
            "pop_pct": round(self.pop_pct, 1),
            "pop_classic_pct": round(self.pop_classic_pct, 1),
            "expected_value": round(self.expected_value, 0),
            "payoff_std": round(self.payoff_std, 0),
            # None = UNBOUNDED, exactly as max_profit above. Reporting 0.0 for
            # an unbounded upside reads as "no reward" — the worst possible
            # value on the very metric the docs tell you to fall back to when
            # EV is untrustworthy (a steep smile), which is when unbounded
            # structures dominate.
            "rr_ratio": (None if self.rr_ratio == float("inf")
                         else round(self.rr_ratio, 2)),
            "rr_unbounded": self.rr_ratio == float("inf"),
            "friction": round(self.friction, 0),
            "margin_estimate": round(self.margin_estimate, 0),
            "greeks": {k: round(v, 4) for k, v in self.greeks.items()},
            "tags": self.tags,
            "rationale": self.rationale,
            "score": round(self.score, 4),
        }
