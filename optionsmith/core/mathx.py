"""Black-Scholes pricing, greeks and implied vol — no scipy, no external deps.

Standalone by design: OptionSmith imports nothing from any other project.
All prices are per SHARE in rupees; multiply by lot size for per-lot rupees.
"""
from __future__ import annotations

import math

RISK_FREE = 0.07          # ~RBI repo environment; only mildly affects ranking
SQRT2PI = math.sqrt(2.0 * math.pi)


# ── normal distribution ────────────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def _d1_d2(spot: float, strike: float, t: float, sigma: float,
           r: float) -> tuple[float, float]:
    v = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / v
    return d1, d1 - v


# ── pricing & greeks ───────────────────────────────────────────────────
def bs_price(is_call: bool, spot: float, strike: float, t: float,
             sigma: float, r: float = RISK_FREE) -> float:
    """European option price. Degenerate inputs fall back to intrinsic."""
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    d1, d2 = _d1_d2(spot, strike, t, sigma, r)
    disc = math.exp(-r * t)
    if is_call:
        return spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    return strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_delta(is_call: bool, spot: float, strike: float, t: float,
             sigma: float, r: float = RISK_FREE) -> float:
    if t <= 0 or sigma <= 0:
        itm = (spot > strike) if is_call else (spot < strike)
        return (1.0 if is_call else -1.0) if itm else 0.0
    d1, _ = _d1_d2(spot, strike, t, sigma, r)
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0


def bs_gamma(spot: float, strike: float, t: float, sigma: float,
             r: float = RISK_FREE) -> float:
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, sigma, r)
    return norm_pdf(d1) / (spot * sigma * math.sqrt(t))


def bs_vega(spot: float, strike: float, t: float, sigma: float,
            r: float = RISK_FREE) -> float:
    """Vega per 1.00 (100 vol points) of sigma; /100 for per-vol-point."""
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, sigma, r)
    return spot * norm_pdf(d1) * math.sqrt(t)


def bs_theta(is_call: bool, spot: float, strike: float, t: float,
             sigma: float, r: float = RISK_FREE) -> float:
    """Theta per YEAR (divide by 365 for per-day)."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(spot, strike, t, sigma, r)
    term = -(spot * norm_pdf(d1) * sigma) / (2 * math.sqrt(t))
    disc = math.exp(-r * t)
    if is_call:
        return term - r * strike * disc * norm_cdf(d2)
    return term + r * strike * disc * norm_cdf(-d2)


def implied_vol(is_call: bool, price: float, spot: float, strike: float,
                t: float, r: float = RISK_FREE,
                lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """Bisection IV inversion. None when the price is outside no-arb bounds."""
    if price <= 0 or t <= 0 or spot <= 0 or strike <= 0:
        return None
    intrinsic = max(0.0, (spot - strike * math.exp(-r * t)) if is_call
                    else (strike * math.exp(-r * t) - spot))
    if price < intrinsic - 1e-6:
        return None
    if bs_price(is_call, spot, strike, t, hi, r) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(is_call, spot, strike, t, mid, r) < price:
            lo = mid
        else:
            hi = mid
    iv = 0.5 * (lo + hi)
    return iv if 1e-3 < iv < 4.99 else None


# ── Student-t (fat tails for realistic probability of profit) ──────────
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-9:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    """Student-t CDF (exact via incomplete beta)."""
    if df <= 0:
        return norm_cdf(x)
    p = 0.5 * _betai(0.5 * df, 0.5, df / (df + x * x))
    return 1.0 - p if x > 0 else p
