"""OptionSmith test suite — no network, no broker, no pytest required.

    python tests/test_optionsmith.py

Covers the parts where a silent error would produce confident nonsense:
payoff identities, put-call parity, POP sanity, view/vol-edge behaviour,
shape naming, and the advisor end to end.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionsmith import (BY_KEY, Leg, advise, build_named, compute_metrics,  # noqa: E402
                         evaluate, generate, infer_view, synthetic)
from optionsmith.core.mathx import bs_price, implied_vol                      # noqa: E402
from optionsmith.core.payoff import payoff_at                                 # noqa: E402
from optionsmith.strategies.naming import classify                            # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ── 1. option maths ────────────────────────────────────────────────────
def test_math():
    print("\n[1] Black-Scholes core")
    S, K, T, sig = 1000.0, 1000.0, 0.25, 0.30
    c = bs_price(True, S, K, T, sig)
    p = bs_price(False, S, K, T, sig)
    # put-call parity: C - P = S - K e^{-rT}
    lhs, rhs = c - p, S - K * math.exp(-0.07 * T)
    check("put-call parity", approx(lhs, rhs, 1e-9), f"{lhs:.6f} vs {rhs:.6f}")
    check("call price positive", c > 0)
    iv = implied_vol(True, c, S, K, T)
    check("IV inversion round-trips", iv and approx(iv, sig, 1e-4), f"{iv}")
    check("deep OTM call ~ 0", bs_price(True, 100, 1000, 0.02, 0.2) < 1e-6)
    check("IV of nonsense price -> None",
          implied_vol(True, 1e9, S, K, T) is None)


# ── 2. payoff identities ───────────────────────────────────────────────
def test_payoff():
    print("\n[2] payoff engine identities")
    ch = synthetic("T", spot=1000, dte=30, lot_size=100, seed=1)
    atm, step, lot = ch.atm, ch.strike_step, ch.lot_size

    lc = evaluate([Leg(atm, True, 1, 1)], ch, "long call")
    check("long call: unlimited profit", lc.max_profit == float("inf"))
    check("long call: loss = premium",
          approx(lc.max_loss, lc.net_premium * lot, 1e-6),
          f"{lc.max_loss} vs {lc.net_premium*lot}")

    sp = evaluate([Leg(atm, False, -1, 1)], ch, "short put")
    check("short put: downside bounded, not infinite",
          sp.max_loss != float("inf") and sp.max_loss > 0)
    check("short put: max loss ~ (K - premium) * lot",
          approx(sp.max_loss, (atm + sp.net_premium) * lot, 1e-6),
          f"{sp.max_loss}")

    sc = evaluate([Leg(atm, True, -1, 1)], ch, "short call")
    check("short call: unlimited loss", sc.max_loss == float("inf"))

    width = 2 * step
    bcs = evaluate([Leg(atm, True, 1, 1), Leg(atm + width, True, -1, 1)], ch)
    check("vertical: maxP + maxL = width x lot",
          approx(bcs.max_profit + bcs.max_loss, width * lot, 1e-6),
          f"{bcs.max_profit}+{bcs.max_loss} vs {width*lot}")

    ic = evaluate([Leg(atm - 2 * step, False, 1, 1), Leg(atm - step, False, -1, 1),
                   Leg(atm + step, True, -1, 1), Leg(atm + 2 * step, True, 1, 1)], ch)
    check("iron condor: credit + maxL = width x lot",
          approx(ic.max_profit + ic.max_loss, step * lot, 1e-6),
          f"{ic.max_profit}+{ic.max_loss} vs {step*lot}")
    check("iron condor: two breakevens", len(ic.breakevens) == 2, str(ic.breakevens))
    check("iron condor: opens for a credit", ic.net_premium < 0)

    straddle = evaluate([Leg(atm, True, 1, 1), Leg(atm, False, 1, 1)], ch)
    check("straddle: two breakevens", len(straddle.breakevens) == 2)
    check("straddle: loss = total premium",
          approx(straddle.max_loss, straddle.net_premium * lot, 1e-6))

    # payoff at a breakeven must be ~0
    be = bcs.breakevens[0]
    check("breakeven pays ~0", abs(payoff_at(bcs.legs, be)) < 1e-6)


# ── 3. probability of profit ───────────────────────────────────────────
def test_pop():
    print("\n[3] probability of profit")
    # 1 sigma over 30d at ~28% IV is ~8%, so "wide" must mean >= 1 sigma out,
    # not merely a few strikes (the mistake this test originally encoded).
    ch = synthetic("T", spot=1000, dte=30, lot_size=100, n_strikes=41, seed=3)
    atm, step = ch.atm, ch.strike_step

    def condor(n_short: int, n_long: int):
        return [Leg(atm - n_long * step, False, 1, 1),
                Leg(atm - n_short * step, False, -1, 1),
                Leg(atm + n_short * step, True, -1, 1),
                Leg(atm + n_long * step, True, 1, 1)]

    narrow = evaluate(condor(2, 4), ch)
    wide = evaluate(condor(12, 14), ch)          # short strikes ~1.5 sigma out
    check("wider condor has higher POP", wide.pop_pct > narrow.pop_pct,
          f"wide {wide.pop_pct:.1f} vs narrow {narrow.pop_pct:.1f}")
    check("wide condor POP is high", wide.pop_pct > 55, f"{wide.pop_pct:.1f}")
    for label, r in (("wide", wide), ("narrow", narrow)):
        check(f"{label}: realistic POP <= textbook POP",
              r.pop_pct <= r.pop_classic_pct + 1e-9,
              f"{r.pop_pct:.1f} vs {r.pop_classic_pct:.1f}")
    otm = evaluate([Leg(atm + 5 * step, True, 1, 1)], ch)
    check("far OTM long call POP low", otm.pop_pct < 35, f"{otm.pop_pct:.1f}")
    check("POP within [0,100]", all(0 <= r.pop_pct <= 100
                                    for r in (wide, narrow, otm)))


# ── 4. no-arbitrage EV baseline ────────────────────────────────────────
def test_ev_baseline():
    print("\n[4] EV baseline under a neutral view")
    ch = synthetic("T", spot=1200, dte=30, lot_size=100, seed=5)
    atm, step = ch.atm, ch.strike_step
    cases = {
        "long call": [Leg(atm, True, 1, 1)],
        "long put": [Leg(atm, False, 1, 1)],
        "OTM call": [Leg(atm + 3 * step, True, 1, 1)],
        "OTM put": [Leg(atm - 3 * step, False, 1, 1)],
    }
    for n, legs in cases.items():
        r = evaluate(legs, ch, n, tilt=0.0)
        # a fairly priced option must not show a real edge with no view:
        # EV should sit between -(friction + spread) and ~0
        check(f"{n}: EV <= 0 with no view", r.expected_value <= 1.0,
              f"EV {r.expected_value:.0f}")
        check(f"{n}: EV not absurdly negative",
              r.expected_value > -0.25 * max(r.max_loss, 1),
              f"EV {r.expected_value:.0f} vs maxL {r.max_loss:.0f}")


# ── 5. view + volatility edge ──────────────────────────────────────────
def test_view():
    print("\n[5] view and volatility edge")
    ch = synthetic("T", spot=1400, dte=30, lot_size=250, seed=7,
                   wall_call=1470, wall_put=1330)
    m = compute_metrics(ch)
    check("call wall above spot", m.call_wall > ch.spot, str(m.call_wall))
    check("put wall below spot", m.put_wall < ch.spot, str(m.put_wall))
    check("PCR positive", m.pcr_oi > 0)
    check("max pain on the grid", m.max_pain in ch.strikes)
    check("ATM IV sensible", 0.05 < (m.atm_iv or 0) < 2.0, str(m.atm_iv))

    rich = infer_view(m, iv_percentile=90.0)
    cheap = infer_view(m, iv_percentile=10.0)
    check("IV 90th -> iv_rich", rich.volatility == "iv_rich")
    check("IV 10th -> iv_cheap", cheap.volatility == "iv_cheap")
    check("rich shrinks the density", rich.vol_multiplier < 1.0)
    check("cheap widens the density", cheap.vol_multiplier > 1.0)

    bull = advise(ch, user_direction="bullish", user_target_move_pct=4.0, top_n=3)
    bear = advise(ch, user_direction="bearish", user_target_move_pct=-4.0, top_n=3)
    bull_delta = sum(r["greeks"]["delta"] for r in bull.recommendations)
    bear_delta = sum(r["greeks"]["delta"] for r in bear.recommendations)
    check("bullish view -> net long delta", bull_delta > 0, f"{bull_delta:.0f}")
    check("bearish view -> net short delta", bear_delta < 0, f"{bear_delta:.0f}")

    r_rich = advise(ch, user_direction="neutral", iv_percentile=92.0, top_n=3)
    credits = sum(1 for r in r_rich.recommendations
                  if r["credit_or_debit"] == "CREDIT")
    check("rich IV + neutral -> credit structures preferred", credits >= 2,
          f"{credits}/3 credit")
    r_cheap = advise(ch, user_direction="neutral", iv_percentile=8.0, top_n=3)
    debits = sum(1 for r in r_cheap.recommendations
                 if r["credit_or_debit"] == "DEBIT")
    check("cheap IV + neutral -> debit structures preferred", debits >= 2,
          f"{debits}/3 debit")


# ── 6. shape naming ────────────────────────────────────────────────────
def test_naming():
    print("\n[6] structure naming")
    cases = [
        ([Leg(100, True, 1, 1)], "long call"),
        ([Leg(100, False, -1, 1)], "short put"),
        ([Leg(100, True, 1, 1), Leg(110, True, -1, 1)], "bull call spread"),
        ([Leg(90, False, -1, 1), Leg(80, False, 1, 1)], "bull put spread"),
        ([Leg(100, True, 1, 1), Leg(100, False, 1, 1)], "long straddle"),
        ([Leg(110, True, 1, 1), Leg(90, False, 1, 1)], "long strangle"),
        ([Leg(90, False, 1, 1), Leg(95, False, -1, 1),
          Leg(105, True, -1, 1), Leg(110, True, 1, 1)], "iron condor"),
        ([Leg(90, False, 1, 1), Leg(100, False, -1, 1),
          Leg(100, True, -1, 1), Leg(110, True, 1, 1)], "iron butterfly"),
        ([Leg(90, True, 1, 1), Leg(100, True, -1, 2), Leg(110, True, 1, 1)],
         "long call butterfly"),
        ([Leg(90, True, 1, 1), Leg(100, True, -1, 2), Leg(115, True, 1, 1)],
         "broken-wing call butterfly"),
        ([Leg(100, True, 1, 1), Leg(110, True, -1, 2)], "call ratio spread (1x2)"),
        ([Leg(95, False, -1, 1), Leg(105, True, 1, 1)], "risk reversal"),
    ]
    for legs, expect in cases:
        got = classify(legs)
        check(f"naming: {expect}", got == expect, f"got '{got}'")
    weird = classify([Leg(90, True, 1, 1), Leg(97, False, -1, 3),
                      Leg(112, True, -1, 1)])
    check("unrecognised shape -> honest label", weird == "custom structure", weird)


# ── 7. library + generator + advisor ───────────────────────────────────
def test_advisor():
    print("\n[7] library, generator, advisor")
    ch = synthetic("ADV", spot=1400, dte=30, lot_size=250, seed=11,
                   wall_call=1460, wall_put=1340)
    built = failed = 0
    for key in BY_KEY:
        try:
            r = build_named(ch, key)
            built += 1
            assert r.legs and r.lot_size == ch.lot_size
        except Exception:
            failed += 1
    check(f"library builds on a normal chain ({built}/{len(BY_KEY)})",
          built >= len(BY_KEY) - 2, f"{failed} failed")

    v = infer_view(compute_metrics(ch), iv_percentile=80.0)
    gen = generate(ch, v, top_n=6, max_loss_rupees=30_000)
    check("generator returns structures", len(gen) > 0, str(len(gen)))
    check("all generated respect the loss cap",
          all(g.max_loss <= 30_000 for g in gen))
    check("all generated are defined-risk",
          all(g.max_loss != float("inf") for g in gen))
    check("generated are deduped by payoff shape",
          len({tuple(sorted((l.strike, l.is_call, l.side, l.qty)
                            for l in g.legs)) for g in gen}) == len(gen))

    rep = advise(ch, top_n=5, iv_percentile=80.0)
    check("advisor produces recommendations", len(rep.recommendations) > 0)
    check("report carries metrics + view + menu",
          bool(rep.metrics) and bool(rep.view) and len(rep.classic_menu) > 0)
    check("recommendations are sorted by score",
          all(a["score"] >= b["score"] for a, b in
              zip(rep.recommendations, rep.recommendations[1:])))
    check("no undefined-risk in recommendations",
          all(r["max_loss"] is not None for r in rep.recommendations))
    check("each structure reports provenance",
          all(r["source"] in ("library", "generated") for r in rep.recommendations))
    import json
    check("report is JSON-serialisable", bool(json.dumps(rep.to_dict())))


# ── 8. edge cases ──────────────────────────────────────────────────────
def test_forward():
    """The chain's own put-call parity forward.

    The synthetic chain is built with bs_price at RISK_FREE, so its parity
    forward MUST come back as spot*e^(rt) — if the estimator disagrees with the
    model on a chain the model itself generated, it is the estimator that is
    wrong. The second case plants a real basis and checks it is both measured
    and reported, since an unreported reference error shows up later as skew.
    """
    print("\n[9] forward / basis")
    from optionsmith.core.mathx import RISK_FREE

    ch = synthetic("F", spot=1000, dte=30, lot_size=50, seed=5)
    m = compute_metrics(ch)
    expect = 1000 * math.exp(RISK_FREE * ch.t_years)
    check("parity forward recovers the model's own forward",
          m.forward is not None and abs(m.forward - expect) < 1.0,
          f"got {m.forward}, expected {expect:.2f}")
    check("basis matches the carry it was built with",
          m.basis_pct is not None and abs(m.basis_pct - (expect / 1000 - 1) * 100) < 0.12,
          str(m.basis_pct))
    check("implied carry recovers RISK_FREE",
          m.carry_implied is not None and abs(m.carry_implied - RISK_FREE) < 0.02,
          str(m.carry_implied))
    check("no reference-error warning when model and chain agree",
          not any("reference error" in n for n in m.notes), str(m.notes))

    # plant a 1% basis: lift every call, drop every put, as a futures premium does
    ch2 = synthetic("G", spot=1000, dte=30, lot_size=50, seed=5)
    for q in ch2.quotes:
        shift = 5.0 if q.is_call else -5.0
        q.ltp = max(0.05, q.ltp + shift)
        q.bid = max(0.05, q.bid + shift)
        q.ask = max(0.05, q.ask + shift)
    m2 = compute_metrics(ch2)
    check("planted basis is detected",
          m2.forward is not None and m2.forward > m.forward + 8,
          f"{m2.forward} vs {m.forward}")
    check("uncalibrated chain reports the reference error",
          any("reference error" in n for n in m2.notes), str(m2.notes))

    # calibration is what actually FIXES it — the warning above is only for a
    # chain that never went through the enrichment path.
    from optionsmith.chain.loaders import calibrate_carry
    ch3 = synthetic("H", spot=1000, dte=30, lot_size=50, seed=5)
    for q in ch3.quotes:
        shift = 5.0 if q.is_call else -5.0
        q.ltp = max(0.05, q.ltp + shift)
        q.bid = max(0.05, q.bid + shift)
        q.ask = max(0.05, q.ask + shift)
    calibrate_carry(ch3)
    check("calibration picks the carry the chain implies",
          ch3.carry_rate is not None and ch3.carry_rate > RISK_FREE + 0.05,
          str(ch3.carry_rate))
    m3 = compute_metrics(ch3)
    check("calibrated chain no longer carries a reference error",
          not any("reference error" in n for n in m3.notes), str(m3.notes))
    check("calibrated chain still says it is pricing off the forward",
          any("own forward" in n for n in m3.notes), str(m3.notes))

    # and the point of all of it: the phantom skew disappears. Inverting a
    # basis-shifted chain against cash spot lifts call IVs and drops put IVs at
    # the SAME strike, which is arithmetically impossible for a real market.
    def atm_iv_gap(ch):
        from optionsmith.chain.loaders import fill_missing_ivs
        for q in ch.quotes:
            q.iv = None
        fill_missing_ivs(ch)
        c, p = ch.get(ch.atm, True), ch.get(ch.atm, False)
        return abs((c.iv or 0) - (p.iv or 0))

    ch4 = synthetic("I", spot=1000, dte=30, lot_size=50, seed=5)
    for q in ch4.quotes:
        shift = 5.0 if q.is_call else -5.0
        q.ltp = max(0.05, q.ltp + shift)
        q.bid = max(0.05, q.bid + shift)
        q.ask = max(0.05, q.ask + shift)
    ch4.carry_rate = RISK_FREE          # pin the OLD behaviour
    uncal = atm_iv_gap(ch4)
    ch5 = synthetic("J", spot=1000, dte=30, lot_size=50, seed=5)
    for q in ch5.quotes:
        shift = 5.0 if q.is_call else -5.0
        q.ltp = max(0.05, q.ltp + shift)
        q.bid = max(0.05, q.bid + shift)
        q.ask = max(0.05, q.ask + shift)
    cal = atm_iv_gap(ch5)               # carry_rate None -> calibrated
    check("wrong carry manufactures an ATM call/put IV gap", uncal > 0.02,
          f"{uncal:.4f}")
    check("calibration collapses the phantom ATM skew", cal < uncal / 5,
          f"calibrated {cal:.4f} vs uncalibrated {uncal:.4f}")


def test_edges():
    print("\n[8] edge cases")
    thin = synthetic("THIN", spot=500, dte=2, lot_size=100, n_strikes=5, seed=13)
    rep = advise(thin, top_n=3)
    check("thin/near-expiry chain does not crash", isinstance(rep.warnings, list))
    check("expiry-week warning present",
          any("expiry" in w.lower() for w in rep.warnings), str(rep.warnings))
    try:
        build_named(thin, "not_a_strategy")
        check("unknown key raises", False)
    except KeyError:
        check("unknown key raises", True)
    ch = synthetic("Z", spot=1000, dte=30, lot_size=50, seed=17)
    r = evaluate([Leg(ch.atm, True, 1, 1)], ch)
    check("greeks present", set(r.greeks) == {"delta", "gamma", "vega", "theta"})
    check("long call delta in (0,1) x lot",
          0 < r.greeks["delta"] < ch.lot_size)
    check("long option has positive vega", r.greeks["vega"] > 0)
    check("long option bleeds theta", r.greeks["theta"] < 0)


# ── 10. closed-form moments + ranking honesty ──────────────────────────
def test_closed_form():
    """The closed form replaced a 401-point Riemann sum. Two things must hold:
    it agrees with a FINE grid (so it is right, not merely different), and it
    is exact where the grid was not."""
    print("\n[10] closed-form payoff moments")
    from optionsmith.core.payoff import (payoff_moments, payoff_moments_grid,
                                         _sigma_for)
    from optionsmith.strategies.library import BY_KEY

    ch = synthetic("C", spot=1400, dte=30, lot_size=250, n_strikes=17, seed=3)
    from optionsmith.chain.loaders import fill_missing_ivs
    fill_missing_ivs(ch)
    atm_iv = ch.get(ch.atm, True).iv

    def priced(key):
        legs = BY_KEY[key].build(ch)
        if not legs or not all(ch.get(l.strike, l.is_call) for l in legs):
            return None
        out = []
        for l in legs:
            q = ch.get(l.strike, l.is_call)
            out.append(Leg(l.strike, l.is_call, l.side, l.qty,
                           q.exec_price(l.side), q.iv or atm_iv))
        return out

    # STRONGEST check: for a single call the closed form has an independent
    # analytic answer already in the codebase. The density here is exactly the
    # risk-neutral one for spot=center, so E[(S-K)+] = e^(rt) * BS_call, and
    # the mean payoff is lot * (that - premium). This tests the moment code
    # against bs_price, not against another integrator with its own error.
    prem, sig = 40.0, 0.28
    for tilt in (0.0, 0.04, -0.04):
        k = ch.atm
        leg = [Leg(k, True, 1, 1, prem, sig)]
        centre = ch.spot * (1.0 + tilt)
        want = ch.lot_size * (math.exp(ch.r * ch.t_years)
                              * bs_price(True, centre, k, ch.t_years, sig, ch.r)
                              - prem)
        got, _ = payoff_moments(leg, ch.spot, sig, ch.t_years, ch.lot_size,
                                tilt, r=ch.r)
        check(f"long-call EV matches e^rt*BS exactly (tilt {tilt:+.2f})",
              approx(got, want, 1e-9), f"{got:.6f} vs {want:.6f}")

    # put-call parity on the moments: long call - long put at the same strike
    # is a forward, whose expected payoff is E[S] - K - (premiums).
    k = ch.atm
    combo = [Leg(k, True, 1, 1, 40.0, sig), Leg(k, False, -1, 1, 30.0, sig)]
    got, _ = payoff_moments(combo, ch.spot, sig, ch.t_years, ch.lot_size, r=ch.r)
    want = ch.lot_size * (ch.spot * math.exp(ch.r * ch.t_years) - k - 40.0 + 30.0)
    check("synthetic forward EV obeys put-call parity",
          approx(got, want, 1e-9), f"{got:.4f} vs {want:.4f}")

    # agreement with a FINE grid, on the scale that matters. The residual is
    # the GRID's error, not ours: it truncates at +/-5 sigma and never recovers
    # that mass, so it plateaus at n=200k instead of converging (verified).
    # Tolerance is therefore relative to the structure's own payoff std.
    worst = 0.0
    for key in ("iron_condor", "bull_call_spread", "call_butterfly",
                "long_straddle", "call_ratio_spread", "long_call"):
        legs = priced(key)
        if not legs:
            continue
        sg = _sigma_for(legs, atm_iv)
        for tilt in (0.0, 0.03, -0.03):
            c = payoff_moments(legs, ch.spot, sg, ch.t_years, ch.lot_size,
                               tilt, r=ch.r)
            g = payoff_moments_grid(legs, ch.spot, sg, ch.t_years,
                                    ch.lot_size, tilt, n=40001, r=ch.r)
            worst = max(worst, abs(c[0] - g[0]) / max(0.5, 1e-4 * c[1]))
    check("closed form agrees with a 40k-point grid on the payoff's own scale",
          worst <= 1.0, f"worst {worst:.3f} x tolerance")

    # a box spread pays the same at every price: its payoff std is EXACTLY 0.
    # The old grid manufactured ~0.22 of phantom volatility on it.
    legs = priced("box")
    if legs:
        _, sd = payoff_moments(legs, ch.spot, atm_iv, ch.t_years, ch.lot_size,
                               r=ch.r)
        check("riskless box spread has exactly zero payoff std", sd < 1e-6,
              f"std {sd:.6f}")

    # degenerate inputs must not raise
    for bad in ((0.0, 0.2), (1400.0, 0.0)):
        m = payoff_moments([Leg(1400, True, 1, 1, 40.0, 0.28)], bad[0], bad[1],
                           ch.t_years, 250)
        check(f"degenerate input {bad} returns zeros", m == (0.0, 0.0), str(m))


def test_ranking_honesty():
    """rr_ratio, EV damping and delta selection must not lie by omission."""
    print("\n[11] ranking honesty")
    import json
    from optionsmith.strategies.generator import ev_confidence
    from optionsmith.strategies.library import _by_delta, DELTA_TOLERANCE

    ch = synthetic("H", spot=1400, dte=30, lot_size=250, n_strikes=17, seed=3)

    # unbounded upside must not read as "no reward" — 0.0 is the WORST value on
    # the metric the docs tell you to trust when EV is untrustworthy.
    d = build_named(ch, "call_backspread").to_dict()
    check("unbounded profit reports max_profit None", d["max_profit"] is None)
    check("unbounded profit reports rr_ratio None, not 0.0",
          d["rr_ratio"] is None, f"rr {d['rr_ratio']}")
    check("unbounded profit is flagged", d["rr_unbounded"] is True)
    d2 = build_named(ch, "iron_condor").to_dict()
    check("bounded structure still reports a numeric RR",
          isinstance(d2["rr_ratio"], float) and not d2["rr_unbounded"],
          str(d2["rr_ratio"]))
    check("result stays JSON-serialisable", bool(json.dumps(d)))

    # EV confidence: full weight on a flat smile, monotonically less as it steepens
    ws = [ev_confidence(x) for x in (None, 0, 5, 10, 20, 40)]
    check("flat/unknown smile keeps full EV weight",
          ws[0] == 1.0 and ws[1] == 1.0 and ws[2] == 1.0, str(ws))
    check("EV weight falls monotonically as the smile steepens",
          all(a >= b for a, b in zip(ws[2:], ws[3:])) and ws[-1] < ws[2],
          str(ws))
    check("EV weight stays in (0, 1]", all(0 < w <= 1 for w in ws), str(ws))

    # delta selection must FAIL rather than silently mislabel itself
    k = _by_delta(ch, 0.25, True)
    check("25-delta call found on a normal chain", k is not None)
    check("impossible delta target returns None, not the nearest strike",
          _by_delta(ch, 0.999, True) is None)
    narrow = synthetic("N", spot=1400, dte=60, lot_size=250, n_strikes=7, seed=3)
    from optionsmith.chain.loaders import fill_missing_ivs
    fill_missing_ivs(narrow)
    from optionsmith.core.mathx import bs_delta
    got = _by_delta(narrow, 0.20, True)
    if got is not None:
        q = narrow.get(got, True)
        err = abs(bs_delta(True, narrow.spot, got, narrow.t_years, q.iv,
                           narrow.r) - 0.20)
        check("a returned delta strike is inside tolerance",
              err <= DELTA_TOLERANCE, f"missed by {err:.3f}")
    else:
        check("narrow window drops the delta recipe rather than mislabel it",
              True)


if __name__ == "__main__":
    print("OptionSmith test suite")
    for fn in (test_math, test_payoff, test_pop, test_ev_baseline, test_view,
               test_naming, test_advisor, test_forward, test_edges,
               test_closed_form, test_ranking_honesty):
        fn()
    print(f"\n{'='*46}\n  {PASS} passed, {FAIL} failed\n{'='*46}")
    sys.exit(1 if FAIL else 0)
