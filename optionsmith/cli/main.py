"""OptionSmith CLI.

    python -m optionsmith demo                          synthetic chain demo
    python -m optionsmith advise chain.json --view bullish --move 3
    python -m optionsmith advise RELIANCE --live --iv-pct 85    live chain
    python -m optionsmith build chain.json iron_condor
    python -m optionsmith list                          the strategy catalogue
    python -m optionsmith payoff chain.json iron_condor  ASCII payoff diagram
    python -m optionsmith chain RELIANCE                 live chain, printed
    python -m optionsmith expiries NIFTY                 listed expiries
    python -m optionsmith ui [--port 8899]              web dashboard

`--live` reads the chain from the trading Gateway (see optionsmith/gateway):
the argument is then an underlying symbol, not a file path.
"""
from __future__ import annotations

import argparse
import json
import sys

from ..advisor import advise, build_named
from ..chain.loaders import from_gateway, from_json, synthetic
from ..strategies.library import CATALOG


def _load(path: str, a=None):
    """Resolve the `chain` argument: a symbol when --live, else a file or demo."""
    if a is not None and getattr(a, "live", False):
        return from_gateway(path, exchange=getattr(a, "exchange", "NSE"),
                            expiry=getattr(a, "expiry", "") or "",
                            count=getattr(a, "strikes", 15))
    return synthetic() if path in ("-", "demo") else from_json(path)


def _live_args(s) -> None:
    """The flags every chain-taking command shares."""
    s.add_argument("--live", action="store_true",
                   help="fetch the chain live from the Gateway; treat the "
                        "chain argument as an underlying symbol")
    s.add_argument("--exchange", default="NSE",
                   help="underlying exchange for --live (NSE, BSE, MCX)")
    s.add_argument("--expiry", default="",
                   help="expiry for --live, broker format e.g. 25-AUG-2026 "
                        "(default: nearest)")
    s.add_argument("--strikes", type=int, default=15, dest="strikes",
                   help="strikes each side of ATM for --live (default 15)")


def _fmt_money(x) -> str:
    if x is None:
        return "unlimited"
    return f"Rs.{x:,.0f}"


def _print_result(r: dict, indent: str = "  ") -> None:
    legs = " | ".join(
        f"{l['action']} {l['strike']:g} {l['right']}"
        + (f" x{l['qty']}" if l["qty"] > 1 else "") + f" @ {l['price']:.2f}"
        for l in r["legs"])
    src = "GENERATED" if r["source"] == "generated" else "classic"
    print(f"{indent}{r['name']}  [{src}{'/custom shape' if r['is_custom'] else ''}]")
    print(f"{indent}  {legs}")
    print(f"{indent}  {r['credit_or_debit']} {_fmt_money(abs(r['net_premium_per_lot']))}"
          f" | max profit {_fmt_money(r['max_profit'])}"
          f" | max loss {_fmt_money(r['max_loss'])}"
          f" | margin ~{_fmt_money(r['margin_estimate'])}")
    print(f"{indent}  breakevens {r['breakevens']} | POP {r['pop_pct']}% "
          f"(textbook {r['pop_classic_pct']}%) | EV {_fmt_money(r['expected_value'])}"
          f" | RR {r['rr_ratio']}")
    g = r["greeks"]
    print(f"{indent}  delta {g.get('delta', 0):+.1f} gamma {g.get('gamma', 0):+.3f} "
          f"vega {g.get('vega', 0):+.1f}/pt theta {g.get('theta', 0):+.1f}/day")
    if r.get("rationale"):
        print(f"{indent}  note: {r['rationale']}")


def cmd_advise(a) -> None:
    ch = _load(a.chain, a)
    rep = advise(ch, user_direction=a.view, user_volatility=a.vol,
                 user_target_move_pct=a.move, iv_percentile=a.iv_pct,
                 top_n=a.top, max_loss_rupees=a.max_loss, max_legs=a.max_legs)
    if a.json:
        print(json.dumps(rep.to_dict(), indent=2))
        return
    m, v = rep.metrics, rep.view
    print(f"\n{'='*74}\n {rep.symbol}  spot {rep.spot:,.2f}  expiry {rep.expiry} "
          f"({rep.dte}d)  lot {rep.lot_size}\n{'='*74}")
    print(f"\nCHAIN READ")
    print(f"  call wall {m['call_wall']}  put wall {m['put_wall']}  "
          f"max pain {m['max_pain']}")
    print(f"  PCR(OI) {m['pcr_oi']}  ATM IV {m['atm_iv_pct']}%  "
          f"25d skew {m['iv_skew_25d_pts']} pts  net GEX {m['net_gex']:,}")
    if m["oi_builds"]:
        top = ", ".join(f"{b['strike']:g}{b['right']} {b['d_oi_pct']:+.0f}%"
                        for b in m["oi_builds"][:4])
        print(f"  fresh OI builds: {top}")
    print(f"\nVIEW: {v['direction'].upper()} / vol {v['volatility']} / "
          f"range {v['range_conviction']}"
          f"{'  (user supplied)' if v['user_supplied'] else '  (inferred)'}")
    for reason in v["reasons"][:5]:
        print(f"  - {reason}")
    print(f"\nTOP {len(rep.recommendations)} STRUCTURES")
    for i, r in enumerate(rep.recommendations, 1):
        print(f"\n{i}.", end=" ")
        _print_result(r, indent="   ")
    if rep.warnings:
        print("\nWARNINGS")
        for w in rep.warnings:
            print(f"  ! {w}")
    print()


def cmd_build(a) -> None:
    ch = _load(a.chain, a)
    res = build_named(ch, a.key)
    print()
    _print_result(res.to_dict())
    print(f"\n  caveat: {res.rationale}\n")


def cmd_list(a) -> None:
    print(f"\n{len(CATALOG)} strategies in the catalogue:\n")
    fam = {}
    for r in CATALOG:
        fam.setdefault(r.family, []).append(r)
    for f in sorted(fam):
        print(f"  [{f}]")
        for r in fam[f]:
            print(f"    {r.key:22s} {r.name:28s} views: {', '.join(r.views)}")
    print()


def cmd_payoff(a) -> None:
    from ..core.payoff import payoff_at
    ch = _load(a.chain, a)
    res = build_named(ch, a.key)
    lo, hi = ch.spot * 0.85, ch.spot * 1.15
    rows, W = 21, 62
    xs = [lo + (hi - lo) * i / (W - 1) for i in range(W)]
    ys = [payoff_at(res.legs, x) * ch.lot_size for x in xs]
    ymin, ymax = min(ys), max(ys)
    span = (ymax - ymin) or 1
    print(f"\n{res.name} — payoff at expiry (Rs. per lot)\n")
    for r in range(rows):
        lvl = ymax - span * r / (rows - 1)
        line = "".join("#" if y >= lvl - span / (2 * rows) else " " for y in ys)
        mark = "0 " if abs(lvl) < span / rows else "  "
        print(f"{lvl:>10,.0f} {mark}|{line}")
    print(" " * 13 + "+" + "-" * W)
    print(f"{'':13s} {lo:,.0f}{' ' * (W - 14)}{hi:,.0f}   (spot {ch.spot:,.0f})")
    print(f"\n  breakevens {[round(b, 1) for b in res.breakevens]} | "
          f"max loss {_fmt_money(res.max_loss)} | POP {res.pop_pct:.1f}%\n")


def cmd_demo(a) -> None:
    ch = synthetic("DEMO", spot=1400, dte=30, lot_size=250,
                   wall_call=1470, wall_put=1330)
    print(f"\nSynthetic chain: {len(ch.quotes)} contracts, spot {ch.spot}, "
          f"{ch.days_to_expiry}d to expiry, step {ch.strike_step:g}")
    for label, kw in (("IV rich (90th pct), neutral",
                       dict(user_direction="neutral", iv_percentile=90.0)),
                      ("Bullish +4%",
                       dict(user_direction="bullish", user_target_move_pct=4.0))):
        rep = advise(ch, top_n=3, **kw)
        print(f"\n--- {label} ---")
        for r in rep.recommendations:
            _print_result(r)
    print()


def cmd_chain(a) -> None:
    """Print the live chain itself — the input the advisor reasons over.

    Worth having separately from `advise`: when a recommendation looks wrong the
    first question is always whether the data was right, and this shows exactly
    what arrived, including which legs quoted no book.
    """
    ch = from_gateway(a.symbol, exchange=a.exchange, expiry=a.expiry or "",
                      count=a.strikes)
    from ..chain.loaders import fill_missing_ivs
    fill_missing_ivs(ch)
    print(f"\n{ch.symbol}  spot {ch.spot:,.2f}  expiry {ch.expiry} "
          f"({ch.days_to_expiry}d)  lot {ch.lot_size}  [{ch.source}]\n")
    print(f"{'':>34s}{'STRIKE':^9s}")
    print(f"{'OI':>10s}{'IV%':>7s}{'BID':>8s}{'ASK':>8s} "
          f"{'':^7s} {'BID':<8s}{'ASK':<8s}{'IV%':<7s}{'OI':<10s}")
    for k in ch.strikes:
        c, p = ch.get(k, True), ch.get(k, False)
        def side(q, right=False):
            if not q:
                return " " * 33
            iv = f"{q.iv*100:.1f}" if q.iv else "—"
            cells = [f"{q.oi:,}" if q.oi else "—", iv,
                     f"{q.bid:.2f}" if q.bid else "—",
                     f"{q.ask:.2f}" if q.ask else "—"]
            if right:
                return f"{cells[2]:<8s}{cells[3]:<8s}{cells[1]:<7s}{cells[0]:<10s}"
            return f"{cells[0]:>10s}{cells[1]:>7s}{cells[2]:>8s}{cells[3]:>8s}"
        mark = "*" if abs(k - ch.atm) < 1e-6 else " "
        print(f"{side(c)} {k:^7g}{mark}{side(p, right=True)}")
    booked = sum(1 for q in ch.quotes if q.bid > 0 and q.ask > 0)
    print(f"\n  {len(ch.quotes)} contracts, {booked} with a two-sided book, "
          f"{len(ch.liquid())} pass the liquidity gate")
    print(f"  * = ATM ({ch.atm:g}), step {ch.strike_step:g}\n")


def cmd_expiries(a) -> None:
    from ..gateway.client import GatewayClient
    with GatewayClient() as c:
        rows = c.expiries(a.symbol, a.exchange)
    if not rows:
        print(f"no listed expiries for {a.symbol} on {a.exchange}")
        return
    print(f"\n{a.symbol} — {len(rows)} listed expiries\n")
    for r in rows:
        print(f"  {r['expiry']:<14s} {r['expiry_iso']}")
    print()


def cmd_ui(a) -> None:
    from ..ui.server import run
    run(port=a.port, host=a.host)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="optionsmith", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("advise", help="full analysis + ranked structures")
    s.add_argument("chain", help="chain JSON path, or 'demo' for synthetic")
    s.add_argument("--view", choices=["bullish", "bearish", "neutral"])
    s.add_argument("--vol", choices=["iv_rich", "iv_cheap", "normal"])
    s.add_argument("--move", type=float, help="target move %% (e.g. 3 or -2.5)")
    s.add_argument("--iv-pct", type=float, dest="iv_pct",
                   help="ATM IV percentile vs its own history (0-100)")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--max-loss", type=float, default=25_000.0, dest="max_loss")
    s.add_argument("--max-legs", type=int, default=4, dest="max_legs")
    s.add_argument("--json", action="store_true")
    _live_args(s)
    s.set_defaults(fn=cmd_advise)

    s = sub.add_parser("build", help="price one named strategy")
    s.add_argument("chain")
    s.add_argument("key", help="catalogue key, e.g. iron_condor")
    _live_args(s)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("payoff", help="ASCII payoff diagram")
    s.add_argument("chain")
    s.add_argument("key")
    _live_args(s)
    s.set_defaults(fn=cmd_payoff)

    s = sub.add_parser("list", help="show the strategy catalogue")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("demo", help="run on a synthetic chain (no data needed)")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("chain", help="print the live option chain from the Gateway")
    s.add_argument("symbol", help="underlying, e.g. RELIANCE or NIFTY")
    s.add_argument("--exchange", default="NSE")
    s.add_argument("--expiry", default="", help="broker format, e.g. 25-AUG-2026")
    s.add_argument("--strikes", type=int, default=15, dest="strikes")
    s.set_defaults(fn=cmd_chain)

    s = sub.add_parser("expiries", help="list an underlying's listed expiries")
    s.add_argument("symbol")
    s.add_argument("--exchange", default="NSE")
    s.set_defaults(fn=cmd_expiries)

    s = sub.add_parser("ui", help="web dashboard")
    s.add_argument("--port", type=int, default=8899)
    s.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 to serve on the network")
    s.set_defaults(fn=cmd_ui)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Gateway failures (unreachable, bad credentials, broker disconnected)
        # are ordinary operating conditions for a live run, not a bug worth a
        # traceback — report the reason and exit non-zero.
        from ..gateway.client import GatewayError
        if isinstance(e, GatewayError):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise
    return 0
