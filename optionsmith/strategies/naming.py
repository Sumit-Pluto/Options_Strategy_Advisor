"""Shape classifier — gives a generated leg-set its textbook name.

Anything the classifier does not recognise is honestly called a
"custom structure" rather than being forced into a familiar label.
"""
from __future__ import annotations

from ..core.models import Leg


def _merged(legs: list[Leg]) -> list[tuple[bool, float, int, int]]:
    """Collapse duplicate (right, strike) legs into net (is_call, k, side, qty)."""
    acc: dict[tuple[bool, float], int] = {}
    for l in legs:
        acc[(l.is_call, l.strike)] = acc.get((l.is_call, l.strike), 0) + \
            l.side * l.qty
    out = []
    for (is_call, k), net in acc.items():
        if net == 0:
            continue
        out.append((is_call, k, 1 if net > 0 else -1, abs(net)))
    return out


def classify(legs: list[Leg]) -> str:
    if not legs:
        return "custom structure"
    m = _merged(legs)
    if not m:
        return "custom structure"
    calls = sorted([x for x in m if x[0]], key=lambda x: x[1])
    puts = sorted([x for x in m if not x[0]], key=lambda x: x[1])
    n = len(m)

    # ── 1 leg ──────────────────────────────────────────────────────────
    if n == 1:
        is_call, _, side, qty = m[0]
        base = f"{'long' if side > 0 else 'short'} {'call' if is_call else 'put'}"
        return base + (f" x{qty}" if qty > 1 else "")

    # ── 2 legs, same right ─────────────────────────────────────────────
    if n == 2 and (len(calls) == 2 or len(puts) == 2):
        rows = calls if len(calls) == 2 else puts
        t = "call" if len(calls) == 2 else "put"
        (_, k1, s1, q1), (_, k2, s2, q2) = rows
        if q1 == q2 == 1:
            if t == "call":
                return "bull call spread" if s1 > 0 else "bear call spread"
            return "bear put spread" if s2 > 0 else "bull put spread"
        if t == "call":
            if s1 > 0 and q1 == 1 and s2 < 0 and q2 == 2:
                return "call ratio spread (1x2)"
            if s1 < 0 and q1 == 1 and s2 > 0 and q2 == 2:
                return "call ratio backspread"
        else:
            if s2 > 0 and q2 == 1 and s1 < 0 and q1 == 2:
                return "put ratio spread (1x2)"
            if s2 < 0 and q2 == 1 and s1 > 0 and q1 == 2:
                return "put ratio backspread"
        return "custom structure"

    # ── 2 legs, one call + one put ─────────────────────────────────────
    if n == 2 and len(calls) == 1 and len(puts) == 1:
        (_, kc, sc, qc) = calls[0]
        (_, kp, sp, qp) = puts[0]
        if qc == qp == 1:
            if sc > 0 and sp > 0:
                if kc == kp:
                    return "long straddle"
                return "long strangle" if kp < kc else "long guts"
            if sc < 0 and sp < 0:
                if kc == kp:
                    return "short straddle"
                return "short strangle" if kp < kc else "short guts"
            if sc > 0 and sp < 0:
                return "synthetic long (combo)" if kc == kp else "risk reversal"
            return "synthetic short (combo)" if kc == kp else "reverse risk reversal"
        if qc == 2 and qp == 1 and sc > 0 and sp > 0 and kc == kp:
            return "strap (2 calls + 1 put)"
        if qp == 2 and qc == 1 and sc > 0 and sp > 0 and kc == kp:
            return "strip (1 call + 2 puts)"
        return "custom structure"

    # ── 3 legs ─────────────────────────────────────────────────────────
    if n == 3 and (len(calls) == 3 or len(puts) == 3):
        rows = calls if len(calls) == 3 else puts
        t = "call" if len(calls) == 3 else "put"
        (_, k1, s1, q1), (_, k2, s2, q2), (_, k3, s3, q3) = rows
        if (q1, q2, q3) == (1, 2, 1):
            even = abs((k2 - k1) - (k3 - k2)) < 1e-6
            if (s1, s2, s3) == (1, -1, 1):
                return f"long {t} butterfly" if even else \
                    f"broken-wing {t} butterfly"
            if (s1, s2, s3) == (-1, 1, -1):
                return f"short {t} butterfly" if even else \
                    f"short broken-wing {t} butterfly"
        return "custom structure"

    if n == 3 and len(puts) == 1 and len(calls) == 2:
        (_, kp, sp, _) = puts[0]
        (_, c1, sc1, _), (_, c2, sc2, _) = calls
        if sp < 0 and sc1 < 0 and sc2 > 0 and c1 < c2 and kp < c1:
            return "jade lizard"
    if n == 3 and len(calls) == 1 and len(puts) == 2:
        (_, kc, sc, _) = calls[0]
        (_, p1, sp1, _), (_, p2, sp2, _) = puts
        if sc < 0 and sp2 < 0 and sp1 > 0 and p1 < p2 and kc > p2:
            return "reverse jade lizard"

    # ── 4 legs ─────────────────────────────────────────────────────────
    if n == 4 and len(calls) == 2 and len(puts) == 2 and all(x[3] == 1 for x in m):
        (_, p1, sp1, _), (_, p2, sp2, _) = puts
        (_, c1, sc1, _), (_, c2, sc2, _) = calls
        if (sp1, sp2, sc1, sc2) == (1, -1, -1, 1) and p2 <= c1:
            return "iron butterfly" if p2 == c1 else "iron condor"
        if (sp1, sp2, sc1, sc2) == (-1, 1, 1, -1) and p2 <= c1:
            return "reverse iron butterfly" if p2 == c1 else "reverse iron condor"
        if (sp1, sp2, sc1, sc2) == (-1, 1, 1, -1) and p1 == c1 and p2 == c2:
            return "box spread"
        if (sp1, sp2, sc1, sc2) == (1, -1, 1, -1) and p1 == c1 and p2 == c2:
            return "box spread"
        return "custom structure"

    if n == 4 and (len(calls) == 4 or len(puts) == 4) and all(x[3] == 1 for x in m):
        rows = calls if len(calls) == 4 else puts
        t = "call" if len(calls) == 4 else "put"
        sides = tuple(x[2] for x in rows)
        k = [x[1] for x in rows]
        if sides == (1, -1, -1, 1):
            even = abs((k[1] - k[0]) - (k[3] - k[2])) < 1e-6
            return f"long {t} condor" if even else f"broken-wing {t} condor"
        if sides == (-1, 1, 1, -1):
            return f"short {t} condor"
        return "custom structure"

    return "custom structure"
