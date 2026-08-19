# OptionSmith — Option Strategy Advisor

A standalone module that turns an option chain into **ranked, fully-priced
option structures**: the classic named strategies *and* custom structures
generated from the chain's own pricing — each with exact payoff maths,
realistic probability of profit, greeks, a margin estimate, and its practical
caveat.

Independent by construction: **imports nothing from any other project**, has
no mandatory third-party dependency for the core (only `fastapi`/`uvicorn` for
the optional dashboard, `requests` for live data), and runs fully offline on a
synthetic chain. Live data arrives over HTTP from the trading Gateway — a
service boundary, not an import — so the broker session and its secrets stay
where they belong.

```
pip install fastapi uvicorn requests   # only for the dashboard and live data
python -m optionsmith demo             # works with nothing installed but Python
```

## 60-second tour

```bash
python -m optionsmith demo                          # synthetic chain, 2 views
python -m optionsmith list                          # the 36-strategy catalogue
python -m optionsmith payoff demo iron_condor       # ASCII payoff diagram
python -m optionsmith advise demo --view bullish --move 3 --iv-pct 85
python -m optionsmith ui                            # dashboard on :8899

python -m optionsmith chain RELIANCE                # live chain from the Gateway
python -m optionsmith advise NIFTY --live --iv-pct 85
```

```python
from optionsmith import synthetic, advise, build_named

chain  = synthetic("RELIANCE", spot=1400, dte=30, lot_size=250)
report = advise(chain, user_direction="neutral", iv_percentile=90)

for r in report.recommendations:
    print(r["name"], r["credit_or_debit"], r["max_loss"], r["pop_pct"])

ic = build_named(chain, "iron_condor")     # price one classic directly
print(ic.breakevens, ic.max_loss, ic.pop_pct)
```

## What it computes, and why you can trust the numbers

| Output | How it is produced |
|---|---|
| max profit / max loss | exact, from the payoff's kink points; unbounded detected analytically. The **downside is never treated as infinite** (price cannot go below zero) — a mistake that silently deletes every put structure. Unbounded profit reports `None` in both `max_profit` and `rr_ratio` (never `0.0`, which reads as *no reward* on the metric you fall back to when EV is untrustworthy) |
| breakevens | exact zero-crossings (the payoff is piecewise-linear, so interpolation between kinks is not an approximation) |
| **POP** | fat-tailed (Student-t, df=5), **friction-aware** (a win is payoff > costs, not > 0), and smile-aware (sigma = mean of the legs' own IVs). The textbook lognormal figure is also reported as `pop_classic_pct` so the optimism gap stays visible. Note it is *hold-to-expiry* and *view-conditional* — it says nothing about the path, and it shifts with your directional tilt |
| expected value | **closed form**, not a grid. The payoff is piecewise-linear, so E[P] and Var[P] are exact sums of truncated lognormal moments — verified against `bs_price` to 1e-9. Deliberately *not* the fat-tailed density: a log-t has no finite mean, so integrating it makes every long call look like a 2× edge |
| ranking | EV per unit of **payoff standard deviation** (a trade "Sharpe") + small POP/shape terms. Ranking by EV per rupee of max loss instead always picks the cheapest lottery ticket — an artefact of the denominator, not an edge |
| prices used | **executable** prices: buy at ask, sell at bid. With a neutral view every structure therefore shows EV ≈ −(friction + spread), which is the correct no-arbitrage answer |
| margin | rough SPAN proxy (defined-risk ≈ max loss; naked shorts ≈ 15% of notional). An estimate — your broker is the source of truth |

## The two engines

**1. Classic library — 36 named structures** (`optionsmith/strategies/library.py`)
singles · verticals · straddle/strangle (delta-selected) · iron condor &
butterfly · call/put flies incl. broken wing · condors · ratio spreads and
backspreads · risk reversal · synthetic long · jade lizard · strap/strip ·
guts · box. Each carries the view it expresses and an honest caveat
("iron condors win 70–80% of the time and lose multiples of the credit when
they fail — check the RR, not the POP").

**Strike placement comes in two flavours.** 30 recipes place strikes at a fixed
offset from ATM (`_k`) or by delta. Six **wall-anchored** variants place them
where the chain itself says the levels are — shorts *at* the OI walls, a
butterfly body *at* max pain. Until these existed the walls and max pain were
computed, displayed and used to pick a direction, then discarded at exactly the
moment they were most useful: an iron condor wrote its short call at ATM+2
whether the wall sat at 1425 or 1500.

Measured on a chain with walls at ±7% (`test_wall_anchored`):

| | offset | @ walls |
|---|---|---|
| iron condor POP | 25.0% | **50.8%** |
| bull put spread POP | 58.6% | **77.1%** |
| bear call spread POP | 54.8% | **73.6%** |
| iron condor credit/lot | ₹4,892 | ₹1,033 |

**This is not free POP.** You are selling further from the money, so the credit
collapses and RR falls with it — the ordinary trade-off, but now struck at the
level the market marks rather than at an arbitrary offset. Both placements are
scored identically and compete in the same ranking, so the report shows which
one actually won. Anchoring does not always win: the jade lizard is worse at
the walls (69.1% → 54.5%), and when the walls sit close to spot there is
nothing to gain. A wall on the wrong side of spot is deep-ITM inventory rather
than a barrier, so those recipes refuse to build instead of writing a short in
the money.

**2. Custom generator** (`optionsmith/strategies/generator.py`)
enumerates 1–4 leg combinations over the chain's **liquid** contracts, keeps
only defined-risk shapes inside your loss budget, dedupes by payoff-shape
fingerprint, then names the survivors — a recognised pattern gets its textbook
name, anything else is honestly labelled **"custom structure"**. Both engines
are scored identically, so the report shows whether the generated shape
actually beat the textbook one.

## How the view drives everything

```
chain ──> metrics ──> view ──┬──> classic library ──┐
   (walls, PCR, max pain,    │                      ├──> unified ranking
    IV, 25Δ skew, GEX,       └──> custom generator ─┘
    OI builds/unwinds)
```

* **Direction** tilts the terminal-price density (`target_move_pct`).
* **Volatility regime** sets `vol_multiplier` — the expected *realised* vol as
  a fraction of *implied*: `iv_rich → 0.85`, `iv_cheap → 1.15`. This is where
  volatility edge actually enters. Legs are priced at market IV but valued
  under the vol you expect to be realised; without it the IV regime would
  change nothing at all.
* **Your view always overrides the inferred one** — the module is a calculator
  for *your* thesis first, and a read of the chain second.

Behaviour this produces (verified in the test suite):

| Your input | What it recommends |
|---|---|
| neutral, IV normal | nothing with a real edge (EV ≈ −costs) — and it says so |
| neutral, IV 90th pct | credit structures (bull put / bear call spreads) |
| neutral, IV 10th pct | long straddle / strangle |
| bullish +4%, IV rich | bull put spread (credit, bullish) |
| bearish −4% | long puts, put backspreads |

## Data sources

```python
from optionsmith import synthetic, from_json, from_csv, from_gateway

synthetic("DEMO", spot=1400, dte=30, lot_size=250)      # offline, realistic
from_json("examples/chain_reliance.json")                # any feed you export
from_csv("chain.csv", "RELIANCE", 1400, "2026-09-24", 250)
from_gateway("RELIANCE")                                 # LIVE, via the Gateway
from_gateway("NIFTY", expiry="25-AUG-2026", count=20)
```

### Live data — the Gateway

Live chains come from the trading Gateway (`gateway_system`), which owns the
broker session. OptionSmith holds **no broker credentials**: it authenticates as
a service, gets a short-lived JWT scoped to `market`, and has no order-placing
call anywhere in it.

```bash
# on the Gateway host — issue credentials, market scope only
cd /opt/gateway_system/Gateway/gateway_backend
./venv/bin/python -m scripts.register_service --name optionsmith --scopes market,status

cp .env.example .env      # then paste GATEWAY_CLIENT_ID / GATEWAY_CLIENT_SECRET
./install.sh
```

```bash
python -m optionsmith chain RELIANCE                 # the chain, as received
python -m optionsmith expiries NIFTY                 # what's listed
python -m optionsmith advise RELIANCE --live --iv-pct 85
python -m optionsmith ui --port 8030                 # dashboard, live toggle
```

What arrives per leg: executable **bid/ask**, LTP, **open interest**,
day-over-day **OI change**, volume, lot size. IV is not sent by the broker and is
inverted on load. The Gateway resolves the underlying's spot itself, so the
window is centred on a real ATM rather than returning every listed strike.

Two things the Gateway had to learn for this to be trustworthy:

* **Quote verification.** Shoonya intermittently answers an option-token request
  with a *different* instrument's quote — usually the underlying's. Measured on
  a 206-leg chain it hit ~17% of legs, and it happens at concurrency 1, so it is
  the broker misrouting. Nothing downstream can spot it: the price is
  well-formed, just for the wrong contract, and a deep-ITM call priced at spot
  makes a spread look free. The adapter now checks the returned token against
  the requested one and retries.
* **OI history.** No broker endpoint returns previous-day OI — the quote has
  only today's, and the daily series has no OI column at all. The Gateway
  snapshots it per contract per day, so `prev_oi` is empty until it has seen two
  sessions. Treat `prev_oi = 0` as *unknown*, never as *unchanged*.

Chain JSON schema:

```json
{"symbol":"RELIANCE","spot":1400.0,"expiry":"2026-09-24","lot_size":250,
 "asof":"2026-08-17",
 "quotes":[{"strike":1400,"right":"CE","ltp":42.5,"bid":42.0,"ask":43.0,
            "oi":18500,"prev_oi":16000,"volume":9200,"iv":28.4}]}
```

`iv` is optional (inverted from price when absent) and accepted as a percent
or a decimal. Missing bid/ask falls back to LTP with a warning in the report.

## Layout

```
optionsmith/
  core/       mathx (BS, greeks, IV, Student-t) · models · payoff engine
  chain/      loaders: json · csv · synthetic · gateway · dhan
              + carry calibration and IV backfill
  gateway/    service-token client for the trading Gateway (read-only)
  analytics/  metrics (walls, PCR, max pain, skew, GEX, OI builds,
              forward/basis, smile spread) · view
  strategies/ library (30 recipes) · generator · naming (shape classifier)
  advisor.py  the orchestrator: advise() / build_menu() / build_named()
  cli/        python -m optionsmith …
  ui/         FastAPI + one self-contained dashboard page
tests/        145 assertions, no pytest needed
examples/     sample chain JSON
deploy/       systemd unit
```

## Tests

```bash
python tests/test_optionsmith.py      # 115 passed — maths, payoff, POP, carry
python tests/test_gateway.py          # 30 passed  — payload mapping, offline
```

`test_gateway.py` replays a captured Gateway payload, so the mapping that would
otherwise mis-price a whole chain from one wrong field name is testable with no
broker session. Its fixture is deliberately awkward — an unquoted leg, a leg
with no book, a leg with no prior OI — because that is the normal state of a
real chain's wings.

The suite deliberately targets the places where a silent error would produce
*confident nonsense*: put-call parity, `maxP + maxL = width × lot` for every
vertical and condor, bounded short-put downside, POP monotonicity in condor
width, the no-arbitrage EV baseline, the vol-edge direction, and shape naming
(including that an unrecognised shape is *not* forced into a familiar label).

## Honest limitations

* **EV is biased by the smile, and on a live chain the bias is large.** Legs are
  priced at their own IVs off the smile, but the payoff is then integrated under
  a *single* sigma (the mean of the legs' IVs). Any structure spanning strikes
  with different IVs books the smile's curvature as edge. Measured on a live
  8-DTE RELIANCE chain (smile spread 15.9 vol points), neutral view, no vol
  thesis — where EV should be ≈ −(friction + spread):

  | | top structure | EV |
  |---|---|---|
  | real smile | call ratio backspread | **+4,169** |
  | flat smile, same chain | long put | −164 ← correct |

  So on a steep chain, **rank by RR and POP, not by EV**. `ChainMetrics` warns
  whenever the liquid smile spans more than 5 vol points, and the ranker now
  *damps* the EV term as the smile steepens (`generator.ev_confidence`: full
  weight to 5 vol points, half at 10, a quarter at 20). That damping is a
  mitigation, not a fix — it is near-uniform within one chain, so it barely
  reorders a single report; its job is to stop the steepest-smile *names*
  dominating when many chains are ranked against each other. Fixing it properly
  means building the terminal density from the smile (Breeden–Litzenberger on
  the call curve) instead of from one sigma — not yet done.
* **Carry is calibrated, not assumed — but only when the chain can say.** Indian
  options are priced off the future, not cash. The chain's own put-call parity
  sets `chain.carry_rate`; without it, IVs invert against the wrong forward and
  manufacture skew (live RELIANCE: a 2.3-point ATM call-over-put IV gap, and a
  −3.3-point "calls bid" skew reading that was pure artifact — both collapse to
  ~0 once calibrated). A chain with too few paired near-the-money strikes falls
  back to a fixed 7% and says so in the warnings.
* **European expiry evaluation.** Fine for comparing structures on monthly
  stock options; early assignment on deep-ITM shorts is not modelled.
* **POP is an estimate for ranking, not a promise.** Fat tails help; they do
  not make the number a probability you should stake a business on.
* **Margin is a proxy**, not a SPAN calculation.
* **Delta-selected recipes can refuse to build.** The 25Δ strangle, 20Δ
  strangle and 25Δ risk reversal now return nothing when the closest listed
  strike misses the target delta by more than 0.05, instead of silently
  relabelling themselves. On a ±5-strike window at 60 DTE the old behaviour
  produced a "20Δ short strangle" whose call was actually 39Δ. Widen the strike
  window (`--strikes`) if a delta recipe disappears from the menu.
* **Single expiry.** Calendars and diagonals need a two-expiry chain — the
  models support the legs, the loaders and generator do not yet.
* **No live order routing** — by design. This is an advisor.
