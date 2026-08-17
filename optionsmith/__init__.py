"""OptionSmith — standalone option strategy advisor.

Give it an option chain, get back ranked, fully-priced structures: the
classic named strategies AND custom structures generated from the chain's
own pricing, each with exact payoff math, realistic probability of profit,
greeks, margin estimate and its practical caveat.

Quick start (no broker, no network):

    from optionsmith import synthetic, advise
    report = advise(synthetic("RELIANCE", spot=1400, dte=30))
    for r in report.recommendations:
        print(r["name"], r["max_loss"], r["pop_pct"])

Standalone by construction: imports nothing from any other project.
"""
from .advisor import AdviceReport, advise, build_menu, build_named   # noqa: F401
from .analytics.metrics import ChainMetrics, compute as compute_metrics  # noqa: F401
from .analytics.view import MarketView, infer as infer_view           # noqa: F401
from .chain.loaders import (from_csv, from_dhan, from_gateway,       # noqa: F401
                            from_gateway_payload, from_json, synthetic)
from .core.models import Chain, Leg, OptionQuote, StrategyResult      # noqa: F401
from .core.payoff import evaluate                                     # noqa: F401
from .strategies.library import CATALOG, BY_KEY                       # noqa: F401
from .strategies.generator import generate                            # noqa: F401

__version__ = "1.0.0"
__all__ = [
    "advise", "AdviceReport", "build_menu", "build_named",
    "Chain", "Leg", "OptionQuote", "StrategyResult", "evaluate",
    "synthetic", "from_json", "from_csv", "from_dhan",
    "from_gateway", "from_gateway_payload",
    "compute_metrics", "ChainMetrics", "infer_view", "MarketView",
    "generate", "CATALOG", "BY_KEY", "__version__",
]
