"""Live data via the trading Gateway.

    from optionsmith import from_gateway, advise
    chain  = from_gateway("RELIANCE")          # live chain, real bid/ask + OI
    report = advise(chain, iv_percentile=85)

Optional: the rest of OptionSmith runs offline on a synthetic or file chain and
never imports this package.
"""
from .client import BrokerOffline, GatewayClient, GatewayError   # noqa: F401
from .config import GatewaySettings                              # noqa: F401

__all__ = ["GatewayClient", "GatewaySettings", "GatewayError", "BrokerOffline"]
