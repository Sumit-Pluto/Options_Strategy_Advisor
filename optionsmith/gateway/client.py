"""REST client for the trading Gateway.

Mirrors the pattern the snowball engine uses: exchange client credentials for a
scoped service JWT, attach it to every call, and transparently re-login once on
the 401 that means the token expired. The distinction that matters is between
*our* token being stale (re-login and retry) and the broker session being down
(a human must reconnect — retrying is pointless and hides the real cause).

Read-only by construction. OptionSmith is an advisor: it needs `market` scope
and nothing else, and this client exposes no order-placing call at all.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import GatewaySettings

logger = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """Any failure talking to the Gateway, with the Gateway's own reason kept."""


class BrokerOffline(GatewayError):
    """The Gateway is up but has no live broker session — a human must connect."""


# 401 details that mean OUR service token is bad (worth re-login + retry), as
# opposed to the broker-layer "Not authenticated with ... broker" 401, where a
# fresh service token changes nothing.
_TOKEN_401_HINTS = ("token expired", "invalid token", "authorization header",
                    "no longer exists")


class GatewayClient:
    """Synchronous by design.

    The advisor's core is synchronous, and the dashboard's route handlers are
    plain `def`, which FastAPI already runs in a threadpool. One sync
    implementation therefore serves both without an async duplicate.
    """

    def __init__(self, settings: GatewaySettings | None = None):
        self.settings = settings or GatewaySettings.load()
        if not self.settings.configured:
            raise GatewayError(
                "gateway credentials missing — set GATEWAY_CLIENT_ID and "
                "GATEWAY_CLIENT_SECRET (issue them with the Gateway's "
                "`python -m scripts.register_service --name optionsmith "
                "--scopes market`)")
        try:
            import requests
        except ImportError as e:                          # pragma: no cover
            raise GatewayError("the gateway loader needs `requests` "
                               "(pip install requests)") from e
        self._requests = requests
        self._session = requests.Session()
        self._token: str | None = None

    # ---------------- auth ----------------

    def _login(self) -> None:
        r = self._session.post(
            f"{self.settings.base_url}/api/auth/service-token",
            json={"client_id": self.settings.client_id,
                  "client_secret": self.settings.client_secret},
            timeout=self.settings.timeout, verify=self.settings.verify_tls)
        if r.status_code == 401:
            raise GatewayError("gateway rejected the service credentials "
                               "(check GATEWAY_CLIENT_ID / GATEWAY_CLIENT_SECRET)")
        r.raise_for_status()
        body = r.json()
        self._token = body["token"]
        scopes = body.get("scopes") or []
        if "market" not in scopes:
            logger.warning("service token lacks the 'market' scope (has %s) — "
                           "quote and chain calls will be refused", scopes)

    def _detail(self, r) -> str:
        try:
            return str((r.json() or {}).get("detail", ""))
        except Exception:
            return (r.text or "")[:300]

    def _request(self, method: str, path: str, **kw) -> Any:
        if self._token is None:
            self._login()
        url = f"{self.settings.base_url}{path}"
        kw.setdefault("timeout", self.settings.timeout)
        kw.setdefault("verify", self.settings.verify_tls)

        def send():
            return self._session.request(
                method, url, headers={"Authorization": f"Bearer {self._token}"}, **kw)

        try:
            r = send()
            if r.status_code == 401:
                detail = self._detail(r).lower()
                if any(h in detail for h in _TOKEN_401_HINTS):
                    self._login()
                    r = send()
                else:
                    raise BrokerOffline(
                        f"gateway has no live broker session: {self._detail(r)}")
        except self._requests.RequestException as e:
            raise GatewayError(f"cannot reach gateway at "
                               f"{self.settings.base_url}: {e}") from e

        if r.status_code >= 400:
            raise GatewayError(f"gateway {method} {path} failed "
                               f"[{r.status_code}]: {self._detail(r)}")
        return r.json()

    # ---------------- market data ----------------

    def status(self) -> dict:
        """Broker-session state. `connected` False means every quote below will
        fail until a human reconnects the broker in the Gateway UI."""
        return self._request("GET", "/api/status")

    def search(self, query: str, exchange: str = "") -> list[dict]:
        return self._request("GET", "/api/search",
                             params={"exchange": exchange, "q": query}).get("results", [])

    def quote(self, exchange: str, token: str) -> dict:
        return self._request("GET", "/api/quote",
                             params={"exchange": exchange, "token": token})

    def option_chain(self, symbol: str, exchange: str = "NSE", *,
                     expiry: str = "", atm: float = 0.0, count: int = 15,
                     use_cache: bool = True) -> dict:
        """The full chain payload — see the Gateway's OptionChainResponse.

        `expiry` is the broker's own format ("25-AUG-2026"); empty selects the
        nearest. `atm` may be left at 0, in which case the Gateway resolves the
        centre from the underlying's live quote.
        """
        return self._request("POST", "/api/option-chain", json={
            "symbol": symbol, "exchange": exchange, "expiry": expiry,
            "atm": atm, "count": count, "use_cache": use_cache})

    def expiries(self, symbol: str, exchange: str = "NSE") -> list[dict]:
        """Listed expiries for a symbol, both formats, nearest first.

        Asks for a single strike: the expiry list is built from the scripmaster
        and does not depend on the window, so there is no reason to pay for a
        full chain's quotes just to populate a dropdown.
        """
        d = self.option_chain(symbol, exchange, count=0)
        return [{"expiry": e, "expiry_iso": i}
                for e, i in zip(d.get("expiries", []), d.get("expiries_iso", []))]

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
