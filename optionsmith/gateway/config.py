"""Gateway connection settings, read from the environment or a .env file.

OptionSmith never holds broker credentials. It authenticates to the Gateway as a
*service* (client_id + client_secret issued by the Gateway's
`scripts/register_service.py`) and receives a short-lived scoped JWT; the broker
login, its session and its secrets stay inside the Gateway.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path | None = None) -> dict:
    """Parse a .env into a dict. Values already in the real environment win, so
    a systemd unit's EnvironmentFile or an explicit export always overrides the
    file — the file is the fallback, not the authority."""
    candidates = [Path(path)] if path else []
    candidates += [Path.cwd() / ".env",
                   Path(__file__).resolve().parents[2] / ".env"]
    values: dict[str, str] = {}
    for fp in candidates:
        try:
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except OSError:
            continue
        if values:
            break
    return values


@dataclass
class GatewaySettings:
    base_url: str = "http://127.0.0.1:8000"
    client_id: str = ""
    client_secret: str = ""
    timeout: float = 90.0        # a cold chain build is tens of seconds of broker I/O
    verify_tls: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @classmethod
    def load(cls, env_path: str | Path | None = None) -> "GatewaySettings":
        env = {**_load_dotenv(env_path), **os.environ}
        return cls(
            base_url=env.get("GATEWAY_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            client_id=env.get("GATEWAY_CLIENT_ID", ""),
            client_secret=env.get("GATEWAY_CLIENT_SECRET", ""),
            timeout=float(env.get("GATEWAY_TIMEOUT", "90")),
            verify_tls=env.get("GATEWAY_VERIFY_TLS", "1") not in ("0", "false", "False"),
        )
