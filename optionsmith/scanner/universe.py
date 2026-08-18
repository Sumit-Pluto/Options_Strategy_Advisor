"""The F&O universe to scan.

The list is bundled rather than derived: the Gateway's scripmaster can look a
symbol up but has no "list every F&O underlying" call, and NSE revises the
eligible list only a few times a year. A stale entry costs one failed fetch and
is logged, not crashed on — but the list SHOULD be refreshed from NSE's own
publication each time they revise it.

Override without touching code by writing one symbol per line to the file named
by OPTIONSMITH_UNIVERSE.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import store

# NSE index derivatives — these carry weekly expiries; single stocks do not.
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# NSE stock derivatives. Last reviewed 2026-08-19 — verify against NSE's
# current F&O eligibility list before trusting it wholesale.
STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BANDHANBNK",
    "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL",
    "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT", "CANBK",
    "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE",
    "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELTACORP", "DIVISLAB", "DIXON",
    "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP",
    "GRANULES", "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER",
    "HINDPETRO", "HINDUNILVR", "IBULHSGFIN", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDEA", "IDFC", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL",
    "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC",
    "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL",
    "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT",
    "LTF", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM",
    "MARICO", "MARUTI", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
    "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND",
    "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB",
    "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD",
    "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE",
    "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR",
    "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZYDUSLIFE",
]


def _from_file() -> list[str] | None:
    p = os.environ.get("OPTIONSMITH_UNIVERSE")
    if not p:
        return None
    try:
        lines = Path(p).read_text().splitlines()
    except OSError:
        return None
    out = [ln.strip().upper() for ln in lines
           if ln.strip() and not ln.startswith("#")]
    return out or None


def all_symbols(include_indices: bool = False) -> list[str]:
    """Every candidate, before the blacklist.

    Indices are OFF by default: they carry weekly expiries, so a scan that
    mixes them with monthly stock options compares structures with very
    different times to expiry under one POP filter.
    """
    override = _from_file()
    if override is not None:
        return override
    return (INDICES if include_indices else []) + STOCKS


def scan_list(include_indices: bool = False) -> list[str]:
    """What this scan will actually touch — universe minus the blacklist."""
    black = store.blacklist()
    return [s for s in all_symbols(include_indices) if s not in black]


def with_status(include_indices: bool = True) -> list[dict]:
    """Every symbol plus whether it is switched off — what the UI ticks."""
    black = store.blacklist()
    return [{"symbol": s, "blacklisted": s in black,
             "is_index": s in INDICES}
            for s in all_symbols(include_indices=include_indices)]
