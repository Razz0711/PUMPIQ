"""
RobinPump On-Chain Data Collector
====================================
Thin adapter over :class:`DexScreenerCollector`.

RobinPump is a Solana-based token launchpad (similar to pump.fun).
Because both collectors use the exact same DexScreener + Solana RPC
data sources and logic, this module simply re-exports the DexScreener
collector and token dataclass under RobinPump-specific names for
backwards-compatible imports.

If RobinPump-specific endpoints or logic are added in the future,
override them in :class:`RobinPumpCollector` below.
"""

from __future__ import annotations

from src.data_collectors.dexscreener_collector import (
    DexPairInfo,               # noqa: F401 – re-export
    DexScreenerCollector,
    DexScreenerToken,
)

# Backwards-compatible alias — functionally identical to DexScreenerToken
RobinPumpToken = DexScreenerToken


class RobinPumpCollector(DexScreenerCollector):
    """
    Collects on-chain data for RobinPump tokens.

    Inherits 100 % of :class:`DexScreenerCollector` behaviour (DexScreener
    API + Solana RPC + optional Apify).  Override methods here if RobinPump
    ever requires divergent logic.

    Usage::

        rp = RobinPumpCollector()
        token = await rp.get_token("CONTRACT_ADDRESS_HERE")
        tokens = await rp.search_tokens("BONK")
    """

    pass

