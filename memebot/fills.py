"""Fill models: how much a market order really costs.

This is the part that decides whether paper results mean anything. Filling at the last
price makes almost any meme-coin strategy look profitable; these models charge spread,
market impact and latency, and refuse size the market could not have absorbed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .config import CostsCfg


@dataclass
class Quote:
    price: float      # average execution price before fees
    qty: float        # quantity the market can absorb (may be < requested)
    ref_price: float  # undisturbed reference (bar open / book mid), for slippage stats


class FillModel(Protocol):
    def quote(self, symbol: str, side: str, qty: float) -> Quote | None: ...


class BarFillModel:
    """Backtest fills at the next bar's open, plus spread and square-root market impact.

    impact = impact_coef * daily_vol * sqrt(notional / daily_quote_volume)
    Orders are capped at `max_participation` of an average bar's quote volume.
    Liquidity and volatility inputs are trailing values known at decision time.
    """

    def __init__(self, costs: CostsCfg, bars_per_day: int):
        self.costs = costs
        self.bars_per_day = bars_per_day
        self.ref: dict[str, float] = {}
        self.adv_quote: dict[str, float] = {}
        self.daily_vol: dict[str, float] = {}

    def set_context(self, ref: dict[str, float], adv_quote: dict[str, float],
                    daily_vol: dict[str, float]) -> None:
        self.ref, self.adv_quote, self.daily_vol = ref, adv_quote, daily_vol

    def quote(self, symbol: str, side: str, qty: float) -> Quote | None:
        ref = self.ref.get(symbol)
        adv = self.adv_quote.get(symbol, 0.0)
        if ref is None or not math.isfinite(ref) or ref <= 0 or not adv > 0:
            return None
        max_notional = self.costs.max_participation * adv / self.bars_per_day
        qty = min(qty, max_notional / ref)
        if qty <= 0:
            return None
        sigma = self.daily_vol.get(symbol)
        if sigma is None or not math.isfinite(sigma):
            sigma = 0.10  # unknown volatility: assume meme-coin-typical, not zero
        impact = self.costs.impact_coef * sigma * math.sqrt(qty * ref / adv)
        slip = self.costs.half_spread_bps / 1e4 + impact
        price = ref * (1 + slip) if side == "buy" else ref * (1 - slip)
        return Quote(price=price, qty=qty, ref_price=ref)


class BookFillModel:
    """Paper-trading fills: walk the exchange's live order book, then add latency slippage."""

    def __init__(self, client, costs: CostsCfg, depth: int = 50):
        self.client = client
        self.costs = costs
        self.depth = depth

    def quote(self, symbol: str, side: str, qty: float) -> Quote | None:
        book = self.client.fetch_order_book(symbol, self.depth)
        return walk_book(book, side, qty, self.costs.latency_bps)


def walk_book(book: dict, side: str, qty: float, latency_bps: float = 0.0) -> Quote | None:
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks or qty <= 0:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    levels = asks if side == "buy" else bids
    remaining, cost = qty, 0.0
    for level in levels:
        price, size = level[0], level[1]
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            break
    filled = qty - remaining
    if filled <= 0:
        return None
    avg = cost / filled
    adj = latency_bps / 1e4
    avg = avg * (1 + adj) if side == "buy" else avg * (1 - adj)
    return Quote(price=avg, qty=filled, ref_price=mid)
