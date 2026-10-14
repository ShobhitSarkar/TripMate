"""Exchange implementations sharing one interface, so strategy code never knows which is live.

Interface: `cash`, `holdings`, `avg_cost`, `fills`, `equity(prices)`,
`market_order(symbol, side, qty, ts) -> Fill | None`.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import pandas as pd

from .fills import FillModel

DUST = 1e-12


@dataclass
class Fill:
    ts: pd.Timestamp
    symbol: str
    side: str
    qty: float
    price: float          # average execution price, before fees
    fee: float            # in quote currency
    ref_price: float      # undisturbed price at decision; used to measure slippage
    realized_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def slippage_bps(self) -> float:
        sign = 1 if self.side == "buy" else -1
        return sign * (self.price / self.ref_price - 1) * 1e4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = pd.Timestamp(self.ts).isoformat()
        d["slippage_bps"] = self.slippage_bps
        return d


class MockExchange:
    """Spot-only simulated account. Long only, no leverage, fees charged in quote."""

    def __init__(self, cash: float, fee_rate: float, fill_model: FillModel,
                 min_notional: float = 10.0):
        self.cash = cash
        self.fee_rate = fee_rate
        self.fill_model = fill_model
        self.min_notional = min_notional
        self.holdings: dict[str, float] = {}
        self.avg_cost: dict[str, float] = {}   # per unit, including buy fees
        self.fills: list[Fill] = []

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self.holdings.items() if q > DUST}

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for s, q in self.holdings.items():
            p = prices.get(s)
            if p is not None and math.isfinite(p):
                total += q * p
            else:  # no price (halted/delisted): value at cost rather than silently zero
                total += q * self.avg_cost.get(s, 0.0)
        return total

    def market_order(self, symbol: str, side: str, qty: float, ts) -> Fill | None:
        if side == "sell":
            qty = min(qty, self.holdings.get(symbol, 0.0))
        if qty <= 0:
            return None
        q = self.fill_model.quote(symbol, side, qty)
        if q is None:
            return None
        qty, price = q.qty, q.price
        if side == "buy":
            affordable = self.cash / (price * (1 + self.fee_rate))
            qty = min(qty, affordable)
        if qty * price < self.min_notional:
            return None

        fee = qty * price * self.fee_rate
        fill = Fill(ts=ts, symbol=symbol, side=side, qty=qty, price=price, fee=fee,
                    ref_price=q.ref_price)
        held = self.holdings.get(symbol, 0.0)
        if side == "buy":
            self.cash -= qty * price + fee
            new_held = held + qty
            self.avg_cost[symbol] = (held * self.avg_cost.get(symbol, 0.0) + qty * price + fee) / new_held
            self.holdings[symbol] = new_held
        else:
            self.cash += qty * price - fee
            fill.realized_pnl = qty * price - fee - qty * self.avg_cost.get(symbol, 0.0)
            remaining = held - qty
            if remaining * price < 1e-6:
                remaining = 0.0
                self.avg_cost.pop(symbol, None)
            self.holdings[symbol] = remaining
        if abs(self.cash) < 1e-9:
            self.cash = 0.0
        self.fills.append(fill)
        return fill


class LiveExchange:
    """Real orders through ccxt. Deliberately hard to turn on.

    Requires MEMEBOT_ENABLE_LIVE=yes, API keys in env, and per-order / total caps.
    Positions are only those this bot bought (tracked locally), so it never sells
    coins you hold for other reasons.
    """

    ENABLE_ENV = "MEMEBOT_ENABLE_LIVE"

    def __init__(self, client, fee_rate: float, max_order_usd: float, max_total_usd: float,
                 min_notional: float = 10.0, quote_ccy: str = "USDT"):
        if os.environ.get(self.ENABLE_ENV) != "yes":
            raise RuntimeError(f"live trading disabled: set {self.ENABLE_ENV}=yes to enable")
        self.client = client
        self.fee_rate = fee_rate
        self.max_order_usd = max_order_usd
        self.max_total_usd = max_total_usd
        self.min_notional = min_notional
        self.quote_ccy = quote_ccy
        self.holdings: dict[str, float] = {}
        self.avg_cost: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.cash = self._exchange_cash()

    def _exchange_cash(self) -> float:
        bal = self.client.fetch_balance()
        free = float(bal.get("free", {}).get(self.quote_ccy) or 0.0)
        deployed = sum(q * self.avg_cost.get(s, 0.0) for s, q in self.holdings.items())
        return max(0.0, min(free, self.max_total_usd - deployed))

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self.holdings.items() if q > DUST}

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices.get(s, self.avg_cost.get(s, 0.0))
                               for s, q in self.holdings.items())

    def market_order(self, symbol: str, side: str, qty: float, ts) -> Fill | None:
        ticker = self.client.fetch_ticker(symbol)
        ref = float(ticker["last"])
        if side == "sell":
            qty = min(qty, self.holdings.get(symbol, 0.0))
        else:
            qty = min(qty, self.max_order_usd / ref, self.cash / (ref * (1 + self.fee_rate)))
        qty = float(self.client.amount_to_precision(symbol, qty))
        if qty <= 0 or qty * ref < self.min_notional:
            return None
        order = self.client.create_order(symbol, "market", side, qty)
        if order.get("status") != "closed" or not order.get("filled"):
            order = self.client.fetch_order(order["id"], symbol)
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            return None
        price = float(order.get("average") or order.get("price") or ref)
        fee_info = order.get("fee") or {}
        fee = float(fee_info["cost"]) if fee_info.get("currency") == self.quote_ccy else filled * price * self.fee_rate
        fill = Fill(ts=ts, symbol=symbol, side=side, qty=filled, price=price, fee=fee, ref_price=ref)
        held = self.holdings.get(symbol, 0.0)
        if side == "buy":
            self.avg_cost[symbol] = (held * self.avg_cost.get(symbol, 0.0) + filled * price + fee) / (held + filled)
            self.holdings[symbol] = held + filled
        else:
            fill.realized_pnl = filled * price - fee - filled * self.avg_cost.get(symbol, 0.0)
            self.holdings[symbol] = max(0.0, held - filled)
        self.fills.append(fill)
        self.cash = self._exchange_cash()
        return fill
