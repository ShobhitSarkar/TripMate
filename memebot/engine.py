"""Backtest engine and the rebalancing logic shared with paper/live trading."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import Panel, bars_per_day, timeframe_ms
from .exchange import Fill, MockExchange
from .fills import BarFillModel
from .risk import RiskManager
from .strategies import MarketView, Strategy


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    fills: list[Fill]
    risk_events: list = field(default_factory=list)


def liquidity_context(panel: Panel, bpd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing 24h quote volume and trailing 7d daily volatility, per bar (known at bar close)."""
    quote_vol = panel.volume * panel.close
    adv = quote_vol.rolling(bpd, min_periods=max(1, bpd // 2)).sum()
    ret = panel.close.pct_change(fill_method=None)
    dvol = ret.rolling(bpd * 7, min_periods=bpd).std() * math.sqrt(bpd)
    return adv, dvol


def held_symbols(ex, prices: dict[str, float], min_notional: float) -> set[str]:
    return {s for s, q in ex.positions().items()
            if q * prices.get(s, ex.avg_cost.get(s, 0.0)) >= min_notional}


def rebalance(ex, targets: dict[str, float], prices: dict[str, float], ts,
              min_trade: float, threshold: float) -> list[Fill]:
    """Move the account toward target weights. Sells first to free cash for buys."""
    equity = ex.equity(prices)
    held = ex.positions()
    sells, buys = [], []
    for s in set(held) | set(targets):
        p = prices.get(s)
        if p is None or not math.isfinite(p) or p <= 0:
            continue
        cur = held.get(s, 0.0) * p
        tgt = targets.get(s, 0.0) * equity
        if tgt <= 0 and cur >= min_trade:
            sells.append((s, held[s]))          # full exit, ignore threshold
            continue
        delta = tgt - cur
        if abs(delta) < max(min_trade, threshold * equity):
            continue
        (buys if delta > 0 else sells).append((s, abs(delta) / p))
    fills = []
    for s, q in sells:
        if f := ex.market_order(s, "sell", q, ts):
            fills.append(f)
    for s, q in sorted(buys, key=lambda x: -x[1] * prices[x[0]]):
        if f := ex.market_order(s, "buy", q, ts):
            fills.append(f)
    return fills


def run_backtest(panel: Panel, strategy: Strategy, cfg: Config, *, use_risk: bool = True,
                 start: int | None = None, end: int | None = None,
                 name: str | None = None) -> BacktestResult:
    """Decide at the close of bar t, fill at the open of bar t+1 (never at the signal price)."""
    tf = cfg.exchange.timeframe
    bpd = bars_per_day(tf)
    step = pd.Timedelta(milliseconds=timeframe_ms(tf))
    adv, dvol = liquidity_context(panel, bpd)
    fill_model = BarFillModel(cfg.costs, bpd)
    ex = MockExchange(cfg.portfolio.initial_cash, cfg.costs.taker_fee, fill_model,
                      cfg.portfolio.min_trade_notional)
    risk = RiskManager(cfg.risk, step) if use_risk else None
    bench = cfg.universe.benchmark
    memes = [s for s in panel.symbols if s != bench]

    marks_df = panel.close.ffill()
    window = strategy.warmup
    first = max(start or 0, window - 1)
    last = min(end if end is not None else len(panel) - 1, len(panel) - 1)
    if first >= last:
        raise ValueError(f"not enough bars: need > {window} for warmup, have {last - (start or 0)}")

    idx = panel.index
    opens = panel.open.to_numpy()
    cols = panel.symbols
    times, values = [], []
    for t in range(first, last):
        decision_ts = idx[t] + step
        marks = {s: v for s, v in marks_df.iloc[t].items() if pd.notna(v)}
        equity = ex.equity(marks)
        times.append(decision_ts)
        values.append(equity)

        lo = t + 1 - window
        held = held_symbols(ex, marks, cfg.portfolio.min_trade_notional)
        view = MarketView(
            close=panel.close.iloc[lo:t + 1], high=panel.high.iloc[lo:t + 1],
            low=panel.low.iloc[lo:t + 1], volume=panel.volume.iloc[lo:t + 1],
            ts=decision_ts, held=held, benchmark=bench, symbols=memes,
        )
        targets = strategy.target_weights(view)
        adv_t = adv.iloc[t].to_dict()
        if risk:
            targets = risk.apply(targets, decision_ts, equity, marks, ex.avg_cost, adv_t, held)

        next_open = {s: float(v) for s, v in zip(cols, opens[t + 1]) if np.isfinite(v)}
        fill_model.set_context(next_open, adv_t, dvol.iloc[t].to_dict())
        rebalance(ex, targets, marks, idx[t + 1], cfg.portfolio.min_trade_notional,
                  cfg.portfolio.rebalance_threshold)

    final_marks = {s: v for s, v in marks_df.iloc[last].items() if pd.notna(v)}
    times.append(idx[last] + step)
    values.append(ex.equity(final_marks))
    return BacktestResult(
        name=name or strategy.describe(),
        equity=pd.Series(values, index=pd.DatetimeIndex(times), name="equity"),
        fills=ex.fills,
        risk_events=risk.events if risk else [],
    )
