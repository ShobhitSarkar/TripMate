"""Typed configuration loaded from config.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ExchangeCfg:
    name: str = "okx"
    timeframe: str = "1h"


@dataclass
class UniverseCfg:
    benchmark: str = "BTC/USDT"
    symbols: list[str] = field(default_factory=list)


@dataclass
class CostsCfg:
    taker_fee: float = 0.001
    half_spread_bps: float = 5.0
    impact_coef: float = 1.0
    max_participation: float = 0.01
    latency_bps: float = 5.0


@dataclass
class PortfolioCfg:
    initial_cash: float = 10_000.0
    min_trade_notional: float = 10.0
    rebalance_threshold: float = 0.02


@dataclass
class RiskCfg:
    max_weight: float = 0.25
    max_gross: float = 1.0
    min_adv_quote: float = 1_000_000.0
    max_pct_adv: float = 0.002
    stop_loss: float | None = 0.15
    stop_cooldown_bars: int = 24
    daily_loss_limit: float | None = 0.08
    max_drawdown_halt: float | None = 0.40


@dataclass
class StrategyCfg:
    name: str = "momentum"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateCfg:
    min_days: int = 60
    min_trades: int = 100
    min_excess_return: float = 0.0
    require_sharpe_above_btc: bool = True
    max_drawdown: float = 0.35
    rolling_window_days: int = 14
    min_rolling_win_rate: float = 0.60
    max_top5_profit_share: float = 0.50


@dataclass
class LiveCfg:
    max_order_usd: float = 50.0
    max_total_usd: float = 500.0


@dataclass
class Config:
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    costs: CostsCfg = field(default_factory=CostsCfg)
    portfolio: PortfolioCfg = field(default_factory=PortfolioCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    live: LiveCfg = field(default_factory=LiveCfg)

    @property
    def all_symbols(self) -> list[str]:
        syms = list(self.universe.symbols)
        if self.universe.benchmark not in syms:
            syms.append(self.universe.benchmark)
        return syms


def _build(cls, raw: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path = "config.toml") -> Config:
    raw = tomllib.loads(Path(path).read_text())
    sections = {f.name: f.default_factory for f in fields(Config)}
    kwargs = {}
    for name, value in raw.items():
        if name not in sections:
            raise ValueError(f"unknown config section [{name}]")
        kwargs[name] = _build(type(sections[name]()), value)
    return Config(**kwargs)
