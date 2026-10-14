"""Strategies map a window of past market data to target portfolio weights.

They only ever see data up to and including the decision bar (the engine slices the
window), so look-ahead is impossible by construction. They are stateless: anything
they need about the account is passed in on the view (e.g. `held`).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketView:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    ts: pd.Timestamp            # decision time (close of the last bar)
    held: set[str]
    benchmark: str
    symbols: list[str]          # tradable meme universe (excludes benchmark)


class Strategy:
    name = "base"
    warmup = 1

    def target_weights(self, v: MarketView) -> dict[str, float]:
        raise NotImplementedError

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.name}({params})"


def _live(v: MarketView, symbols: list[str]) -> list[str]:
    last = v.close.iloc[-1]
    return [s for s in symbols if s in last.index and pd.notna(last[s])]


def _risk_on(v: MarketView, regime_ma: int) -> bool:
    """BTC above its moving average. Meme coins are high-beta BTC; don't fight a BTC downtrend."""
    if not regime_ma:
        return True
    btc = v.close[v.benchmark].dropna()
    if len(btc) < regime_ma:
        return False
    return btc.iloc[-1] > btc.iloc[-regime_ma:].mean()


class BuyAndHold(Strategy):
    """Benchmark: hold the given symbols equally weighted (default: the benchmark coin)."""
    name = "hold"

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols

    def target_weights(self, v: MarketView) -> dict[str, float]:
        syms = _live(v, self.symbols or [v.benchmark])
        return {s: 1.0 / len(syms) for s in syms} if syms else {}


class EqualWeightMemes(Strategy):
    """Benchmark: passively hold every listed meme coin in the universe, equal weight."""
    name = "hold_memes"

    def target_weights(self, v: MarketView) -> dict[str, float]:
        syms = _live(v, v.symbols)
        return {s: 1.0 / len(syms) for s in syms} if syms else {}


class MomentumRotation(Strategy):
    """Hold the top-N meme coins by trailing return, only while BTC is in an uptrend.

    Thesis: attention (and so flows) in meme coins is autocorrelated over days;
    winners keep winning for a while, and the whole sector dies when BTC falls.
    """
    name = "momentum"

    def __init__(self, lookback: int = 72, skip: int = 1, top_n: int = 4,
                 regime_ma: int = 200, min_momentum: float = 0.0):
        self.lookback, self.skip, self.top_n = lookback, skip, top_n
        self.regime_ma, self.min_momentum = regime_ma, min_momentum
        self.warmup = max(lookback + skip, regime_ma) + 1

    def target_weights(self, v: MarketView) -> dict[str, float]:
        if not _risk_on(v, self.regime_ma):
            return {}
        c = v.close[_live(v, v.symbols)]
        end = c.iloc[-1 - self.skip]
        start = c.iloc[-1 - self.skip - self.lookback]
        mom = (end / start - 1).dropna()
        winners = mom[mom > self.min_momentum].nlargest(self.top_n)
        return {s: 1.0 / self.top_n for s in winners.index}


class VolumeBreakout(Strategy):
    """Enter on a new N-bar high confirmed by a volume spike; exit below a moving average.

    Thesis: meme pumps start with abnormal volume; a breakout without volume is noise.
    """
    name = "breakout"

    def __init__(self, lookback: int = 48, vol_window: int = 72, vol_mult: float = 3.0,
                 exit_ma: int = 24, weight: float = 0.2, regime_ma: int = 0):
        self.lookback, self.vol_window, self.vol_mult = lookback, vol_window, vol_mult
        self.exit_ma, self.weight, self.regime_ma = exit_ma, weight, regime_ma
        self.warmup = max(lookback, vol_window, exit_ma, regime_ma) + 2

    def target_weights(self, v: MarketView) -> dict[str, float]:
        risk_on = _risk_on(v, self.regime_ma)
        out: dict[str, float] = {}
        for s in _live(v, v.symbols):
            c, h, vol = v.close[s], v.high[s], v.volume[s]
            if c.iloc[-self.warmup + 1:].isna().any():
                continue  # not enough clean history yet (fresh listing)
            price = c.iloc[-1]
            if s in v.held:
                if price > c.iloc[-self.exit_ma:].mean():
                    out[s] = self.weight
            elif risk_on:
                prior_high = h.iloc[-1 - self.lookback:-1].max()
                avg_vol = vol.iloc[-1 - self.vol_window:-1].mean()
                if price > prior_high and vol.iloc[-1] > self.vol_mult * avg_vol:
                    out[s] = self.weight
        gross = sum(out.values())
        return {s: w / gross for s, w in out.items()} if gross > 1 else out


STRATEGIES: dict[str, type[Strategy]] = {
    "momentum": MomentumRotation,
    "breakout": VolumeBreakout,
    "hold_btc": BuyAndHold,
    "hold_memes": EqualWeightMemes,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name](**(params or {}))
