"""Performance metrics and the go-live gate."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GateCfg
from .exchange import Fill


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(1 - (equity / equity.cummax()).min())


def summarize(equity: pd.Series, fills: list[Fill], periods_per_year: float) -> dict:
    rets = equity.pct_change().dropna()
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9) if len(equity) > 1 else 0.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    std = rets.std()
    downside = rets[rets < 0].std()
    sells = [f for f in fills if f.side == "sell"]
    gains = sorted((f.realized_pnl for f in sells if f.realized_pnl > 0), reverse=True)
    notional = sum(f.notional for f in fills)
    return {
        "days": days,
        "total_return": total,
        "cagr": (1 + total) ** (365 / days) - 1 if days >= 1 and total > -1 else float("nan"),
        "sharpe": float(rets.mean() / std * math.sqrt(periods_per_year)) if std and std > 0 else 0.0,
        "sortino": float(rets.mean() / downside * math.sqrt(periods_per_year)) if downside and downside > 0 else 0.0,
        "max_drawdown": max_drawdown(equity),
        "trades": len(fills),
        "round_trips": len(sells),
        "win_rate": sum(f.realized_pnl > 0 for f in sells) / len(sells) if sells else float("nan"),
        "fees": sum(f.fee for f in fills),
        "avg_slippage_bps": float(np.mean([f.slippage_bps for f in fills])) if fills else 0.0,
        "turnover": notional / float(equity.mean()) if len(equity) else 0.0,
        "top5_profit_share": sum(gains[:5]) / sum(gains) if gains else float("nan"),
    }


def rolling_win_rate(strat: pd.Series, bench: pd.Series, window_days: int) -> float:
    """Fraction of rolling N-day windows (stepped daily) in which the strategy beat the benchmark."""
    s = strat.resample("1D").last().ffill()
    b = bench.reindex(strat.index, method="ffill").resample("1D").last().ffill()
    if len(s) <= window_days:
        return float("nan")
    rs = s.pct_change(window_days).dropna()
    rb = b.pct_change(window_days).reindex(rs.index)
    return float((rs > rb).mean())


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    threshold: str
    fmt: str = "{:.1%}"

    def shown(self) -> str:
        if isinstance(self.value, float) and math.isnan(self.value):
            return "n/a"
        return self.fmt.format(self.value)


def evaluate_gate(g: GateCfg, strat: dict, bench: dict, win_rate: float) -> list[Check]:
    excess = strat["total_return"] - bench["total_return"]
    checks = [
        Check("enough history", strat["days"] >= g.min_days, strat["days"], f">= {g.min_days} days", "{:.1f}d"),
        Check("enough trades", strat["trades"] >= g.min_trades, strat["trades"], f">= {g.min_trades}", "{:.0f}"),
        Check("beats BTC after costs", excess > g.min_excess_return, excess, f"> {g.min_excess_return:+.1%}"),
        Check("max drawdown", strat["max_drawdown"] <= g.max_drawdown, strat["max_drawdown"], f"<= {g.max_drawdown:.0%}"),
        Check(f"beats BTC in rolling {g.rolling_window_days}d windows",
              win_rate >= g.min_rolling_win_rate if not math.isnan(win_rate) else False,
              win_rate, f">= {g.min_rolling_win_rate:.0%}"),
        Check("profit not from a few lucky trades",
              strat["top5_profit_share"] <= g.max_top5_profit_share
              if not math.isnan(strat["top5_profit_share"]) else False,
              strat["top5_profit_share"], f"<= {g.max_top5_profit_share:.0%}"),
    ]
    if g.require_sharpe_above_btc:
        checks.append(Check("Sharpe above BTC", strat["sharpe"] > bench["sharpe"],
                            strat["sharpe"], f"> {bench['sharpe']:.2f}", "{:.2f}"))
    return checks


def format_table(rows: dict[str, dict]) -> str:
    keys = [
        ("total_return", "Total return", "{:+.1%}"), ("cagr", "CAGR", "{:+.1%}"),
        ("sharpe", "Sharpe", "{:.2f}"), ("sortino", "Sortino", "{:.2f}"),
        ("max_drawdown", "Max drawdown", "{:.1%}"), ("trades", "Trades", "{:.0f}"),
        ("win_rate", "Win rate (sells)", "{:.0%}"), ("fees", "Fees paid", "${:,.0f}"),
        ("avg_slippage_bps", "Avg slippage", "{:.1f} bps"), ("turnover", "Turnover", "{:.1f}x"),
        ("top5_profit_share", "Top-5 profit share", "{:.0%}"),
    ]
    names = list(rows)
    width = max(18, *(len(n) for n in names))
    lines = [" " * 20 + "".join(f"{n:>{width + 2}}" for n in names)]
    for k, label, fmt in keys:
        cells = []
        for n in names:
            v = rows[n].get(k, float("nan"))
            cells.append(f"{'—' if isinstance(v, float) and math.isnan(v) else fmt.format(v):>{width + 2}}")
        lines.append(f"{label:<20}" + "".join(cells))
    return "\n".join(lines)


def format_gate(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:<38} {c.shown():>10}  (need {c.threshold})")
    verdict = "PASSED — eligible for a small live run" if all(c.passed for c in checks) else "NOT PASSED — keep paper trading"
    lines.append(f"  => {verdict}")
    return "\n".join(lines)
