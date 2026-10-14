"""Risk manager: sits between strategy targets and execution. It can only reduce exposure."""
from __future__ import annotations

import math

import pandas as pd

from .config import RiskCfg


class RiskManager:
    def __init__(self, cfg: RiskCfg, bar: pd.Timedelta):
        self.cfg = cfg
        self.bar = bar
        self.peak: float | None = None
        self.halted = False
        self.day: str | None = None
        self.day_start_equity: float | None = None
        self.cooldown_until: dict[str, pd.Timestamp] = {}
        self.events: list[tuple[pd.Timestamp, str]] = []

    # -- persistence (paper/live restarts) --
    def state(self) -> dict:
        return {
            "peak": self.peak, "halted": self.halted, "day": self.day,
            "day_start_equity": self.day_start_equity,
            "cooldown_until": {s: t.isoformat() for s, t in self.cooldown_until.items()},
        }

    def load_state(self, st: dict) -> None:
        self.peak = st.get("peak")
        self.halted = st.get("halted", False)
        self.day = st.get("day")
        self.day_start_equity = st.get("day_start_equity")
        self.cooldown_until = {s: pd.Timestamp(t) for s, t in st.get("cooldown_until", {}).items()}

    def _event(self, ts, msg: str) -> None:
        self.events.append((ts, msg))

    def apply(self, targets: dict[str, float], ts: pd.Timestamp, equity: float,
              marks: dict[str, float], avg_cost: dict[str, float],
              adv_quote: dict[str, float], held: set[str]) -> dict[str, float]:
        c = self.cfg

        # Portfolio drawdown kill switch: once tripped, stays tripped until a human resets it.
        self.peak = equity if self.peak is None else max(self.peak, equity)
        if c.max_drawdown_halt and not self.halted and equity < self.peak * (1 - c.max_drawdown_halt):
            self.halted = True
            self._event(ts, f"HALT: drawdown {1 - equity / self.peak:.1%} from peak")
        if self.halted:
            return {}

        day = ts.strftime("%Y-%m-%d")
        if day != self.day:
            self.day, self.day_start_equity = day, equity
        if c.daily_loss_limit and equity < self.day_start_equity * (1 - c.daily_loss_limit):
            if not self.events or self.events[-1][1] != f"daily loss limit {day}":
                self._event(ts, f"daily loss limit {day}")
            return {}

        # Per-coin stop loss on average entry, checked at bar close.
        if c.stop_loss:
            for s in held:
                cost, px = avg_cost.get(s), marks.get(s)
                if cost and px and px < cost * (1 - c.stop_loss):
                    self.cooldown_until[s] = ts + self.bar * c.stop_cooldown_bars
                    self._event(ts, f"stop loss {s} at {px:.6g} (entry {cost:.6g})")

        out: dict[str, float] = {}
        for s, w in targets.items():
            if w <= 0:
                continue
            until = self.cooldown_until.get(s)
            if until is not None and ts < until:
                continue
            adv = adv_quote.get(s, 0.0)
            if not (adv and math.isfinite(adv)) or adv < c.min_adv_quote:
                continue
            out[s] = min(w, c.max_weight, c.max_pct_adv * adv / equity if equity > 0 else 0.0)
        gross = sum(out.values())
        if gross > c.max_gross:
            out = {s: w * c.max_gross / gross for s, w in out.items()}
        return out
