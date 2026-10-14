import math

import numpy as np
import pandas as pd
import pytest

from memebot.config import Config, CostsCfg, RiskCfg, load_config
from memebot.data import Panel
from memebot.engine import rebalance, run_backtest
from memebot.exchange import LiveExchange, MockExchange
from memebot.fills import BarFillModel, Quote, walk_book
from memebot.metrics import evaluate_gate, max_drawdown, rolling_win_rate, summarize
from memebot.paper import Store
from memebot.risk import RiskManager
from memebot.strategies import BuyAndHold, MomentumRotation, Strategy

HOUR = pd.Timedelta(hours=1)


class FixedPrice:
    """Fill model with no costs, for accounting tests."""

    def __init__(self, prices):
        self.prices = prices

    def quote(self, symbol, side, qty):
        p = self.prices[symbol]
        return Quote(price=p, qty=qty, ref_price=p)


def make_panel(closes: dict[str, list[float]], volume: float = 1e9) -> Panel:
    n = len(next(iter(closes.values())))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = pd.DataFrame(closes, index=idx)
    open_ = close.shift(1).fillna(close)
    return Panel(open=open_, high=np.maximum(open_, close), low=np.minimum(open_, close),
                 close=close, volume=pd.DataFrame(volume, index=idx, columns=close.columns))


def test_walk_book_averages_across_levels_and_reports_partial_fill():
    book = {"bids": [[99, 1]], "asks": [[100, 1], [110, 1]]}
    q = walk_book(book, "buy", 1.5)
    assert q.qty == 1.5
    assert q.price == pytest.approx((100 + 0.5 * 110) / 1.5)
    assert q.ref_price == 99.5
    assert walk_book(book, "buy", 5).qty == 2  # book only has 2


def test_bar_fill_model_charges_spread_and_impact_and_caps_size():
    m = BarFillModel(CostsCfg(half_spread_bps=10, impact_coef=1.0, max_participation=0.01), bars_per_day=24)
    m.set_context({"X": 1.0}, {"X": 2_400_000.0}, {"X": 0.1})
    buy = m.quote("X", "buy", 500)
    sell = m.quote("X", "sell", 500)
    assert buy.price > 1.0 > sell.price
    impact = 0.1 * math.sqrt(500 / 2_400_000)
    assert buy.price == pytest.approx(1 + 0.001 + impact)
    # bar volume = 100k; 1% participation = 1k notional max
    assert m.quote("X", "buy", 1_000_000).qty == pytest.approx(1000)
    assert m.quote("Y", "buy", 1) is None


def test_mock_exchange_accounting():
    ex = MockExchange(1000, 0.001, FixedPrice({"X": 10.0}), min_notional=1)
    f = ex.market_order("X", "buy", 50, ts=0)
    assert f.fee == pytest.approx(0.5)
    assert ex.cash == pytest.approx(1000 - 500 - 0.5)
    ex.fill_model.prices["X"] = 12.0
    s = ex.market_order("X", "sell", 1000, ts=1)  # can't sell more than held
    assert s.qty == 50
    assert s.realized_pnl == pytest.approx(600 - 0.6 - 500.5)
    assert ex.cash == pytest.approx(1000 - 500.5 + 600 - 0.6)
    assert ex.positions() == {}


def test_mock_exchange_cannot_overspend():
    ex = MockExchange(100, 0.01, FixedPrice({"X": 1.0}), min_notional=1)
    f = ex.market_order("X", "buy", 1000, ts=0)
    assert f.qty * f.price + f.fee == pytest.approx(100)
    assert ex.cash == pytest.approx(0)


def test_rebalance_exits_fully_and_respects_threshold():
    ex = MockExchange(1000, 0.0, FixedPrice({"A": 1.0, "B": 1.0}), min_notional=1)
    rebalance(ex, {"A": 0.5}, {"A": 1.0, "B": 1.0}, 0, min_trade=1, threshold=0.02)
    assert ex.holdings["A"] == pytest.approx(500)
    fills = rebalance(ex, {"A": 0.51}, {"A": 1.0, "B": 1.0}, 1, min_trade=1, threshold=0.02)
    assert fills == []  # 1% drift is below the 2% threshold
    fills = rebalance(ex, {"B": 0.5}, {"A": 1.0, "B": 1.0}, 2, min_trade=1, threshold=0.02)
    assert [f.side for f in fills] == ["sell", "buy"]
    assert ex.positions() == {"B": pytest.approx(500)}


def test_backtest_fills_at_next_open_not_signal_close():
    class Peek(Strategy):
        warmup = 1

        def target_weights(self, v):
            return {"BTC/USDT": 1.0}

    closes = [100.0 + i for i in range(40)]
    panel = make_panel({"BTC/USDT": closes})
    res = run_backtest(panel, Peek(), Config(), use_risk=False)
    first = res.fills[0]
    i = panel.index.get_loc(first.ts)
    # decided at close of bar i-1 (price closes[i-1]); filled at bar i's open, which is that close
    # in this synthetic panel, but never at a price from bar i's close
    assert first.ref_price == panel.open.iloc[i, 0]
    assert first.ref_price != panel.close.iloc[i, 0]
    assert first.price > first.ref_price  # paid spread + impact


def test_momentum_view_has_no_future_data():
    seen = []

    class Spy(MomentumRotation):
        def target_weights(self, v):
            seen.append(v.close.index[-1] + HOUR == v.ts)
            return {}

    panel = make_panel({"BTC/USDT": list(range(100, 130)), "A/USDT": list(range(1, 31))})
    run_backtest(panel, Spy(lookback=3, regime_ma=0), Config(), use_risk=False)
    assert seen and all(seen)


def test_risk_caps_weights_and_filters_illiquid():
    r = RiskManager(RiskCfg(max_weight=0.25, min_adv_quote=1e6, max_pct_adv=0.001,
                            daily_loss_limit=None, max_drawdown_halt=None), HOUR)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    out = r.apply({"A": 0.9, "B": 0.5, "C": 0.5}, ts, 10_000, {}, {}, {"A": 1e9, "B": 1e5, "C": 1.5e6}, set())
    assert out["A"] == 0.25
    assert "B" not in out                      # below min daily volume
    assert out["C"] == pytest.approx(0.001 * 1.5e6 / 10_000)  # capped by % of volume


def test_risk_drawdown_halt_is_permanent_and_stop_loss_cools_down():
    r = RiskManager(RiskCfg(max_drawdown_halt=0.3, daily_loss_limit=None, min_adv_quote=0,
                            stop_loss=0.1, stop_cooldown_bars=5), HOUR)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    adv = {"A": 1e12}
    out = r.apply({"A": 0.2}, ts, 100, {"A": 0.85}, {"A": 1.0}, adv, {"A"})
    assert out == {}  # stopped out
    assert r.apply({"A": 0.2}, ts + 6 * HOUR, 100, {"A": 0.85}, {}, adv, set()) == {"A": 0.2}
    assert r.apply({"A": 0.2}, ts + 7 * HOUR, 60, {}, {}, adv, set()) == {}
    assert r.halted
    assert r.apply({"A": 0.2}, ts + 8 * HOUR, 200, {}, {}, adv, set()) == {}


def test_metrics_and_gate():
    idx = pd.date_range("2025-01-01", periods=24 * 90, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.005, len(idx))
    strat = pd.Series(100 * np.exp(np.cumsum(noise + 0.0004)), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(noise[::-1])), index=idx)
    assert max_drawdown(pd.Series([100, 50, 75, 120])) == pytest.approx(0.5)
    assert rolling_win_rate(strat, bench, 14) > 0.6

    from memebot.exchange import Fill
    fills = [Fill(idx[i], "A", "sell", 1, 1, 0, 1, realized_pnl=1.0) for i in range(120)]
    cfg = Config()
    s = summarize(strat, fills, 24 * 365)
    b = summarize(bench, [], 24 * 365)
    checks = evaluate_gate(cfg.gate, s, b, rolling_win_rate(strat, bench, 14))
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    checks = evaluate_gate(cfg.gate, b, s, rolling_win_rate(bench, strat, 14))
    assert not all(c.passed for c in checks)


def test_buy_and_hold_benchmark_tracks_price():
    panel = make_panel({"BTC/USDT": [100.0] * 30 + [200.0] * 5})
    res = run_backtest(panel, BuyAndHold(), Config(), use_risk=False)
    assert res.equity.iloc[-1] == pytest.approx(20_000, rel=0.01)


def test_live_exchange_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MEMEBOT_ENABLE_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        LiveExchange(object(), 0.001, 50, 500)


def test_store_roundtrip(tmp_path):
    from memebot.exchange import Fill
    st = Store(tmp_path / "t.db")
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    st.add_fills([Fill(ts, "A", "buy", 1, 2, 0.01, 2)])
    st.add_equity(ts, 100, 50, 30000, {"A": 0.5})
    st.put("k", {"x": 1})
    st.commit()
    assert st.load_fills()[0].price == 2
    assert st.load_equity()["equity"].iloc[0] == 100
    assert st.get("k") == {"x": 1}


def test_repo_config_loads():
    cfg = load_config("config.toml")
    assert cfg.universe.benchmark in cfg.all_symbols
    assert cfg.strategy.name == "momentum"
