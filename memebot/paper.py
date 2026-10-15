"""Forward trading loop: paper (mock fills against the live order book) or live.

Each bar close it fetches fresh candles, asks the strategy for targets, runs them through
the risk manager and rebalances, using exactly the code paths the backtester uses.
State lives in SQLite so the process can be restarted without losing the track record.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

from .config import Config
from .data import Panel, bars_per_day, make_client, timeframe_ms
from .engine import held_symbols, liquidity_context, rebalance
from .exchange import Fill, LiveExchange, MockExchange
from .fills import BookFillModel
from .metrics import evaluate_gate, rolling_win_rate, summarize
from .risk import RiskManager
from .strategies import MarketView, build_strategy


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS fills (
                ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, fee REAL,
                ref_price REAL, realized_pnl REAL);
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT PRIMARY KEY, equity REAL, cash REAL, benchmark_price REAL, targets TEXT);
        """)

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))

    def add_fills(self, fills: list[Fill]) -> None:
        self.db.executemany("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?)", [
            (pd.Timestamp(f.ts).isoformat(), f.symbol, f.side, f.qty, f.price, f.fee,
             f.ref_price, f.realized_pnl) for f in fills])

    def add_equity(self, ts, equity: float, cash: float, bench_px: float, targets: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?)",
                        (pd.Timestamp(ts).isoformat(), equity, cash, bench_px, json.dumps(targets)))

    def commit(self) -> None:
        self.db.commit()

    def load_fills(self) -> list[Fill]:
        rows = self.db.execute("SELECT * FROM fills ORDER BY ts").fetchall()
        return [Fill(pd.Timestamp(r[0]), *r[1:]) for r in rows]

    def load_equity(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM equity ORDER BY ts", self.db)
        df.index = pd.to_datetime(df.pop("ts"), utc=True, format="ISO8601")
        return df


def fetch_recent(client, symbols: list[str], tf: str, bars: int) -> Panel:
    step = timeframe_ms(tf)
    now = client.milliseconds()
    frames = {}
    for s in symbols:
        rows = client.fetch_ohlcv(s, tf, limit=bars + 1)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        frames[s] = df[df["ts"] + step <= now]   # closed bars only
    return Panel.from_frames(frames)


def gate_report(cfg: Config, store: Store):
    eq = store.load_equity()
    if len(eq) < 2:
        return None, None, None, None
    fills = store.load_fills()
    bench_eq = cfg.portfolio.initial_cash * (1 - cfg.costs.taker_fee) * eq["benchmark_price"] / eq["benchmark_price"].iloc[0]
    ppy = 365 * bars_per_day(cfg.exchange.timeframe)
    s = summarize(eq["equity"], fills, ppy)
    b = summarize(bench_eq, [], ppy)
    wr = rolling_win_rate(eq["equity"], bench_eq, cfg.gate.rolling_window_days)
    return s, b, wr, evaluate_gate(cfg.gate, s, b, wr)


def run_forward(cfg: Config, db_path: str, live: bool = False, once: bool = False, log=print) -> None:
    tf = cfg.exchange.timeframe
    step = pd.Timedelta(milliseconds=timeframe_ms(tf))
    bpd = bars_per_day(tf)
    store = Store(db_path)
    mode = "live" if live else "paper"
    if store.get("mode", mode) != mode:
        raise RuntimeError(f"{db_path} holds a {store.get('mode')} record; use a different --db")
    store.put("mode", mode)

    if live:
        client = make_client(cfg.exchange.name, os.environ.get("MEMEBOT_API_KEY"),
                             os.environ.get("MEMEBOT_API_SECRET"), os.environ.get("MEMEBOT_API_PASSWORD"))
    else:
        client = make_client(cfg.exchange.name)
    client.load_markets()
    symbols = [s for s in cfg.all_symbols if s in client.markets]
    missing = set(cfg.all_symbols) - set(symbols)
    if missing:
        log(f"not listed on {cfg.exchange.name}, ignored: {sorted(missing)}")

    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    if live:
        ex = LiveExchange(client, cfg.costs.taker_fee, cfg.live.max_order_usd, cfg.live.max_total_usd,
                          cfg.portfolio.min_trade_notional)
    else:
        ex = MockExchange(cfg.portfolio.initial_cash, cfg.costs.taker_fee,
                          BookFillModel(client, cfg.costs), cfg.portfolio.min_trade_notional)
    account = store.get("account")
    if account:
        ex.holdings, ex.avg_cost = account["holdings"], account["avg_cost"]
        if not live:
            ex.cash = account["cash"]
        else:
            ex.cash = ex._exchange_cash()
    risk = RiskManager(cfg.risk, step)
    risk.load_state(store.get("risk", {}))

    bench = cfg.universe.benchmark
    memes = [s for s in symbols if s != bench]
    history = max(strategy.warmup, bpd * 7) + 2
    log(f"{mode} trading {strategy.describe()} on {len(memes)} coins, db={db_path}")

    while True:
        panel = fetch_recent(client, symbols, tf, history)
        t = len(panel) - 1
        decision_ts = panel.index[t] + step
        if store.get("last_bar") != decision_ts.isoformat():
            adv, _ = liquidity_context(panel, bpd)
            marks = {s: v for s, v in panel.close.ffill().iloc[t].items() if pd.notna(v)}
            equity = ex.equity(marks)
            held = held_symbols(ex, marks, cfg.portfolio.min_trade_notional)
            lo = max(0, t + 1 - strategy.warmup)
            view = MarketView(close=panel.close.iloc[lo:], high=panel.high.iloc[lo:],
                              low=panel.low.iloc[lo:], volume=panel.volume.iloc[lo:],
                              ts=decision_ts, held=held, benchmark=bench, symbols=memes)
            targets = strategy.target_weights(view) if len(panel) >= strategy.warmup else {}
            n_events = len(risk.events)
            targets = risk.apply(targets, decision_ts, equity, marks, ex.avg_cost, adv.iloc[t].to_dict(), held)
            new_events = [[ts.isoformat(), msg] for ts, msg in risk.events[n_events:]]
            for ts, msg in new_events:
                log(f"RISK {msg}")
            if new_events:
                store.put("events", (store.get("events", []) + new_events)[-100:])
            fills = rebalance(ex, targets, marks, pd.Timestamp.now(tz="UTC"),
                              cfg.portfolio.min_trade_notional, cfg.portfolio.rebalance_threshold)

            store.add_equity(decision_ts, equity, ex.cash, marks.get(bench, float("nan")), targets)
            store.add_fills(fills)
            store.put("account", {"cash": ex.cash, "holdings": ex.holdings, "avg_cost": ex.avg_cost})
            store.put("risk", risk.state())
            store.put("last_bar", decision_ts.isoformat())
            store.commit()
            log(f"{decision_ts:%Y-%m-%d %H:%M} equity ${equity:,.2f}  targets "
                f"{ {s: round(w, 3) for s, w in targets.items()} }")
            for f in fills:
                log(f"   {f.side:4} {f.symbol:<14} {f.qty:.6g} @ {f.price:.6g}  "
                    f"fee ${f.fee:.2f}  slip {f.slippage_bps:.1f}bps  pnl ${f.realized_pnl:+.2f}")
        if once:
            return
        next_close = decision_ts + step
        wait = (next_close - pd.Timestamp.now(tz="UTC")).total_seconds() + 15
        time.sleep(max(wait, 15))


def export_status(cfg: Config, db_path: str, out: str | Path) -> dict:
    """Write a JSON snapshot of the paper account for the dashboard."""
    store = Store(db_path)
    eq = store.load_equity()
    fills = store.load_fills()
    account = store.get("account") or {"cash": cfg.portfolio.initial_cash, "holdings": {}, "avg_cost": {}}
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    bench = cfg.universe.benchmark

    client = make_client(cfg.exchange.name)
    held = [s for s, q in account["holdings"].items() if q > 0]
    tickers = client.fetch_tickers(sorted(set(held) | {bench})) if held or bench else {}
    prices = {s: float(t["last"]) for s, t in tickers.items() if t.get("last")}

    positions = []
    for s in held:
        q, cost, px = account["holdings"][s], account["avg_cost"].get(s, 0.0), prices.get(s)
        value = q * px if px else q * cost
        positions.append({"symbol": s, "qty": q, "avg_cost": cost, "price": px, "value": value,
                          "pnl": value - q * cost, "pnl_pct": (px / cost - 1) if px and cost else None})
    equity_now = account["cash"] + sum(p["value"] for p in positions)

    regime = None
    regime_ma = getattr(strategy, "regime_ma", 0)
    if regime_ma:
        candles = client.fetch_ohlcv(bench, cfg.exchange.timeframe, limit=regime_ma + 1)[:-1]
        closes = [c[4] for c in candles]
        if len(closes) >= regime_ma:
            ma = sum(closes[-regime_ma:]) / regime_ma
            regime = {"benchmark": bench, "price": closes[-1], "ma": ma, "ma_bars": regime_ma,
                      "risk_on": closes[-1] > ma}

    summary = None
    if len(eq) >= 2:
        s, b, wr, checks = gate_report(cfg, store)
        clean = lambda d: {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}
        summary = {"strategy": clean(s), "benchmark": clean(b),
                   "rolling_win_rate": None if wr != wr else wr,
                   "gate": [{"name": c.name, "passed": bool(c.passed), "value": c.shown(),
                             "threshold": c.threshold} for c in checks],
                   "gate_passed": all(c.passed for c in checks)}

    start_bench = eq["benchmark_price"].iloc[0] if len(eq) else None
    status = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": store.get("mode", "paper"),
        "exchange": cfg.exchange.name,
        "timeframe": cfg.exchange.timeframe,
        "strategy": strategy.describe(),
        "benchmark": bench,
        "universe": cfg.universe.symbols,
        "initial_cash": cfg.portfolio.initial_cash,
        "taker_fee": cfg.costs.taker_fee,
        "last_bar": store.get("last_bar"),
        "started_at": eq.index[0].isoformat() if len(eq) else None,
        "cash": account["cash"],
        "equity": equity_now,
        "benchmark_price": prices.get(bench),
        "benchmark_start_price": start_bench,
        "positions": positions,
        "targets": json.loads(eq["targets"].iloc[-1]) if len(eq) else {},
        "regime": regime,
        "risk": store.get("risk", {}),
        "risk_events": store.get("events", [])[-20:],
        "equity_curve": [{"t": t.isoformat(), "equity": r.equity, "benchmark_price": r.benchmark_price}
                         for t, r in eq.iterrows()],
        "fills": [f.to_dict() for f in fills[-100:]][::-1],
        "summary": summary,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(status, default=str))
    return status
