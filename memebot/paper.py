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
            for ts, msg in risk.events[n_events:]:
                log(f"RISK {msg}")
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
