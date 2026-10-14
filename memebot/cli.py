"""Command line: python -m memebot <command> [options]"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import bars_per_day, download, load_panel
from .engine import BacktestResult, run_backtest
from .metrics import evaluate_gate, format_gate, format_table, rolling_win_rate, summarize
from .strategies import BuyAndHold, EqualWeightMemes, build_strategy

RESULTS = Path("results")

SWEEP_GRIDS = {
    "momentum": {"lookback": [24, 72, 168], "top_n": [2, 4], "regime_ma": [0, 200]},
    "breakout": {"lookback": [24, 48, 96], "vol_mult": [2.0, 3.0], "exit_ma": [12, 24, 48]},
}


def _ppy(cfg: Config) -> float:
    return 365 * bars_per_day(cfg.exchange.timeframe)


def _benchmarks(panel, cfg, start, end, warmup: int = 1) -> tuple[BacktestResult, BacktestResult]:
    """Benchmarks start on the same bar as the strategy, so both pay the same entry costs."""
    start = max(start or 0, warmup - 1)
    btc = run_backtest(panel, BuyAndHold(), cfg, use_risk=False, start=start, end=end, name="BTC hold")
    memes = run_backtest(panel, EqualWeightMemes(), cfg, use_risk=False, start=start, end=end, name="Meme EW hold")
    return btc, memes


def _align_start(results: list[BacktestResult]) -> list[BacktestResult]:
    """Compare everything over the same window (strategies have different warmups)."""
    t0 = max(r.equity.index[0] for r in results)
    out = []
    for r in results:
        eq = r.equity[r.equity.index >= t0]
        out.append(BacktestResult(r.name, eq, [f for f in r.fills if pd.Timestamp(f.ts) >= t0], r.risk_events))
    return out


def report(results: list[BacktestResult], cfg: Config, label: str) -> dict:
    results = _align_start(results)
    ppy = _ppy(cfg)
    stats = {r.name: summarize(r.equity, r.fills, ppy) for r in results}
    strat, btc = results[0], results[1]
    print(f"\n=== {label}: {strat.equity.index[0]:%Y-%m-%d} -> {strat.equity.index[-1]:%Y-%m-%d} ===")
    print(format_table({("Strategy" if i == 0 else r.name): stats[r.name] for i, r in enumerate(results)}))
    wr = rolling_win_rate(strat.equity, btc.equity, cfg.gate.rolling_window_days)
    print(f"\nGate check on this {label.lower()} (the real gate runs on paper results):")
    print(format_gate(evaluate_gate(cfg.gate, stats[strat.name], stats[btc.name], wr)))
    if strat.risk_events:
        print(f"\nRisk events: {len(strat.risk_events)} (last: {strat.risk_events[-1][1]})")
    return stats


def save(result: BacktestResult, tag: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    result.equity.to_csv(RESULTS / f"{tag}_equity.csv")
    pd.DataFrame([f.to_dict() for f in result.fills]).to_csv(RESULTS / f"{tag}_fills.csv", index=False)


def cmd_fetch(cfg: Config, args) -> None:
    print(f"Downloading {args.days}d of {cfg.exchange.timeframe} candles from {cfg.exchange.name}")
    download(cfg.exchange.name, cfg.exchange.timeframe, cfg.all_symbols, args.days)


def _strategy_from_args(cfg: Config, args):
    name = args.strategy or cfg.strategy.name
    params = dict(cfg.strategy.params) if name == cfg.strategy.name else {}
    if args.params:
        params.update(json.loads(args.params))
    return build_strategy(name, params)


def cmd_backtest(cfg: Config, args) -> None:
    panel = load_panel(cfg.exchange.name, cfg.exchange.timeframe, cfg.all_symbols)
    strat = _strategy_from_args(cfg, args)
    print(f"Backtesting {strat.describe()} on {len(panel.symbols) - 1} coins, {len(panel)} bars")
    split = int(len(panel) * args.split) if args.split else None
    segments = [("Backtest", None, None)] if split is None else [
        ("In-sample", None, split), ("Out-of-sample", split, None)]
    for label, start, end in segments:
        res = run_backtest(panel, strat, cfg, start=start, end=end)
        btc, memes = _benchmarks(panel, cfg, start, end, strat.warmup)
        report([res, btc, memes], cfg, label)
        save(res, f"{strat.name}_{label.lower().replace('-', '_')}")
    print(f"\nEquity curves and fills written to {RESULTS}/")


def cmd_sweep(cfg: Config, args) -> None:
    """Pick parameters on in-sample data only, then show how the pick did out of sample."""
    panel = load_panel(cfg.exchange.name, cfg.exchange.timeframe, cfg.all_symbols)
    name = args.strategy or cfg.strategy.name
    grid = SWEEP_GRIDS[name]
    split = int(len(panel) * args.split)
    ppy = _ppy(cfg)
    rows = []
    bench_cache: dict[int, tuple[BacktestResult, BacktestResult]] = {}
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid, combo))
        strat = build_strategy(name, params)
        if strat.warmup not in bench_cache:
            bench_cache[strat.warmup] = (
                run_backtest(panel, BuyAndHold(), cfg, use_risk=False, start=strat.warmup - 1, end=split),
                run_backtest(panel, BuyAndHold(), cfg, use_risk=False, start=split))
        btc_is, btc_oos = bench_cache[strat.warmup]
        r_is, b_is = _align_start([run_backtest(panel, strat, cfg, end=split), btc_is])
        r_oos, b_oos = _align_start([run_backtest(panel, strat, cfg, start=split), btc_oos])
        s_is, s_oos = summarize(r_is.equity, r_is.fills, ppy), summarize(r_oos.equity, r_oos.fills, ppy)
        rows.append({**params,
                     "IS_sharpe": s_is["sharpe"],
                     "IS_vs_btc": s_is["total_return"] - summarize(b_is.equity, [], ppy)["total_return"],
                     "OOS_sharpe": s_oos["sharpe"],
                     "OOS_vs_btc": s_oos["total_return"] - summarize(b_oos.equity, [], ppy)["total_return"],
                     "OOS_maxdd": s_oos["max_drawdown"], "OOS_trades": s_oos["trades"]})
        print(f"  {params}: IS sharpe {s_is['sharpe']:.2f}  OOS sharpe {s_oos['sharpe']:.2f}", flush=True)
    df = pd.DataFrame(rows).sort_values("IS_sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    best = df.iloc[0]
    print(f"\nBest in-sample: {best[list(grid)].to_dict()}")
    print(f"  out-of-sample vs BTC: {best['OOS_vs_btc']:+.1%}, Sharpe {best['OOS_sharpe']:.2f}")
    print(f"  {int((df['IS_vs_btc'] > 0).sum())}/{len(df)} configs beat BTC in-sample, "
          f"{int((df['OOS_vs_btc'] > 0).sum())}/{len(df)} out-of-sample")
    print("Only the out-of-sample column is evidence. The in-sample winner is usually overfit.")


def cmd_paper(cfg: Config, args) -> None:
    from .paper import run_forward
    run_forward(cfg, args.db, live=False, once=args.once)


def cmd_report(cfg: Config, args) -> None:
    from .paper import Store, gate_report
    s, b, wr, checks = gate_report(cfg, Store(args.db))
    if s is None:
        print(f"No paper history in {args.db} yet.")
        sys.exit(1)
    print(format_table({"Paper strategy": s, "BTC hold": b}))
    print(f"\nRolling {cfg.gate.rolling_window_days}d win rate vs BTC: "
          f"{'n/a' if wr != wr else f'{wr:.0%}'}\n\nGo-live gate:")
    print(format_gate(checks))
    sys.exit(0 if all(c.passed for c in checks) else 2)


def cmd_live(cfg: Config, args) -> None:
    from .paper import Store, gate_report, run_forward
    if not Path(args.paper_db).exists():
        sys.exit(f"Refusing to trade live: no paper record at {args.paper_db}.")
    _, _, _, checks = gate_report(cfg, Store(args.paper_db))
    if not checks or not all(c.passed for c in checks):
        print(format_gate(checks or []))
        sys.exit("Refusing to trade live: the paper record has not passed the gate.")
    run_forward(cfg, args.db, live=True, once=args.once)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="memebot", description=__doc__)
    p.add_argument("--config", default="config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download historical candles")
    f.add_argument("--days", type=int, default=365)

    for name in ("backtest", "sweep"):
        b = sub.add_parser(name, help=f"{name} a strategy on downloaded data")
        b.add_argument("--strategy", help="strategy name (default from config)")
        b.add_argument("--split", type=float, default=0.6 if name == "sweep" else None,
                       help="fraction of data used in-sample; the rest is out-of-sample")
        if name == "backtest":
            b.add_argument("--params", help='JSON overrides, e.g. \'{"top_n": 2}\'')

    pp = sub.add_parser("paper", help="paper trade on live data (runs until stopped)")
    pp.add_argument("--db", default="paper.db")
    pp.add_argument("--once", action="store_true", help="process one bar and exit")

    r = sub.add_parser("report", help="paper-trading performance and go-live gate")
    r.add_argument("--db", default="paper.db")

    lv = sub.add_parser("live", help="REAL MONEY. Requires a passed gate and MEMEBOT_ENABLE_LIVE=yes")
    lv.add_argument("--paper-db", default="paper.db")
    lv.add_argument("--db", default="live.db")
    lv.add_argument("--once", action="store_true")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "sweep": cmd_sweep, "paper": cmd_paper,
     "report": cmd_report, "live": cmd_live}[args.cmd](cfg, args)
