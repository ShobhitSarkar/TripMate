"""Exchange client construction, OHLCV download, and loading data into aligned panels."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path("data")
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def make_client(name: str, api_key: str | None = None, secret: str | None = None,
                password: str | None = None) -> ccxt.Exchange:
    opts: dict = {"enableRateLimit": True}
    if api_key:
        opts.update(apiKey=api_key, secret=secret, password=password)
    client = getattr(ccxt, name)(opts)
    # ccxt ignores REQUESTS_CA_BUNDLE by default; honour it so corporate/sandbox proxies work.
    ca = os.environ.get("REQUESTS_CA_BUNDLE")
    if ca:
        client.verify = ca
        client.validateServerSsl = ca
    return client


def timeframe_ms(tf: str) -> int:
    return ccxt.Exchange.parse_timeframe(tf) * 1000


def bars_per_day(tf: str) -> int:
    return max(1, 86_400_000 // timeframe_ms(tf))


def _path(exchange: str, tf: str, symbol: str) -> Path:
    return DATA_DIR / "ohlcv" / exchange / tf / f"{symbol.replace('/', '-')}.csv"


def fetch_ohlcv(client: ccxt.Exchange, symbol: str, tf: str, since_ms: int,
                until_ms: int | None = None, limit: int = 300) -> pd.DataFrame:
    """Page through history. Empty pages (before listing) skip forward."""
    step = timeframe_ms(tf)
    until_ms = until_ms or client.milliseconds()
    rows: list[list] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = client.fetch_ohlcv(symbol, tf, since=cursor, limit=limit)
        if not batch:
            cursor += step * limit
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= cursor:
            break
        cursor = nxt
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("ts").sort_values("ts")
    # Drop the still-forming bar: it would leak a partial candle into backtests.
    df = df[df["ts"] + step <= client.milliseconds()]
    return df.reset_index(drop=True)


def download(exchange: str, tf: str, symbols: list[str], days: int, log=print) -> None:
    client = make_client(exchange)
    client.load_markets()
    since = client.milliseconds() - days * 86_400_000
    for sym in symbols:
        if sym not in client.markets:
            log(f"  {sym}: not listed on {exchange}, skipping")
            continue
        path = _path(exchange, tf, sym)
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        start = since
        if len(existing) and existing["ts"].min() <= since:
            start = int(existing["ts"].max()) + timeframe_ms(tf)
        t0 = time.time()
        new = fetch_ohlcv(client, sym, tf, start)
        df = pd.concat([existing, new]).drop_duplicates("ts").sort_values("ts")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log(f"  {sym}: {len(df)} bars ({len(new)} new, {time.time() - t0:.1f}s)")


@dataclass
class Panel:
    """Aligned OHLCV frames: index = bar open time (UTC), columns = symbols. NaN = not trading."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame  # base-asset volume

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.close)

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "Panel":
        fields_ = {}
        for col in ["open", "high", "low", "close", "volume"]:
            fields_[col] = pd.DataFrame(
                {s: df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))[col]
                 for s, df in frames.items()}
            ).sort_index()
        return cls(**fields_)


def load_panel(exchange: str, tf: str, symbols: list[str]) -> Panel:
    frames = {}
    for sym in symbols:
        path = _path(exchange, tf, sym)
        if path.exists():
            frames[sym] = pd.read_csv(path)
    if not frames:
        raise FileNotFoundError(f"no data under {DATA_DIR}/ohlcv/{exchange}/{tf}; run `fetch` first")
    return Panel.from_frames(frames)
