# memebot

A meme-coin trading bot built to answer one question honestly: **does a strategy beat
just holding Bitcoin, after all costs?** It trades against a mock exchange until the
answer is a measured "yes", and refuses to touch real money until then.

```
market data ─► strategy ─► risk manager ─► exchange interface
                (targets)   (caps, stops,     ├── MockExchange  (backtest: next-bar-open fills,
                             kill switches)   │                  paper: live order-book fills)
                                              └── LiveExchange  (ccxt, gated, capped)
                    └──► equity / fills ─► metrics vs BTC ─► go-live gate
```

The strategy and risk code are identical in backtest, paper and live. Only the exchange
object changes.

## Quick start

```bash
pip install -r requirements.txt
python -m memebot fetch --days 365            # hourly candles for the universe in config.toml
python -m memebot backtest                     # strategy vs BTC hold vs equal-weight memes
python -m memebot backtest --strategy breakout --split 0.6   # in-sample / out-of-sample
python -m memebot sweep --strategy momentum    # tune on first 60%, judge on last 40%
python -m memebot paper                        # forward paper trading, runs until stopped
python -m memebot report                       # paper results + go-live gate (exit 0 = passed)
python -m memebot export --out site/status.json   # dashboard data (open dashboard/index.html next to it)
pytest
```

Run `paper` somewhere that stays up (a small VPS, `tmux`, or a systemd service). It
stores everything in `paper.db` and picks up where it left off after a restart.

## Running it 24/7 (free, on GitHub Actions)

`.github/workflows/paper-trade.yml` runs the paper trader every 15 minutes. Each run:

1. restores `paper.db` from the `gh-pages` branch
2. processes any newly closed hourly bar
3. exports `status.json`
4. publishes the dashboard and state back to `gh-pages`

The **dashboard** is at `https://<user>.github.io/<repo>/`. It reloads the bot's state
every minute and streams live OKX prices every 10 seconds, so equity and P&L move in real
time between bot runs. To reset the paper account, delete the `gh-pages` branch.

GitHub's scheduler can run late during busy periods. The bot only acts on closed hourly
candles, so a late run just means a later fill, which the paper fills price honestly.

## Why the paper results should be believable

Most paper-trading bots lie through their fill prices. Here:

| Effect | Backtest | Paper |
|---|---|---|
| Latency | decide at bar close, fill at **next bar's open** | live book fetched after decision, plus `latency_bps` |
| Spread | `half_spread_bps` per order | real bid/ask |
| Market impact | square-root law: `coef × daily_vol × √(order / daily_volume)` | walks real order-book levels |
| Size limits | max `max_participation` of an average bar's volume | only what's in the book |
| Fees | `taker_fee` both sides | same |
| Look-ahead | strategies only receive data up to the decision bar | n/a |

The benchmarks (BTC buy-and-hold, equal-weight meme basket) run through the same engine
and pay the same costs.

## The go-live gate

Defined in `config.toml` under `[gate]`. **Set these before you look at results.**
Defaults: at least 60 days of paper trading, 100 or more trades, higher return than BTC
after costs, Sharpe above BTC's, max drawdown at most 35%, beats BTC in at least 60% of
rolling 14-day windows ("consistently"), and the top 5 trades make up less than 50% of
gross profit (not luck).

`python -m memebot live` refuses to start unless the paper record in `paper.db` passes
the gate **and** `MEMEBOT_ENABLE_LIVE=yes` is set. Live mode also caps each order
(`max_order_usd`) and total capital (`max_total_usd`), and only ever sells coins the bot
itself bought. API keys come from `MEMEBOT_API_KEY`, `MEMEBOT_API_SECRET` and
`MEMEBOT_API_PASSWORD`. Use a trade-only key with **withdrawals disabled**.

## Risk manager

Applied to every strategy, in every mode. It can only reduce exposure:

- per-coin cap (`max_weight`) and a cap by the coin's daily volume (`max_pct_adv`)
- skips coins trading under `min_adv_quote` per day
- stop loss on average entry, then a cooldown before re-entry
- daily loss limit: go flat for the rest of the UTC day
- drawdown kill switch: go flat and **stay** halted until a human intervenes

## Strategies (`memebot/strategies.py`)

- `momentum`: hold the top-N meme coins by trailing return while BTC is above its moving average
- `breakout`: enter on a new N-bar high with a volume spike, exit below a moving average
- `hold_btc`, `hold_memes`: benchmarks

To add a strategy, subclass `Strategy`, implement `target_weights(view) -> {symbol: weight}`,
and register it in `STRATEGIES`.

## Current results (OKX, 1h, Oct 2025 – Oct 2026)

Neither starter strategy works. Over the last year, both lost more than holding BTC:

| | Return | Sharpe | Max DD |
|---|---|---|---|
| momentum (default) | -39.7% | -2.59 | 40.5% (halted) |
| breakout (default) | -37.7% | -1.43 | 40.2% (halted) |
| BTC hold | about -25% | -0.45 | 49.8% |
| meme basket hold | about -47% | -0.62 | 61% |

In the parameter sweeps, 0 of 12 momentum sets and 0 of 18 breakout sets beat BTC, either
in-sample or out-of-sample.
The main costs are turnover (about 100x per year, so roughly $600–800 in fees on $10k plus
slippage) and meme coins' tendency to reverse right after you buy the breakout. These are
starting points to iterate on, not something to deploy.

## Known limitations

- **Survivorship bias:** the universe is coins listed *today*. Coins that died during the
  test window are missing, which flatters any strategy that buys meme coins.
- Stops are checked at bar close, not intrabar, so real losses can gap past the stop.
- The bar fill model's impact coefficient is a guess. Compare paper fills (real books)
  against backtest fills and calibrate `impact_coef`.
- Spot only: no shorting, leverage or funding.
- Exchange availability depends on your country (Binance returns HTTP 451 from US IPs,
  for example). Set `[exchange].name` to any ccxt exchange that lists the pairs.
