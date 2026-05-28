from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from core.data import fetch
from strategies import BaseStrategy


@dataclass
class BacktestResult:
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    n_trades: int


def run_backtest(strategy: BaseStrategy, ticker: str, start: str, end: str) -> BacktestResult:
    df = fetch(ticker, start, end)
    df = strategy.compute_indicators(df)
    signals = strategy.generate_signals_series(df)
    close = df["Close"]

    if isinstance(signals, tuple):
        entries_long, exits_long, entries_short, exits_short = signals
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_long,
            exits=exits_long,
            short_entries=entries_short,
            short_exits=exits_short,
            freq="D",
        )
    else:
        prev = signals.shift(1).fillna(False).astype(bool)
        entries = signals & ~prev
        exits = ~signals & prev
        pf = vbt.Portfolio.from_signals(close, entries=entries, exits=exits, freq="D")

    n_trades = int(pf.trades.count())
    try:
        win_rate = float(pf.trades.win_rate())
    except Exception:
        win_rate = 0.0

    return BacktestResult(
        sharpe=float(pf.sharpe_ratio()),
        max_drawdown=float(pf.max_drawdown()),
        win_rate=win_rate,
        total_return=float(pf.total_return()),
        n_trades=n_trades,
    )
