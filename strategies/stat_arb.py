from datetime import date, timedelta

import pandas as pd
from statsmodels.tsa.stattools import coint

from core.data import fetch
from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class StatArbStrategy(BaseStrategy):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.ticker_a = params.get("ticker_a", "RTX")
        self.ticker_b = params.get("ticker_b", "NOC")
        self.lookback = params.get("lookback", 60)
        self.entry_z = params.get("entry_z", 2.0)
        self.exit_z = params.get("exit_z", 0.5)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Rolling OLS: β = Cov(A,B) / Var(B) — uses only past `lookback` observations at each point
        cov = df["Close_A"].rolling(self.lookback).cov(df["Close_B"])
        var_b = df["Close_B"].rolling(self.lookback).var()
        hedge_ratio = cov / var_b

        _, pvalue, _ = coint(df["Close_A"], df["Close_B"])

        df["spread"] = df["Close_A"] - hedge_ratio * df["Close_B"]
        df["spread_mean"] = df["spread"].rolling(self.lookback).mean()
        df["spread_std"] = df["spread"].rolling(self.lookback).std()
        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
        df["coint_pvalue"] = pvalue

        return df

    def generate_signals_series(self, df: pd.DataFrame):
        z = df["z_score"]
        entries_long = (z < -self.entry_z).fillna(False)
        exits_long = (z.abs() < self.exit_z).fillna(False)
        entries_short = (z > self.entry_z).fillna(False)
        exits_short = exits_long.copy()
        return entries_long, exits_long, entries_short, exits_short

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        row = df.iloc[-1]
        timestamp = df.index[-1]

        if pd.isna(row["z_score"]):
            return SignalResult(signal=Signal.HOLD, timestamp=timestamp)

        z = float(row["z_score"])
        confidence = round(min(abs(z) / self.entry_z, 1.0), 4)

        if z < -self.entry_z:
            signal = Signal.BUY
        elif z > self.entry_z:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        return SignalResult(signal=signal, timestamp=timestamp, confidence=confidence)

    def get_latest_signal(self, ticker: str) -> SignalResult:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=200)).isoformat()

        df_a = fetch(self.ticker_a, start, end)[["Close"]].rename(columns={"Close": "Close_A"})
        df_b = fetch(self.ticker_b, start, end)[["Close"]].rename(columns={"Close": "Close_B"})
        df = df_a.join(df_b, how="inner")

        df = self.compute_indicators(df)
        return self.generate_signal(df)
