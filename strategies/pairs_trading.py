from datetime import date, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from core.data import fetch
from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.ticker_a = params.get("ticker_a", "GLD")
        self.ticker_b = params.get("ticker_b", "GDX")
        self.delta = params.get("delta", 1e-4)
        self.vt = params.get("vt", 1e-3)
        self.entry_z = params.get("entry_z", 2.0)
        self.exit_z = params.get("exit_z", 0.5)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        close_a = df["Close_A"].values
        close_b = df["Close_B"].values
        n = len(df)

        # State: θ = [hedge_ratio, intercept], evolves as random walk
        # Observation: Close_A[t] = Close_B[t]*β + α + noise
        Q = self.delta / (1 - self.delta) * np.eye(2)  # process noise covariance
        R = self.vt                                      # observation noise variance

        theta = np.zeros(2)
        P = np.eye(2)

        hedge_ratios = np.empty(n)
        innovations = np.empty(n)
        innov_stds = np.empty(n)

        for t in range(n):
            H = np.array([close_b[t], 1.0])

            # Predict (using prior state — no look-ahead)
            P_pred = P + Q
            e = close_a[t] - H @ theta
            S = float(H @ P_pred @ H) + R

            # Update
            K = P_pred @ H / S
            theta = theta + K * e
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            hedge_ratios[t] = theta[0]
            innovations[t] = e
            innov_stds[t] = np.sqrt(S)

        df["kf_hedge_ratio"] = hedge_ratios
        df["kf_spread"] = innovations
        df["kf_spread_std"] = innov_stds
        df["z_score"] = innovations / innov_stds

        _, pvalue, _ = coint(close_a, close_b)
        df["coint_pvalue"] = pvalue

        return df

    def generate_signals_series(self, df: pd.DataFrame):
        z = df["z_score"]
        entries_long = z < -self.entry_z
        exits_long = z.abs() < self.exit_z
        entries_short = z > self.entry_z
        exits_short = exits_long.copy()
        return entries_long, exits_long, entries_short, exits_short

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        row = df.iloc[-1]
        timestamp = df.index[-1]

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
