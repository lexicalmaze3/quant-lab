import pandas as pd
import ta

from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.fast_period = params.get("fast_period", 20)
        self.slow_period = params.get("slow_period", 50)
        self.atr_period = params.get("atr_period", 14)
        self.atr_multiplier = params.get("atr_multiplier", 2.0)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = ta.trend.EMAIndicator(close=df["Close"], window=self.fast_period).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(close=df["Close"], window=self.slow_period).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=self.atr_period
        ).average_true_range()
        return df

    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:
        clean = df.dropna(subset=["ema_fast", "ema_slow", "atr"])

        in_position = False
        entry_price = None
        atr_at_entry = None
        prev_fast = None
        prev_slow = None
        signals = {}

        for i in range(len(clean)):
            row = clean.iloc[i]
            close = float(row["Close"])
            fast = float(row["ema_fast"])
            slow = float(row["ema_slow"])
            atr = float(row["atr"])
            idx = clean.index[i]

            if prev_fast is None:
                prev_fast, prev_slow = fast, slow
                signals[idx] = False
                continue

            if in_position:
                if close < entry_price - self.atr_multiplier * atr_at_entry:
                    in_position = False
                elif fast < slow and prev_fast >= prev_slow:
                    in_position = False
            else:
                if fast > slow and prev_fast <= prev_slow:
                    in_position = True
                    entry_price = close
                    atr_at_entry = atr

            signals[idx] = in_position
            prev_fast, prev_slow = fast, slow

        result = pd.Series(False, index=df.index)
        for idx, val in signals.items():
            result.at[idx] = val
        return result

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        clean = df.dropna(subset=["ema_fast", "ema_slow"])
        timestamp = df.index[-1]

        if len(clean) < 2:
            return SignalResult(signal=Signal.HOLD, timestamp=timestamp)

        prev = clean.iloc[-2]
        curr = clean.iloc[-1]
        fast_prev, slow_prev = float(prev["ema_fast"]), float(prev["ema_slow"])
        fast_curr, slow_curr = float(curr["ema_fast"]), float(curr["ema_slow"])

        confidence = round(min(abs(fast_curr - slow_curr) / slow_curr, 1.0), 4)

        if fast_curr > slow_curr and fast_prev <= slow_prev:
            signal = Signal.BUY
        elif fast_curr < slow_curr and fast_prev >= slow_prev:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        return SignalResult(signal=signal, timestamp=timestamp, confidence=confidence)
