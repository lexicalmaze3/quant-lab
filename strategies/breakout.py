import pandas as pd

from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.channel_period = params.get("channel_period", 20)
        self.volume_period = params.get("volume_period", 20)
        self.volume_multiplier = params.get("volume_multiplier", 1.5)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["donchian_high"] = df["High"].rolling(self.channel_period).max().shift(1)
        df["donchian_low"] = df["Low"].rolling(self.channel_period).min().shift(1)
        df["volume_ma"] = df["Volume"].rolling(self.volume_period).mean()
        return df

    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:
        clean = df.dropna(subset=["donchian_high", "donchian_low", "volume_ma"])

        in_position = False
        signals = {}

        for i in range(len(clean)):
            row = clean.iloc[i]
            close = float(row["Close"])
            donchian_high = float(row["donchian_high"])
            donchian_low = float(row["donchian_low"])
            volume = float(row["Volume"])
            volume_ma = float(row["volume_ma"])
            idx = clean.index[i]

            if not in_position:
                if close > donchian_high and volume > volume_ma * self.volume_multiplier:
                    in_position = True
            else:
                if close < donchian_low:
                    in_position = False

            signals[idx] = in_position

        result = pd.Series(False, index=df.index)
        for idx, val in signals.items():
            result.at[idx] = val
        return result

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        row = df.iloc[-1]
        timestamp = df.index[-1]

        donchian_high = row["donchian_high"]
        donchian_low = row["donchian_low"]
        volume_ma = row["volume_ma"]

        if pd.isna(donchian_high) or pd.isna(donchian_low) or pd.isna(volume_ma):
            return SignalResult(signal=Signal.HOLD, timestamp=timestamp)

        close = float(row["Close"])
        volume = float(row["Volume"])
        donchian_high = float(donchian_high)
        donchian_low = float(donchian_low)
        volume_ma = float(volume_ma)

        confidence = round(min(abs(close - donchian_high) / donchian_high, 1.0), 4)

        if close > donchian_high and volume > volume_ma * self.volume_multiplier:
            signal = Signal.BUY
        elif close < donchian_low:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        return SignalResult(signal=signal, timestamp=timestamp, confidence=confidence)
