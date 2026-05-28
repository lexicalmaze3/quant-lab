import pandas as pd

from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class DummyStrategy(BaseStrategy):
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        return SignalResult(signal=Signal.HOLD, timestamp=df.index[-1])

    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)
