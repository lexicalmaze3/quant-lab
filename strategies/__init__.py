from abc import ABC, abstractmethod
from datetime import date, timedelta

import pandas as pd

from core.data import fetch
from core.signals import SignalResult


class BaseStrategy(ABC):
    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        pass

    @abstractmethod
    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:
        pass

    def get_latest_signal(self, ticker: str) -> SignalResult:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=200)).isoformat()
        df = fetch(ticker, start, end)
        df = self.compute_indicators(df)
        return self.generate_signal(df)
