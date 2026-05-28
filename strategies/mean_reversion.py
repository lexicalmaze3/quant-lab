import pandas as pd
import ta

from core.signals import Signal, SignalResult
from strategies import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.bb_period = params.get("bb_period", 20)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_entry = params.get("rsi_entry", 35)
        self.rsi_exit = params.get("rsi_exit", 65)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bb = ta.volatility.BollingerBands(close=df["Close"], window=self.bb_period)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=self.rsi_period).rsi()
        return df

    def generate_signals_series(self, df: pd.DataFrame) -> pd.Series:
        in_position = False
        signals = []
        for i in range(len(df)):
            close = df["Close"].iloc[i]
            lower = df["bb_lower"].iloc[i]
            upper = df["bb_upper"].iloc[i]
            rsi = df["rsi"].iloc[i]

            if pd.isna(lower) or pd.isna(upper) or pd.isna(rsi):
                signals.append(False)
                continue

            if not in_position:
                if close < lower and rsi < self.rsi_entry:
                    in_position = True
            else:
                if close > upper or rsi > self.rsi_exit:
                    in_position = False

            signals.append(in_position)

        return pd.Series(signals, index=df.index)

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        row = df.iloc[-1]
        close = row["Close"]
        lower = row["bb_lower"]
        upper = row["bb_upper"]
        rsi = row["rsi"]
        timestamp = df.index[-1]

        confidence = abs(rsi - 50) / 50 if not pd.isna(rsi) else 0.0

        if not pd.isna(lower) and not pd.isna(rsi) and close < lower and rsi < self.rsi_entry:
            signal = Signal.BUY
        elif not pd.isna(upper) and not pd.isna(rsi) and close > upper and rsi > self.rsi_exit:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        return SignalResult(signal=signal, timestamp=timestamp, confidence=round(confidence, 4))
