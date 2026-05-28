from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    signal: Signal
    timestamp: pd.Timestamp
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
