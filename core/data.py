from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("./data/cache")


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_parquet(cache_path)
    return df
