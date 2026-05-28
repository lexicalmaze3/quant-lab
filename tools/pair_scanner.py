#!/usr/bin/env python3
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint

START = "2010-01-01"
END = "2024-01-01"

pairs = [
    ("XLF", "XLK"),
    ("GLD", "SLV"),
    ("KO", "PEP"),
    ("XOM", "CVX"),
    ("RTX", "NOC"),
]


def fetch_close(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].rename(ticker)


def analyze_pair(ticker_a, ticker_b):
    a = fetch_close(ticker_a)
    b = fetch_close(ticker_b)
    df = pd.concat([a, b], axis=1).dropna()

    _, pvalue, _ = coint(df[ticker_a], df[ticker_b])

    X = sm.add_constant(df[ticker_b])
    hedge_ratio = sm.OLS(df[ticker_a], X).fit().params[ticker_b]
    spread = df[ticker_a] - hedge_ratio * df[ticker_b]

    spread_mean = abs(spread.mean())
    spread_vol = spread.std() / spread_mean if spread_mean > 1e-8 else float("nan")

    return pvalue, spread_vol


def main():
    rows = []
    for ticker_a, ticker_b in pairs:
        pvalue, spread_vol = analyze_pair(ticker_a, ticker_b)
        verdict = "COINTEGRATED" if pvalue < 0.05 else "NOT COINTEGRATED"
        rows.append((f"{ticker_a} / {ticker_b}", pvalue, spread_vol, verdict))

    rows.sort(key=lambda r: r[1])

    print(f"{'Pair':<16}{'P-Value':<12}{'Spread Vol':<14}Verdict")
    print("-" * 56)
    for pair, pvalue, spread_vol, verdict in rows:
        print(f"{pair:<16}{pvalue:<12.4f}{spread_vol:<14.2f}{verdict}")


if __name__ == "__main__":
    main()
