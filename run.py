import argparse
import importlib
import sys


def snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Strategy"


def _dual_backtest(strategy, ticker_a, ticker_b, start, end):
    import vectorbt as vbt
    from core.backtest import BacktestResult
    from core.data import fetch

    df_a = fetch(ticker_a, start, end)[["Close"]].rename(columns={"Close": "Close_A"})
    df_b = fetch(ticker_b, start, end)[["Close"]].rename(columns={"Close": "Close_B"})
    df = df_a.join(df_b, how="inner")

    df = strategy.compute_indicators(df)
    coint_pvalue = float(df["coint_pvalue"].iloc[-1]) if "coint_pvalue" in df.columns else None

    entries_long, exits_long, entries_short, exits_short = strategy.generate_signals_series(df)

    pf = vbt.Portfolio.from_signals(
        close=df["Close_A"],
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        freq="1D",
    )

    n_trades = int(pf.trades.count())
    try:
        win_rate = float(pf.trades.win_rate())
    except Exception:
        win_rate = 0.0

    return BacktestResult(
        sharpe=float(pf.sharpe_ratio()),
        max_drawdown=float(pf.max_drawdown()),
        win_rate=win_rate,
        total_return=float(pf.total_return()),
        n_trades=n_trades,
    ), coint_pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    class_name = snake_to_pascal(args.strategy)
    try:
        module = importlib.import_module(f"strategies.{args.strategy}")
        cls = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError):
        print(f"Error: strategy '{args.strategy}' not found (expected class '{class_name}' in strategies/{args.strategy}.py)")
        sys.exit(1)

    dual_ticker = "-" in args.ticker
    params = {}
    if dual_ticker:
        ticker_a, ticker_b = args.ticker.split("-", 1)
        params = {"ticker_a": ticker_a, "ticker_b": ticker_b}

    strategy = cls(params) if params else cls()

    if args.live:
        result = strategy.get_latest_signal(args.ticker)
        print(f"Signal: {result.signal.name} | Confidence: {result.confidence} | Timestamp: {result.timestamp}")
    else:
        from core.report import print_report
        coint_pvalue = None
        if dual_ticker:
            result, coint_pvalue = _dual_backtest(strategy, ticker_a, ticker_b, args.start, args.end)
        else:
            from core.backtest import run_backtest
            result = run_backtest(strategy, args.ticker, args.start, args.end)
        print_report(args.ticker, args.strategy, result)
        if coint_pvalue is not None:
            print(f"Coint p-value: {coint_pvalue:.4f}")


if __name__ == "__main__":
    main()
