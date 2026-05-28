from core.backtest import BacktestResult


def print_report(ticker: str, strategy_name: str, result: BacktestResult) -> None:
    print(f"Strategy : {strategy_name}")
    print(f"Ticker   : {ticker}")
    print(f"Sharpe   : {result.sharpe:.2f}")
    print(f"Drawdown : {result.max_drawdown * 100:.1f}%")
    print(f"Win Rate : {result.win_rate * 100:.1f}%")
    print(f"Return   : {result.total_return * 100:.1f}%")
    print(f"Trades   : {result.n_trades}")
