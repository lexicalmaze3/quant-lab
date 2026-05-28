import argparse
import importlib
import sys


def snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Strategy"


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

    strategy = cls()

    if args.live:
        result = strategy.get_latest_signal(args.ticker)
        print(f"Signal: {result.signal.name} | Confidence: {result.confidence} | Timestamp: {result.timestamp}")
    else:
        from core.backtest import run_backtest
        from core.report import print_report
        result = run_backtest(strategy, args.ticker, args.start, args.end)
        print_report(args.ticker, args.strategy, result)


if __name__ == "__main__":
    main()
