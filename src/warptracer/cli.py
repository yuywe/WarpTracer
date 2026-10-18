"""Short commands for the common demo and benchmark workflows."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run a car replay, benchmark, or PPO training")
    parser.add_argument("command", choices=("demo", "benchmark", "train", "evaluate", "map"))
    if not args:
        parser.print_help()
        return
    if args[0] not in ("demo", "benchmark", "train", "evaluate", "map"):
        parser.parse_args(args)
        return
    command, rest = args[0], args[1:]
    if command == "map":
        from .maps import main as inspect_map
        inspect_map(rest)
        return
    if command == "demo":
        from .demo import main as run_demo
        run_demo(rest)
    elif command == "benchmark":
        from .benchmark import main as run_benchmark
        run_benchmark(rest, default_preset="quick")
    else:
        try:
            from .train import main as train, evaluate
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            parser.error("PPO requires the rl extra: uv run --extra rl warptracer " + command)
        (train if command == "train" else evaluate)(rest)
