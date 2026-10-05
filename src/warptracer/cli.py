"""Short commands for the common demo and benchmark workflows."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run a car replay or a named benchmark preset")
    parser.add_argument("command", choices=("demo", "benchmark"))
    if not args:
        parser.print_help()
        return
    if args[0] not in ("demo", "benchmark"):
        parser.parse_args(args)
        return
    command, rest = args[0], args[1:]
    if command == "demo":
        from .demo import main as run_demo
        run_demo(rest)
    else:
        from .benchmark import main as run_benchmark
        run_benchmark(rest, default_preset="quick")
