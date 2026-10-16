"""Command-line entry point: python scripts/run_fit.py --n N --seed S --out PATH"""

import argparse
import sys

from toypipe import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="number of points")
    parser.add_argument("--seed", type=int, required=True, help="noise seed, >= 0")
    parser.add_argument("--out", required=True, help="output JSON path")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments
        return exc.code

    try:
        pipeline.run(args.n, args.seed, args.out)
    except pipeline.InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return pipeline.EXIT_INPUT
    except pipeline.OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return pipeline.EXIT_EXISTS
    except pipeline.FitError as exc:
        print(f"error: fit failed: {exc}", file=sys.stderr)
        return pipeline.EXIT_FIT
    print(f"wrote {args.out}")
    return pipeline.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
