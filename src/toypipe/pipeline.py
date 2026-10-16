"""Generate noisy exponential-decay data, fit it, and write a metrics JSON."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

A = 2.5
B = 1.3
C = 0.5
SIGMA = 0.1
X_MIN = 0.0
X_MAX = 5.0
P0 = (1, 1, 0)
N_MIN = 10
N_MAX = 1_000_000

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_EXISTS = 3
EXIT_FIT = 4


class InputError(ValueError):
    """Invalid n or seed."""


class OutputExistsError(FileExistsError):
    """The output file already exists."""


class FitError(RuntimeError):
    """The fit did not converge or returned non-finite parameters."""


def model(x, a, b, c):
    return a * np.exp(-b * x) + c


def validate_inputs(n, seed):
    for name, value in (("n", n), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InputError(f"{name} must be an integer, got {value!r}")
    if not N_MIN <= n <= N_MAX:
        raise InputError(f"n must be between {N_MIN} and {N_MAX}, got {n}")
    if seed < 0:
        raise InputError(f"seed must be >= 0, got {seed}")


def make_data(n, seed):
    x = np.linspace(X_MIN, X_MAX, n)
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, n)
    return x, model(x, A, B, C) + noise


def fit(x, y):
    try:
        popt, _ = curve_fit(model, x, y, p0=P0)
    except RuntimeError as exc:
        raise FitError(str(exc)) from exc
    if not np.all(np.isfinite(popt)):
        raise FitError("fit returned non-finite parameters")
    return tuple(float(p) for p in popt)


def rmse_true(x, params):
    return float(np.sqrt(np.mean((model(x, *params) - model(x, A, B, C)) ** 2)))


def round_sig(value, digits=10):
    return float(f"{value:.{digits - 1}e}")


def build_result(n, seed):
    """Return (result_dict, ok)."""
    x, y = make_data(n, seed)
    try:
        params = fit(x, y)
    except FitError as exc:
        return {"status": "failed", "n": n, "seed": seed, "error": str(exc)}, False
    a, b, c = params
    result = {
        "status": "ok",
        "n": n,
        "seed": seed,
        "a": round_sig(a),
        "b": round_sig(b),
        "c": round_sig(c),
        "rmse_true": round_sig(rmse_true(x, params)),
    }
    return result, True


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Mode "x" fails atomically if the file exists (safe for array jobs).
        with open(path, "x") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise OutputExistsError(f"output file already exists: {path}") from exc


def run(n, seed, out):
    """Run the pipeline; raise InputError, OutputExistsError or FitError.

    On FitError the failed-status JSON has already been written.
    """
    validate_inputs(n, seed)
    if Path(out).exists():
        raise OutputExistsError(f"output file already exists: {out}")
    result, ok = build_result(n, seed)
    write_result(out, result)
    if not ok:
        raise FitError(result["error"])
    return EXIT_OK
