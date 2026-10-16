# SPEC: Toy exponential-fit pipeline

## Goal

Generate noisy data from a known function, fit it with SciPy, and write a metrics JSON file. The fitted model has the same form as the generator, so parameter recovery can be checked against known truth. The script must be reproducible and safe to run as a Slurm array job.

Function: `y = a·exp(-b·x) + c`, with Gaussian noise.

Fixed constants (named, module level): `A=2.5, B=1.3, C=0.5, SIGMA=0.1`, x in `[0, 5]`, `P0=(1, 1, 0)`, `N_MIN=10`, `N_MAX=1_000_000`.

Layout: logic in src/toypipe/pipeline.py, command-line entry point in scripts/run_fit.py, tests in tests/test_pipeline.py.

## Inputs and outputs

**CLI:** `python scripts/run_fit.py --n INT --seed INT --out PATH`

- `--n`: number of points, integer, `N_MIN <= n <= N_MAX`.
- `--seed`: integer, `>= 0`.
- `--out`: path to the output JSON file. Parent directories are created as needed.

**Output JSON** (floats rounded to 10 significant digits, keys sorted, no timestamps, hostnames or job IDs):

- Success: `{"status": "ok", "n", "seed", "a", "b", "c", "rmse_true"}`
- Fit failure: `{"status": "failed", "n", "seed", "error"}`

`rmse_true` is the RMSE between the fitted curve and the noise-free true function, evaluated at the x grid. It measures recovery error, not goodness of fit to the noisy data.

Outputs go under `runs/`, which is listed in `.gitignore`.

## Behaviour and edge cases

- Data: `x = linspace(0, 5, n)`; noise from `numpy.random.default_rng(seed)`. Only the noise depends on the seed.
- Fit: `scipy.optimize.curve_fit` with the fixed `P0`.
- **Reproducibility:** the same `n` and `seed` give a byte-identical JSON file on the same environment, and match across cluster nodes thanks to the 10-significant-digit rounding.
- **Invalid input** (`n` or `seed` not an integer, `n` outside range, `seed < 0`): clear message on stderr, nonzero exit, no file written.
- **Output exists:** refuse, print a message, exit nonzero, leave the existing file untouched.
- **Fit failure** (`RuntimeError` or non-finite parameters): write the failed-status JSON, then exit nonzero.
- Distinct exit codes: input error (2), output exists (3), fit failure (4).
- Slurm readiness: no hardcoded absolute paths, no interactive prompts, all messages to stderr/stdout, so job logs are readable. Each array task passes its own `--out` path.

## Out of scope

- Extra flags for a, b, c, sigma or the x-range, and config files.
- A `--force` overwrite flag.
- Misspecified-model fits and model comparison.
- Plots, logging frameworks, and recording input errors in JSON.
- The sbatch script itself, job-array orchestration, and result aggregation.

## How we verify it works

pytest tests, run with `pytest`:

1. **Recovery:** with a fixed seed and large `n` (e.g. 1000), fitted a, b, c are within a stated tolerance of the truth, and `rmse_true` is small.
2. **Reproducibility:** two runs with the same seed produce byte-identical files; different seeds produce different files.
3. **Validation:** `n=9`, `n` above `N_MAX`, `seed=-1` and non-integer values each exit nonzero (code 2) and write no file.
4. **Overwrite guard:** a second run to the same `--out` exits with code 3 and leaves the file unchanged.
5. **Fit failure:** with `curve_fit` monkeypatched to raise `RuntimeError`, the file contains `"status": "failed"` and the exit code is 4.
6. **Output location:** nested `--out` directories are created, and `runs/` is ignored by git (`git check-ignore runs/x.json`).

Before the first cluster run, do one manual `sbatch` smoke test with a 3-task array and distinct seeds.

