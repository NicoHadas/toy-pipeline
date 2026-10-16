# toy-pipeline

Toy exponential-fit pipeline; practice project for the computational workflow.

## Commands
- Environment: `conda activate toy-pipeline`
- Tests: `pytest`
- Lint: `ruff check .` and `ruff format .`
- Run: `python scripts/run_fit.py --n 1000 --seed 1 --out runs/trial1/result.json`

## Conventions
- Spec: `docs/SPEC.md`. Follow it; don't edit it without asking
- Logic in `src/toypipe/`, entry points in `scripts/`, tests in `tests/`
- Outputs go in `runs/` (gitignored); never commit data or outputs
- Work on a branch; don't commit or push unless asked
