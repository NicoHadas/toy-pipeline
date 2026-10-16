import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from toypipe import pipeline

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_fit", ROOT / "scripts" / "run_fit.py"
)
run_fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_fit)


def run_cli(n, seed, out):
    return run_fit.main(["--n", str(n), "--seed", str(seed), "--out", str(out)])


def test_recovery(tmp_path):
    out = tmp_path / "r.json"
    assert run_cli(1000, 1, out) == 0
    r = json.loads(out.read_text())
    assert r["status"] == "ok"
    assert abs(r["a"] - pipeline.A) < 0.15
    assert abs(r["b"] - pipeline.B) < 0.15
    assert abs(r["c"] - pipeline.C) < 0.15
    assert r["rmse_true"] < 0.05


def test_result_keys_sorted_and_exact(tmp_path):
    out = tmp_path / "r.json"
    run_cli(100, 0, out)
    r = json.loads(out.read_text())
    assert list(r) == sorted(r)
    assert set(r) == {"status", "n", "seed", "a", "b", "c", "rmse_true"}
    assert r["n"] == 100 and r["seed"] == 0


def test_reproducible(tmp_path):
    run_cli(500, 7, tmp_path / "a.json")
    run_cli(500, 7, tmp_path / "b.json")
    run_cli(500, 8, tmp_path / "c.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert (tmp_path / "a.json").read_bytes() != (tmp_path / "c.json").read_bytes()


@pytest.mark.parametrize(
    "n, seed",
    [
        (9, 1),
        (pipeline.N_MAX + 1, 1),
        (100, -1),
        ("abc", 1),
        ("1.5", 1),
        (100, "x"),
    ],
)
def test_invalid_input(tmp_path, capsys, n, seed):
    out = tmp_path / "sub" / "r.json"
    assert run_cli(n, seed, out) == 2
    assert not out.exists()
    assert not out.parent.exists()
    assert capsys.readouterr().err


def test_overwrite_guard(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert run_cli(100, 1, out) == 0
    before = out.read_bytes()
    assert run_cli(100, 2, out) == 3
    assert out.read_bytes() == before
    assert "exists" in capsys.readouterr().err


def test_fit_failure_runtime_error(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(pipeline, "curve_fit", boom)
    out = tmp_path / "r.json"
    assert run_cli(100, 1, out) == 4
    r = json.loads(out.read_text())
    assert r["status"] == "failed"
    assert set(r) == {"status", "n", "seed", "error"}


def test_fit_failure_non_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "curve_fit", lambda *a, **k: (np.array([np.nan, 1.0, 0.0]), None)
    )
    out = tmp_path / "r.json"
    assert run_cli(100, 1, out) == 4
    assert json.loads(out.read_text())["status"] == "failed"


def test_nested_output_dirs_created(tmp_path):
    out = tmp_path / "x" / "y" / "r.json"
    assert run_cli(100, 1, out) == 0
    assert out.exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_runs_dir_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "runs/x.json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0


def test_round_sig():
    assert pipeline.round_sig(1.23456789012345) == 1.234567890
    assert pipeline.round_sig(123456.789012345) == 123456.7890
    assert pipeline.round_sig(0.0) == 0.0


def test_cli_subprocess_exit_codes(tmp_path):
    out = tmp_path / "r.json"
    cmd = [
        "python",
        str(ROOT / "scripts" / "run_fit.py"),
        "--n",
        "100",
        "--seed",
        "1",
        "--out",
        str(out),
    ]
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 0
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 3
