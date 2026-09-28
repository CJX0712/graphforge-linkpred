"""CLI 冒烟测试（子进程真跑）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-m", "graphforge.cli", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


def test_cli_info():
    result = _run(["info"])
    assert result.returncode == 0, result.stderr
    assert "capability" in result.stdout.lower()
    assert "tier" in result.stdout


def test_cli_operators_and_heuristics():
    for command in ("operators", "heuristics"):
        result = _run([command])
        assert result.returncode == 0
        assert result.stdout.strip()


def test_cli_run_karate_smoke(tmp_path):
    out = tmp_path / "run.json"
    result = _run([
        "run", "--graph", "karate", "--seed", "0", "--dim", "8",
        "--walk-length", "8", "--num-walks", "2", "--window", "2",
        "--negative", "3", "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert "AUC=" in result.stdout
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["graph"]["name"] == "karate"
    assert 0.0 <= payload["metrics"]["auc"] <= 1.0


def test_cli_run_tier1_offline_fallback(tmp_path):
    out = tmp_path / "tier1.json"
    result = _run([
        "run", "--graph", "karate", "--seed", "0", "--dim", "8",
        "--walk-length", "8", "--num-walks", "2", "--window", "2",
        "--negative", "3", "--tier1-only", "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["capabilities"]["tier"] == "tier1"


def test_cli_benchmark_smoke(tmp_path):
    out = tmp_path / "bench.json"
    result = _run([
        "benchmark", "--graph", "karate", "--seeds", "0", "1", "2",
        "--dim", "8", "--walk-length", "8", "--num-walks", "2", "--window", "2",
        "--negative", "3", "--output", str(out),
    ], timeout=300)
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    methods = {row["method"] for row in payload["results"]}
    assert "graphforge-node2vec" in methods
    assert "adamic_adar" in methods
    assert len(payload["per_seed"]["graphforge-node2vec"]) == 3
    assert payload["comparisons"]


def test_cli_rejects_bad_operator():
    result = _run(["run", "--graph", "karate", "--operator", "not-a-real-operator"])
    assert result.returncode != 0


@pytest.mark.parametrize("bad", [
    ["run", "--graph", "karate", "--dim", "1"],
    ["run", "--graph", "karate", "--epochs", "0"],
])
def test_cli_rejects_invalid_config(bad):
    result = _run(bad)
    assert result.returncode != 0
