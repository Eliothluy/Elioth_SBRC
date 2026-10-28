"""Smoke tests for the per-agent DRL example scripts (Frente C).

The scripts ``examples/ppo_1.py`` ... ``examples/ppo_5.py`` and
``examples/dqn.py`` must have a working ``--help`` and validate arguments
before any optional DRL dependency is imported.  These tests therefore only
need the standard library plus pytest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"

_AGENT_SCRIPTS = (
    "ppo_1.py",
    "ppo_2.py",
    "ppo_3.py",
    "ppo_4.py",
    "ppo_5.py",
    "dqn.py",
)

sys.path.insert(0, str(_EXAMPLES))
import _agent_cli


@pytest.mark.parametrize("script", _AGENT_SCRIPTS)
def test_help_works_without_optional_dependencies(script):
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES / script), "--help"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    assert "Table I" in result.stdout


@pytest.mark.parametrize("script", _AGENT_SCRIPTS)
def test_missing_data_source_fails_cleanly(script):
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES / script)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "--dataset" in result.stderr


def test_help_exits_zero_through_run_agent(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _agent_cli.run_agent("PPO-1", ["--help"])
    assert excinfo.value.code == 0
    assert "Table I" in capsys.readouterr().out


def test_eval_only_requires_a_model():
    with pytest.raises(SystemExit) as excinfo:
        _agent_cli.run_agent("DQN", ["--eval-only"])
    assert excinfo.value.code == 2


def test_eval_only_requires_an_environment():
    with pytest.raises(SystemExit) as excinfo:
        _agent_cli.run_agent(
            "PPO-2", ["--eval-only", "--model", "runs/PPO-2/best_model.zip"]
        )
    assert excinfo.value.code == 2


def test_dataset_and_synthetic_are_mutually_exclusive(tmp_path):
    dataset = tmp_path / "offline.npz"
    dataset.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        _agent_cli.run_agent("PPO-3", ["--dataset", str(dataset), "--synthetic", "100"])
    assert excinfo.value.code == 2


def test_unknown_agent_raises_key_error():
    with pytest.raises(KeyError):
        _agent_cli.run_agent("PPO-9", ["--help"])


def test_parser_defaults_match_the_paper():
    from nsoran.drl.config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_TOTAL_TIMESTEPS,
        get_agent_config,
    )

    parser = _agent_cli.build_parser(get_agent_config("PPO-3"))
    args = parser.parse_args(["--dataset", "data/offline.npz"])
    assert args.total_timesteps == DEFAULT_TOTAL_TIMESTEPS
    assert args.batch_size == DEFAULT_BATCH_SIZE
    assert args.apply_reward_config is True
    assert args.evaluate is False
    assert args.eval_only is False

    dqn_parser = _agent_cli.build_parser(get_agent_config("DQN"))
    dqn_args = dqn_parser.parse_args(["--dataset", "data/offline.npz"])
    assert dqn_args.k_heads == 3
    assert dqn_args.cql_alpha == 0.5
