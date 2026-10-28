"""Deep Reinforcement Learning package for the paper implementation (Frente C).

The package is deliberately lazy: ``import nsoran.drl`` only loads this file
and therefore works with the base installation.  NumPy/Gymnasium, PyTorch,
Stable Baselines3 and matplotlib are imported on first access to the relevant
symbol, with an actionable ``ImportError`` when an optional dependency is
missing.

Quick start
-----------
::

    from nsoran.drl import load_offline_dataset, OfflineReplayEnv
    from nsoran.drl import PAPER_AGENTS, get_agent_config

Install the optional dependencies with::

    pip install -e '.[drl]'          # or
    pip install -r requirements-drl.txt

CLI::

    python examples/train_offline.py --help
    python examples/evaluate_policy.py --help
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict

_EXPORTS = {
    # config.py (standard library only)
    "STATE_DIM": (".config", "STATE_DIM"),
    "N_ACTIONS": (".config", "N_ACTIONS"),
    "N_CELLS": (".config", "N_CELLS"),
    "CELL_IDS": (".config", "CELL_IDS"),
    "CELL_KPM_NAMES": (".config", "CELL_KPM_NAMES"),
    "CELL_KPM_INDEX": (".config", "CELL_KPM_INDEX"),
    "AgentConfig": (".config", "AgentConfig"),
    "PAPER_AGENTS": (".config", "PAPER_AGENTS"),
    "AGENT_NAMES": (".config", "AGENT_NAMES"),
    "get_agent_config": (".config", "get_agent_config"),
    "compute_reward": (".config", "compute_reward"),
    "action_to_cells": (".config", "action_to_cells"),
    "cells_to_action": (".config", "cells_to_action"),
    "action_to_bitstring": (".config", "action_to_bitstring"),
    # offline.py
    "OfflineDataset": (".offline", "OfflineDataset"),
    "DatasetValidationError": (".offline", "DatasetValidationError"),
    "load_offline_dataset": (".offline", "load_offline_dataset"),
    "save_offline_dataset": (".offline", "save_offline_dataset"),
    "split_dataset": (".offline", "split_dataset"),
    "synthetic_dataset": (".offline", "synthetic_dataset"),
    "OfflineReplayEnv": (".offline", "OfflineReplayEnv"),
    # real_data.py (Frente D: ns-3 run -> offline dataset)
    "REWARD_COMPONENT_ORDER": (".real_data", "REWARD_COMPONENT_ORDER"),
    "load_scenario_config": (".real_data", "load_scenario_config"),
    "dataset_from_ns3_run": (".real_data", "dataset_from_ns3_run"),
    "dataset_from_ns3_runs": (".real_data", "dataset_from_ns3_runs"),
    "transitions_from_ns3_run": (".real_data", "transitions_from_ns3_run"),
    # rem_dqn.py
    "REMDQNConfig": (".rem_dqn", "REMDQNConfig"),
    "REMDQNAgent": (".rem_dqn", "REMDQNAgent"),
    "OfflineReplayBuffer": (".rem_dqn", "OfflineReplayBuffer"),
    "train_rem_dqn": (".rem_dqn", "train_rem_dqn"),
    # ppo.py
    "EarlyStopCallback": (".ppo", "EarlyStopCallback"),
    "build_ppo_model": (".ppo", "build_ppo_model"),
    "train_ppo": (".ppo", "train_ppo"),
    "load_ppo": (".ppo", "load_ppo"),
    "predict_action": (".ppo", "predict_action"),
    # evaluate.py
    "EpisodeMetrics": (".evaluate", "EpisodeMetrics"),
    "EvaluationResult": (".evaluate", "EvaluationResult"),
    "extract_step_metrics": (".evaluate", "extract_step_metrics"),
    "evaluate_policy": (".evaluate", "evaluate_policy"),
    "summarize_result": (".evaluate", "summarize_result"),
    "save_result": (".evaluate", "save_result"),
    "plot_results": (".evaluate", "plot_results"),
    "make_ns3_env": (".evaluate", "make_ns3_env"),
    "make_dataset_env": (".evaluate", "make_dataset_env"),
}

OPTIONAL_DEPENDENCIES = (
    "numpy",
    "gymnasium",
    "torch",
    "stable_baselines3",
    "matplotlib",
    "pytest",
)

__all__ = sorted(_EXPORTS) + ["dependency_status"]


def dependency_status() -> Dict[str, bool]:
    """Return ``{module: installed}`` for the optional DRL dependencies."""

    return {
        name: importlib.util.find_spec(name) is not None
        for name in OPTIONAL_DEPENDENCIES
    }


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            "Use nsoran.drl.dependency_status() to check optional dependencies."
        ) from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> Any:
    return sorted(set(globals()) | set(_EXPORTS))
