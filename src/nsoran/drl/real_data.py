"""Frente D -- convert real ns-O-RAN (ns-3) runs into offline ``.npz`` datasets.

A ns-o-ran-gym run directory (``examples/output/<uuid>/``) produced by
``examples/energy_saving.py`` contains a ``database.db`` with the E2 KPMs that
:class:`~nsoran.datalake.SQLiteDatabaseAPI` collected (``lte_cu_cp``,
``lte_cu_up``, ``gnb_cu_cp``, ``gnb_cu_up``, ``du``), plus the ``bsState`` and
``grafana`` tables.  The per-step observation vectors are **not** stored, so
this module rebuilds them with the very same code path used by the live
:class:`~environments.es_env.EnergySavingEnv`
(``_read_ue_kpms`` -> :func:`~environments.es_env.cell_kpis_from_dataframe` ->
:func:`~environments.es_env.build_observation` -> ``MinMaxNormalizer``).  A
converted dataset therefore matches what the environment would have produced
for the same run.

The control actions are read back from ``EsActions.txt`` (the file ns-3 writes
after applying the control), one integer per E2 indication.

Replay convention
-----------------
Transition ``i`` pairs the observation at indication ``ts_i`` with the action
that was active at ``ts_i``, the ``grafana`` reward computed at ``ts_i`` and the
following observation ``obs_{i+1}``.  Each run contributes exactly one episode.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from .config import select_state_features, state_feature_names_for_variant
from .offline import OfflineDataset

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:  # make ``environments``/``nsoran`` importable
    sys.path.insert(0, str(_SRC))

#: Order of the five reward components stored in ``OfflineDataset.reward_components``.
REWARD_COMPONENT_ORDER: tuple[str, ...] = (
    "throughput",
    "energy",
    "rlf",
    "es_on_cost",
    "bs_on",
)

#: Default reward configuration (the PPO-1 row of Table I).  The weights and
#: the normalisation must always be changed together: the paper normalises
#: every reward component with the quantiles of the reference distribution and
#: only then applies the weights of Eq. (2).
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "w1": 0.51,
    "w2": 0.19,
    "w3": 0.20,
    "w4": 0.10,
}
DEFAULT_NORMALIZATION: str = "normal"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "heuristicType": [-1],
    "simTime": [10],
    "ues": [9],
    "indicationPeriodicity": [0.1],
    "controlFileName": ["es_actions_for_ns3.csv"],
}


@dataclass(frozen=True)
class EnvironmentModelConfig:
    """Energy/reward model used when replaying a run (D1-D3 of the plan).

    The defaults are the approved main configuration: EARTH watts energy,
    dense activation cost and per-cell normalisation.  Pass
    ``es_cost_mode="impulse"``, ``energy_model="proxy"`` or
    ``per_cell_normalization=False`` for the ablations.
    """

    energy_model: str = "watts"
    power_model: Optional[Mapping[str, float]] = None
    es_cost_mode: str = "dense"
    per_cell_normalization: bool = True
    per_cell_aggregation: str = "mean"
    p_tx: float = 1.0
    prb_total: float = 139.0

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "energy_model": self.energy_model,
            "power_model": self.power_model,
            "es_cost_mode": self.es_cost_mode,
            "per_cell_normalization": self.per_cell_normalization,
            "per_cell_aggregation": self.per_cell_aggregation,
            "p_tx": self.p_tx,
            "prb_total": self.prb_total,
        }

    def summary(self) -> Dict[str, Any]:
        from environments.es_env import _resolve_power_model

        return {
            "energy_model": self.energy_model,
            "power_model": _resolve_power_model(self.power_model),
            "es_cost_mode": self.es_cost_mode,
            "per_cell_normalization": self.per_cell_normalization,
            "per_cell_aggregation": self.per_cell_aggregation,
            "p_tx": self.p_tx,
            "prb_total": self.prb_total,
        }


def resolve_environment_model(
    model: Optional[Union[EnvironmentModelConfig, Mapping[str, Any]]] = None,
) -> EnvironmentModelConfig:
    """Coerce a mapping/None into an :class:`EnvironmentModelConfig`."""
    if model is None:
        return EnvironmentModelConfig()
    if isinstance(model, EnvironmentModelConfig):
        return model
    return EnvironmentModelConfig(**dict(model))


def _normalise_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a config in the ``{name: [value]}`` shape ``NsOranEnv`` expects."""

    merged: Dict[str, Any] = {key: list(value) for key, value in _DEFAULT_CONFIG.items()}
    if config:
        for key, value in config.items():
            merged[key] = list(value) if isinstance(value, (list, tuple)) else [value]
    return merged


def load_scenario_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a scenario JSON (e.g. ``es_use_case.json``) into ``{name: [value]}``."""

    with open(path, "r", encoding="utf-8") as handle:
        return _normalise_config(json.load(handle))


def _read_control_log(path: Path) -> Dict[int, int]:
    """Parse ``EsActions.txt`` into ``{timestamp: integer_action}``.

    ``EsActions.txt`` has the header ``timestamp,cellId,hoAllowed`` and one row
    per (E2 indication, cell).  The integer action encodes the seven ON/OFF
    bits in :data:`~environments.es_env.CELL_IDS` order (LSB = cell 2).
    """

    from environments.es_env import CELL_IDS, cell_bits_to_action

    if not path.is_file():
        return {}

    per_timestamp: Dict[int, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = int(float(row["timestamp"]))
                cell = int(float(row["cellId"]))
                allowed = int(float(row["hoAllowed"]))
            except (KeyError, TypeError, ValueError):
                continue
            per_timestamp.setdefault(timestamp, {})[cell] = allowed

    actions: Dict[int, int] = {}
    for timestamp, cells in per_timestamp.items():
        bits = [cells.get(cell, 1) for cell in CELL_IDS]
        actions[timestamp] = int(cell_bits_to_action(bits))
    return actions


def _indication_timestamps(datalake: Any) -> List[tuple[int, int]]:
    """Return ``[(step, timestamp), ...]`` in replay order for one run.

    The ``grafana`` table holds one row per environment *step*, but the very
    first indication is only observed by ``reset()`` and never reaches the
    table.  The distinct KPM timestamps are therefore merged in so that the
    dataset covers every indication of the run.
    """

    # Preferred: the ``grafana`` table stores one row per environment step,
    # including the step counter and the KPM timestamp.
    grafana_steps: Dict[int, int] = {}
    try:
        rows = datalake.read_table("grafana")
    except Exception:
        rows = None
    if rows:
        for row in rows:
            try:
                timestamp = int(row[0])
                step = int(row[3])
            except (IndexError, TypeError, ValueError):
                continue
            grafana_steps.setdefault(timestamp, step)

    # Fallback / complement: distinct KPM timestamps, in increasing order.
    kpm_timestamps: set[int] = set()
    for table in ("lte_cu_up", "gnb_cu_up", "lte_cu_cp", "gnb_cu_cp", "du"):
        try:
            rows = datalake.read_table(table)
        except Exception:
            rows = None
        if not rows:
            continue
        for row in rows:
            try:
                kpm_timestamps.add(int(row[0]))
            except (IndexError, TypeError, ValueError):
                continue

    timestamps = sorted(set(grafana_steps) | kpm_timestamps)
    return [
        (grafana_steps.get(timestamp, index + 1), timestamp)
        for index, timestamp in enumerate(timestamps)
    ]


def _resolve_reward_config(
    reward_weights: Optional[Mapping[str, float]],
    normalization: Optional[str],
    calibration_file: Optional[Union[str, Path]],
) -> tuple[Dict[str, float], str, Optional[str]]:
    """Return the (weights, normalization, calibration) triple to use."""

    if reward_weights is None:
        reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
    else:
        reward_weights = {str(k): float(v) for k, v in reward_weights.items()}
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION
    return reward_weights, str(normalization), calibration_file


def raw_reward_components_from_ns3_runs(
    run_dirs: Iterable[Union[str, Path]],
    *,
    scenario_configuration: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    model: Optional[Union[EnvironmentModelConfig, Mapping[str, Any]]] = None,
    include_cells: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect the *raw* reward components of one or more ns-3 runs.

    The paper normalises every component of Eq. (2) with the quantiles of its
    reference distribution.  This helper returns the raw samples (throughput,
    energy, RLF, activation cost and number of active cells, one sample per E2
    indication) so that a calibration file can be fitted on real data instead
    of on the generic fallback ranges.

    With ``include_cells=True`` it also returns one array per component/cell
    (``throughput_cell_2`` ... ``es_on_cost_cell_8``), which is what the D3
    per-cell normalisation needs.
    """

    from environments.es_env import CELL_IDS, EnergySavingEnv
    from nsoran.datalake import SQLiteDatabaseAPI

    model = resolve_environment_model(model)
    if config_path is not None:
        config = load_scenario_config(config_path)
    else:
        config = _normalise_config(scenario_configuration)

    samples: Dict[str, List[float]] = {name: [] for name in REWARD_COMPONENT_ORDER}
    cell_samples: Dict[str, List[float]] = {}
    if include_cells:
        for name in REWARD_COMPONENT_ORDER:
            if name == "bs_on":
                continue
            for cell in CELL_IDS:
                cell_samples[f"{name}_cell_{cell}"] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        environment = EnergySavingEnv(
            ns3_path="",
            scenario_configuration=config,
            output_folder=str(run_dir),
            optimized=False,
            setup_sim=False,
            **model.as_kwargs(),
        )
        datalake = SQLiteDatabaseAPI(str(run_dir), 9, debug=False)
        environment.datalake = datalake
        environment.reset_es_tracking()
        for _step, timestamp in _indication_timestamps(datalake):
            environment.last_timestamp = int(timestamp)
            environment._get_obs()
            # ``_compute_reward`` is what fills ``last_reward_components``.
            environment._compute_reward()
            raw = (environment.last_reward_components or {}).get("raw", {})
            for name in REWARD_COMPONENT_ORDER:
                samples[name].append(float(raw.get(name, 0.0)))
            if include_cells:
                per_cell = (environment._last_raw or {}).get("per_cell", {})
                for name in REWARD_COMPONENT_ORDER:
                    if name == "bs_on":
                        continue
                    values = dict(per_cell.get(name, []))
                    for cell in CELL_IDS:
                        cell_samples[f"{name}_cell_{cell}"].append(
                            float(values.get(cell, 0.0))
                        )

    result = {
        name: np.asarray(values, dtype=float) for name, values in samples.items()
    }
    if include_cells:
        result.update(
            {
                name: np.asarray(values, dtype=float)
                for name, values in cell_samples.items()
            }
        )
    return result


def transitions_from_ns3_run(
    run_dir: Union[str, Path],
    *,
    scenario_configuration: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    reward_weights: Optional[Mapping[str, float]] = None,
    normalization: Optional[str] = None,
    calibration_file: Optional[Union[str, Path]] = None,
    default_action: int = 127,
    model: Optional[Union[EnvironmentModelConfig, Mapping[str, Any]]] = None,
    state_variant: str = "full",
) -> Dict[str, np.ndarray]:
    """Rebuild the per-step transitions of a single ns-O-RAN run.

    Returns a dictionary with ``observations``, ``actions``, ``rewards``,
    ``next_observations``, ``terminals``, ``timestamps``,
    ``reward_components`` and ``raw_components`` (each ``[N, 5]``) -- i.e.
    everything needed to build an
    :class:`~nsoran.drl.offline.OfflineDataset`.

    ``state_variant`` selects the state layout: ``"full"`` / ``"paper85"``
    is the paper's 85-dim state (default); ``"compact"`` is the 70-dim E5
    vector of ``docs/plano_tecnico/kpm_audit.md`` (drops ``mac_pdu_64qam``
    and ``prb_percentage`` per cell plus the global ``bs_on``);
    ``"compact78"`` is the legacy 78-dim ablation.
    """

    from environments.es_env import EnergySavingEnv
    from nsoran.datalake import SQLiteDatabaseAPI

    reward_weights, normalization, calibration_file = _resolve_reward_config(
        reward_weights, normalization, calibration_file
    )
    model = resolve_environment_model(model)

    run_dir = Path(run_dir)
    database = run_dir / "database.db"
    if not database.is_file():
        raise FileNotFoundError(f"No database.db found in {run_dir}")

    if config_path is not None:
        config = load_scenario_config(config_path)
    else:
        config = _normalise_config(scenario_configuration)

    environment = EnergySavingEnv(
        ns3_path="",
        scenario_configuration=config,
        output_folder=str(run_dir),
        optimized=False,
        setup_sim=False,
        reward_weights=reward_weights,
        normalization=normalization,
        calibration_file=calibration_file,
        **model.as_kwargs(),
    )
    num_ues = int(environment.scenario_configuration.get("ues", 0) or 0)
    datalake = SQLiteDatabaseAPI(str(run_dir), num_ues, debug=False)
    environment.datalake = datalake

    actions = _read_control_log(run_dir / "EsActions.txt")
    steps = _indication_timestamps(datalake)

    observations: List[np.ndarray] = []
    rewards: List[float] = []
    components: List[List[float]] = []
    raw_components_values: List[List[float]] = []
    action_values: List[int] = []
    timestamps: List[int] = []

    # ``es_on_cost`` is stateful (it tracks how long each cell has been ON/OFF),
    # so iterate in replay order exactly like the live environment would.
    environment.reset_es_tracking()

    for step, timestamp in steps:
        environment.last_timestamp = int(timestamp)
        environment.num_steps = int(step)
        observation = environment._get_obs()
        reward = environment._compute_reward()
        components_dict = environment.last_reward_components or {}
        # The dataset contract stores the *normalised* reward components (see
        # ``nsoran.drl.offline``): they are the ones the paper's Eq. (2) is
        # applied to, and they are what ``with_rewards_from_components`` needs
        # to re-score the dataset with another Table I agent.
        normalized_components = components_dict.get("normalized", {})
        raw_components = components_dict.get("raw", {})

        observations.append(np.asarray(observation, dtype=np.float32))
        rewards.append(float(reward))
        components.append(
            [
                float(normalized_components.get(name, raw_components.get(name, 0.0)))
                for name in REWARD_COMPONENT_ORDER
            ]
        )
        raw_components_values.append(
            [float(raw_components.get(name, 0.0)) for name in REWARD_COMPONENT_ORDER]
        )
        action_values.append(int(actions.get(int(timestamp), default_action)))
        timestamps.append(int(timestamp))

    if not observations:
        raise ValueError(f"No indications found in {run_dir}")

    obs = np.asarray(observations, dtype=np.float32)
    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]

    if state_variant != "full":
        obs = np.asarray(
            select_state_features(obs, state_variant), dtype=np.float32
        )
        next_obs = np.asarray(
            select_state_features(next_obs, state_variant), dtype=np.float32
        )

    terminals = np.zeros(len(obs), dtype=np.int64)
    terminals[-1] = 1

    return {
        "observations": obs,
        "actions": np.asarray(action_values, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": next_obs,
        "terminals": terminals,
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "reward_components": np.asarray(components, dtype=np.float32),
        "raw_components": np.asarray(raw_components_values, dtype=np.float32),
    }


def _config_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: (value[0] if isinstance(value, (list, tuple)) and value else value)
        for key, value in config.items()
    }


def dataset_from_ns3_run(
    run_dir: Union[str, Path],
    *,
    scenario_configuration: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    reward_weights: Optional[Mapping[str, float]] = None,
    normalization: Optional[str] = None,
    calibration_file: Optional[Union[str, Path]] = None,
    validate: bool = True,
    model: Optional[Union[EnvironmentModelConfig, Mapping[str, Any]]] = None,
    state_variant: str = "full",
) -> OfflineDataset:
    """Convert a single ns-O-RAN run directory into an :class:`OfflineDataset`."""

    run_dir = Path(run_dir)
    model = resolve_environment_model(model)
    transitions = transitions_from_ns3_run(
        run_dir,
        scenario_configuration=scenario_configuration,
        config_path=config_path,
        reward_weights=reward_weights,
        normalization=normalization,
        calibration_file=calibration_file,
        model=model,
        state_variant=state_variant,
    )
    config = (
        load_scenario_config(config_path)
        if config_path is not None
        else _normalise_config(scenario_configuration)
    )
    dataset = OfflineDataset(
        observations=transitions["observations"],
        actions=transitions["actions"],
        rewards=transitions["rewards"],
        next_observations=transitions["next_observations"],
        terminals=transitions["terminals"],
        episode_ids=np.zeros(len(transitions["observations"]), dtype=np.int64),
        timestamps=transitions["timestamps"],
        reward_components=transitions["reward_components"],
        raw_components=transitions["raw_components"],
        metadata={
            "source": "ns3",
            "run_dir": str(run_dir),
            "num_transitions": int(len(transitions["observations"])),
            "config": _config_summary(config),
            "environment_model": model.summary(),
            "state_variant": state_variant,
            "obs_dim": int(transitions["observations"].shape[1]),
            "state_feature_names": list(
                state_feature_names_for_variant(state_variant)
            ),
        },
    )
    return dataset.validate() if validate else dataset


def dataset_from_ns3_runs(
    run_dirs: Iterable[Union[str, Path]],
    *,
    scenario_configuration: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    reward_weights: Optional[Mapping[str, float]] = None,
    normalization: Optional[str] = None,
    calibration_file: Optional[Union[str, Path]] = None,
    validate: bool = True,
    model: Optional[Union[EnvironmentModelConfig, Mapping[str, Any]]] = None,
    state_variant: str = "full",
    run_labels: Optional[Sequence[str]] = None,
) -> OfflineDataset:
    """Concatenate several ns-O-RAN runs, one episode per run.

    ``run_labels`` optionally records the policy that generated each run
    (e.g. ``always_on``, ``static``, ``random``).  The labels are stored in
    the metadata so the E1 QoS constraint can derive its throughput reference
    from the Always-On episodes instead of from a mixed-policy mean.
    """

    model = resolve_environment_model(model)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    next_observations: List[np.ndarray] = []
    terminals: List[np.ndarray] = []
    episode_ids: List[np.ndarray] = []
    timestamps: List[np.ndarray] = []
    components: List[np.ndarray] = []
    raw_components: List[np.ndarray] = []
    source_dirs: List[str] = []

    for episode, run_dir in enumerate(run_dirs):
        transitions = transitions_from_ns3_run(
            run_dir,
            scenario_configuration=scenario_configuration,
            config_path=config_path,
            reward_weights=reward_weights,
            normalization=normalization,
            calibration_file=calibration_file,
            model=model,
            state_variant=state_variant,
        )
        count = len(transitions["observations"])
        observations.append(transitions["observations"])
        actions.append(transitions["actions"])
        rewards.append(transitions["rewards"])
        next_observations.append(transitions["next_observations"])
        terminals.append(transitions["terminals"])
        timestamps.append(transitions["timestamps"])
        components.append(transitions["reward_components"])
        raw_components.append(transitions["raw_components"])
        episode_ids.append(np.full(count, episode, dtype=np.int64))
        source_dirs.append(str(run_dir))

    if not observations:
        raise ValueError("No run directories supplied")

    config = (
        load_scenario_config(config_path)
        if config_path is not None
        else _normalise_config(scenario_configuration)
    )
    labels = list(run_labels) if run_labels is not None else []
    if labels and len(labels) != len(source_dirs):
        raise ValueError(
            "run_labels must have one entry per run directory "
            f"({len(source_dirs)} runs, {len(labels)} labels)."
        )
    dataset = OfflineDataset(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        episode_ids=np.concatenate(episode_ids, axis=0),
        timestamps=np.concatenate(timestamps, axis=0),
        reward_components=np.concatenate(components, axis=0),
        raw_components=np.concatenate(raw_components, axis=0),
        metadata={
            "source": "ns3",
            "num_episodes": len(source_dirs),
            "run_dirs": source_dirs,
            "run_policies": labels,
            "config": _config_summary(config),
            "environment_model": model.summary(),
            "state_variant": state_variant,
            "obs_dim": int(np.concatenate(observations, axis=0).shape[1]),
            "state_feature_names": list(
                state_feature_names_for_variant(state_variant)
            ),
        },
    )
    return dataset.validate() if validate else dataset


__all__ = [
    "REWARD_COMPONENT_ORDER",
    "DEFAULT_REWARD_WEIGHTS",
    "DEFAULT_NORMALIZATION",
    "EnvironmentModelConfig",
    "resolve_environment_model",
    "raw_reward_components_from_ns3_runs",
    "load_scenario_config",
    "transitions_from_ns3_run",
    "dataset_from_ns3_run",
    "dataset_from_ns3_runs",
]
