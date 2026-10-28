"""Policy evaluation and paper-style aggregation/plots.

The evaluator supports two environments:

* a live ns-3/Gymnasium ``EnergySavingEnv`` (when ns-3 is built and available);
* an offline ``.npz`` dataset replayed with :class:`OfflineReplayEnv`.

The latter is a fallback for development machines: the policy's actions are
recorded for diagnostics but the replayed transitions are fixed, so the
resulting KPMs characterise the dataset, not the policy.  A warning is written
to the summary in that case.

Aggregated metrics follow Section VI: throughput, energy, RLF and activation
cost.  CDF and throughput-vs-energy trade-off plots are produced only when
matplotlib is installed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # optional dependency group: drl
    import numpy as np
except ImportError:  # keep ``--help`` usable without optional dependencies
    np = None  # type: ignore[assignment]

_NUMPY_ERROR = (
    "nsoran.drl.evaluate requires numpy. Install the DRL extras with "
    "`pip install -e '.[drl]'` or `pip install -r requirements-drl.txt`."
)


def _require_numpy() -> None:
    if np is None:
        raise ImportError(_NUMPY_ERROR)

from .config import (
    CELL_KPM_INDEX,
    GLOBAL_BS_ON_INDEX,
    N_CELLS,
    STATE_DIM,
    get_agent_config,
)

__all__ = [
    "EpisodeMetrics",
    "EvaluationResult",
    "extract_step_metrics",
    "evaluate_policy",
    "summarize_result",
    "save_result",
    "plot_results",
    "make_ns3_env",
    "make_dataset_env",
    "main",
]

#: Flexible key aliases for info dictionaries produced by Frente B.
_INFO_KEYS = {
    "throughput": (
        "throughput",
        "sum_throughput",
        "total_throughput",
        "SUM_QosFlow.PdcpPduVolumeDL_Filter",
        "sum_qosflow.pdcppduvolumedl_filter",
    ),
    "energy": (
        "energy",
        "en_cons",
        "consumption",
        "sum_energy",
        "total_energy",
        "SUM_TB.TotNbrDl.1",
        "sum_tb.totnbrdl.1",
    ),
    "rlf": (
        "rlf",
        "rlf_count",
        "sum_rlf",
        "SUM_RLF_VALUE",
        "sum_rlf_value",
    ),
    "activation_cost": (
        "activation_cost",
        "on_cost",
        "es_on_cost",
        "sum_es_on_cost",
        "SUM_ES_ON_COST",
        "sum_es_on_cost",
    ),
    "bs_on": (
        "bs_on",
        "bs_on_count",
        "num_bs_on",
        "b_son",
    ),
}


def _to_float(value: Any, reduction: str = "sum") -> Optional[float]:
    _require_numpy()
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("sum", "total", "value", "mean", "avg"):
            if key in value:
                return _to_float(value[key], reduction)
        value = list(value.values())
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    if reduction == "mean":
        return float(array.mean())
    if reduction == "last":
        return float(array[-1])
    return float(array.sum())


def _lookup_info(info: Mapping[str, Any], names: Sequence[str]) -> Any:
    if not isinstance(info, Mapping):
        return None
    sources: List[Mapping[str, Any]] = [info]
    for key in ("kpis", "metrics", "raw_kpms", "raw_kpis", "kpm"):
        nested = info.get(key)
        if not isinstance(nested, Mapping):
            continue
        sources.append(nested)
        # The energy-saving environment exposes the raw KPM totals one level
        # deeper ("raw_kpms" -> "totals"); without this the evaluator would
        # silently fall back to the min-max normalised observation proxy.
        totals = nested.get("totals")
        if isinstance(totals, Mapping):
            sources.append(totals)
    for source in sources:
        source_lowered = {str(key).lower(): key for key in source}
        for name in names:
            key = source_lowered.get(name.lower())
            if key is not None:
                return source[key]
    return None


def extract_step_metrics(
    observation: Any,
    info: Optional[Mapping[str, Any]] = None,
    *,
    action: Optional[int] = None,
    previous_action: Optional[int] = None,
    reward: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract the four paper KPMs from ``info`` or, as a proxy, from the state.

    Observation layout: 7 cells x 12 KPMs + global ``bs_on`` (index 84).
    """

    _require_numpy()
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    if obs.size != STATE_DIM:
        raise ValueError(
            f"observation must have {STATE_DIM} features, got {obs.size}."
        )
    cells = obs[: N_CELLS * 12].reshape(N_CELLS, 12)
    info = info or {}

    sources: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}

    for metric_name, aliases in _INFO_KEYS.items():
        value = _lookup_info(info, aliases)
        numeric = _to_float(value, reduction="sum")
        if numeric is not None:
            metrics[metric_name] = numeric
            sources[metric_name] = "info"

    if "throughput" not in metrics:
        metrics["throughput"] = float(cells[:, CELL_KPM_INDEX["throughput_dl"]].sum())
        sources["throughput"] = "observation_proxy"
    if "energy" not in metrics:
        metrics["energy"] = float(cells[:, CELL_KPM_INDEX["energy"]].sum())
        sources["energy"] = "observation_proxy"
    if "rlf" not in metrics:
        metrics["rlf"] = float(cells[:, CELL_KPM_INDEX["rlf_count"]].sum())
        sources["rlf"] = "observation_proxy"
    if "activation_cost" not in metrics:
        metrics["activation_cost"] = float(
            cells[:, CELL_KPM_INDEX["es_on_cost"]].sum()
        )
        sources["activation_cost"] = "observation_proxy"
    if "bs_on" not in metrics:
        metrics["bs_on"] = float(obs[GLOBAL_BS_ON_INDEX])
        sources["bs_on"] = "observation_proxy"

    metrics["action"] = None if action is None else int(action)
    metrics["action_switches"] = (
        0
        if action is None or previous_action is None
        else int(int(action) != int(previous_action))
    )
    metrics["reward"] = None if reward is None else float(reward)
    metrics["sources"] = sources
    return metrics


@dataclass
class EpisodeMetrics:
    episode: int
    steps: int
    throughput: float
    energy: float
    rlf: float
    activation_cost: float
    bs_on_mean: float
    reward: float
    action_switches: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "steps": self.steps,
            "throughput": self.throughput,
            "energy": self.energy,
            "rlf": self.rlf,
            "activation_cost": self.activation_cost,
            "bs_on_mean": self.bs_on_mean,
            "reward": self.reward,
            "action_switches": self.action_switches,
        }


@dataclass
class EvaluationResult:
    algo: str
    model_path: str
    environment: str
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    step_metrics: Dict[str, List[float]] = field(default_factory=dict)
    actions: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metric_sources: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algo": self.algo,
            "model_path": self.model_path,
            "environment": self.environment,
            "notes": self.notes,
            "metric_sources": self.metric_sources,
            "episodes": [episode.as_dict() for episode in self.episodes],
            "summary": summarize_result(self),
        }


def _reset_env(env: Any, seed: Optional[int]) -> Tuple[Any, Mapping[str, Any]]:
    if seed is not None:
        try:
            signature = inspect.signature(env.reset)
            supports_seed = "seed" in signature.parameters
        except (TypeError, ValueError):
            supports_seed = True
        if supports_seed:
            try:
                result = env.reset(seed=seed)
                return result[0], result[1] if len(result) > 1 else {}
            except TypeError:
                pass
    result = env.reset()
    return result[0], result[1] if len(result) > 1 else {}


def _load_policy(algo: str, model_path: Union[str, os.PathLike], env: Any):
    algo = str(algo).lower()
    if algo == "ppo":
        from .ppo import load_ppo

        return "ppo", load_ppo(model_path, env=env)
    if algo == "iql":
        from .iql import IQLAgent

        return "iql", IQLAgent.load(model_path)
    if algo in ("rem_dqn", "dqn", "rem-dqn"):
        from .rem_dqn import REMDQNAgent

        return "rem_dqn", REMDQNAgent.load(model_path)
    raise ValueError(
        f"Unsupported algorithm {algo!r}; expected 'ppo', 'iql' or 'rem_dqn'."
    )


def evaluate_policy(
    algo: str,
    model_path: Union[str, os.PathLike],
    env: Any,
    *,
    episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
    max_steps_per_episode: int = 10_000,
    environment_label: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
) -> EvaluationResult:
    """Run ``episodes`` and aggregate throughput/energy/RLF/activation cost."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    _require_numpy()

    resolved_algo, policy = _load_policy(algo, model_path, env)
    source = environment_label or env.__class__.__name__
    result = EvaluationResult(
        algo=resolved_algo,
        model_path=str(model_path),
        environment=source,
        step_metrics={
            "throughput": [],
            "energy": [],
            "rlf": [],
            "activation_cost": [],
            "bs_on": [],
            "reward": [],
            "action": [],
        },
        notes=list(notes or []),
        actions=[],
    )
    if resolved_algo == "rem_dqn":
        result.notes.append(
            "REM-DQN actions use ensemble-mean Q-values "
            f"({'greedy' if deterministic else 'epsilon-greedy'})."
        )
    elif resolved_algo == "iql":
        result.notes.append(
            "IQL is deterministic: actions are the argmax of the AWR policy."
        )

    for episode_index in range(int(episodes)):
        observation, info = _reset_env(env, seed + episode_index)
        previous_action: Optional[int] = None
        total_throughput = 0.0
        total_energy = 0.0
        total_rlf = 0.0
        total_activation = 0.0
        total_reward = 0.0
        bs_on_values: List[float] = []
        switches = 0
        steps = 0

        for step_index in range(int(max_steps_per_episode)):
            if resolved_algo in ("ppo", "iql"):
                action_array, _ = policy.predict(
                    observation, deterministic=deterministic
                )
                action = int(np.asarray(action_array).reshape(-1)[0])
            else:
                action = int(
                    policy.act(
                        observation,
                        step=step_index,
                        deterministic=deterministic,
                    )
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = env.step(action)
            step_metrics = extract_step_metrics(
                next_observation,
                step_info,
                action=action,
                previous_action=previous_action,
                reward=reward,
            )
            result.metric_sources.update(step_metrics["sources"])

            total_throughput += step_metrics["throughput"]
            total_energy += step_metrics["energy"]
            total_rlf += step_metrics["rlf"]
            total_activation += step_metrics["activation_cost"]
            total_reward += float(reward)
            bs_on_values.append(step_metrics["bs_on"])
            switches += step_metrics["action_switches"]
            steps += 1

            for name in ("throughput", "energy", "rlf", "activation_cost", "bs_on"):
                result.step_metrics[name].append(step_metrics[name])
            result.step_metrics["reward"].append(float(reward))
            result.step_metrics["action"].append(action)
            result.actions.append(action)

            previous_action = action
            observation = next_observation
            if terminated or truncated:
                break

        result.episodes.append(
            EpisodeMetrics(
                episode=episode_index,
                steps=steps,
                throughput=total_throughput,
                energy=total_energy,
                rlf=total_rlf,
                activation_cost=total_activation,
                bs_on_mean=(
                    float(np.mean(bs_on_values)) if bs_on_values else 0.0
                ),
                reward=total_reward,
                action_switches=switches,
            )
        )

    if "observation_proxy" in result.metric_sources.values():
        result.notes.append(
            "Some KPMs were read as proxies from the normalized 85-dim state "
            "because the environment info dict did not expose raw values."
        )
    return result


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    _require_numpy()
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std())}


def summarize_result(result: EvaluationResult) -> Dict[str, Any]:
    _require_numpy()
    fields = (
        "throughput",
        "energy",
        "rlf",
        "activation_cost",
        "bs_on_mean",
        "reward",
        "steps",
        "action_switches",
    )
    summary: Dict[str, Any] = {
        "episodes": len(result.episodes),
        "algo": result.algo,
        "environment": result.environment,
        "metrics": {},
        "per_step_means": {},
    }
    for name in fields:
        summary["metrics"][name] = _mean_std(
            getattr(episode, name) for episode in result.episodes
        )
    for name, values in result.step_metrics.items():
        array = np.asarray(values, dtype=np.float64)
        summary["per_step_means"][name] = (
            0.0 if array.size == 0 else float(array.mean())
        )
    summary["throughput_per_energy"] = (
        summary["metrics"]["throughput"]["mean"]
        / summary["metrics"]["energy"]["mean"]
        if summary["metrics"]["energy"]["mean"] > 0
        else None
    )
    summary["notes"] = list(result.notes)
    summary["metric_sources"] = dict(result.metric_sources)
    return summary


def save_result(
    result: EvaluationResult,
    output_dir: Union[str, os.PathLike],
    *,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Write JSON/NumPy artifacts and optionally CDF/trade-off plots."""

    _require_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    if result.step_metrics:
        np.savez_compressed(
            output_dir / "step_metrics.npz",
            **{
                name: np.asarray(values)
                for name, values in result.step_metrics.items()
            },
        )

    plots: List[str] = []
    if make_plots:
        plots = plot_results(result, output_dir)
        payload["plots"] = plots
        if plots:
            with open(
                output_dir / "evaluation_summary.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
    return payload


def _cdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    _require_numpy()
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return array, array
    probabilities = np.arange(1, array.size + 1, dtype=np.float64) / array.size
    return array, probabilities


def plot_results(
    result: EvaluationResult,
    output_dir: Union[str, os.PathLike],
) -> List[str]:
    """Create CDFs and a throughput-vs-energy trade-off plot.

    Returns the generated paths, or ``[]`` when matplotlib is unavailable.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    plot_specs = (
        ("throughput", "Throughput (KPM units)", "CDF — throughput"),
        ("energy", "Energy (KPM units)", "CDF — energy"),
        ("rlf", "RLF count", "CDF — RLF"),
        ("activation_cost", "Activation cost", "CDF — activation cost"),
    )
    for metric_name, xlabel, title in plot_specs:
        values = result.step_metrics.get(metric_name, [])
        if not values:
            continue
        x_values, probabilities = _cdf(values)
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot(x_values, probabilities, linewidth=1.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("CDF")
        axis.set_title(f"{title} — {result.algo}")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        path = output_dir / f"cdf_{metric_name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated.append(str(path))

    if result.episodes:
        throughputs = [episode.throughput for episode in result.episodes]
        energies = [episode.energy for episode in result.episodes]
        figure, axis = plt.subplots(figsize=(6.0, 4.4))
        axis.scatter(energies, throughputs, s=28, alpha=0.8)
        axis.set_xlabel("Energy per episode")
        axis.set_ylabel("Throughput per episode")
        axis.set_title(f"Throughput vs energy trade-off — {result.algo}")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        path = output_dir / "tradeoff_throughput_energy.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated.append(str(path))

    return generated


# ---------------------------------------------------------------------------
# Environment factories (ns-3 with a clear offline fallback)
# ---------------------------------------------------------------------------


def _default_scenario_config() -> Optional[Path]:
    src_root = Path(__file__).resolve().parents[2]
    candidates = (
        src_root / "environments/scenario_configurations/paper_es_use_case.json",
        src_root / "environments/scenario_configurations/es_use_case.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _environment_model_kwargs(args: Any) -> Dict[str, Any]:
    """Translate the D1-D3 CLI flags into ``EnergySavingEnv`` keyword arguments."""

    power_model: Dict[str, float] = {}
    for key, value in (
        ("p0_w", getattr(args, "p0_w", None)),
        ("p_max_w", getattr(args, "p_max_w", None)),
        ("delta_p", getattr(args, "delta_p", None)),
        ("p_sleep_w", getattr(args, "p_sleep_w", None)),
        ("n_antennas", getattr(args, "n_antennas", None)),
    ):
        if value is not None:
            power_model[key] = float(value)
    return {
        "energy_model": getattr(args, "energy_model", "watts"),
        "power_model": power_model or None,
        "es_cost_mode": getattr(args, "es_cost_mode", "dense"),
        "per_cell_normalization": getattr(args, "per_cell_normalization", True),
        "per_cell_aggregation": getattr(args, "per_cell_aggregation", "mean"),
        "p_tx": float(getattr(args, "p_tx", 1.0)),
        "prb_total": float(getattr(args, "prb_total", 139.0)),
    }


def make_ns3_env(
    ns3_path: Union[str, os.PathLike],
    *,
    config_path: Optional[Union[str, os.PathLike]] = None,
    output_folder: Union[str, os.PathLike] = "output",
    optimized: bool = False,
    env_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Instantiate the live ns-3-backed ``EnergySavingEnv``.

    ``env_kwargs`` carries the D1-D3 model (energy model, activation-cost mode
    and per-cell normalisation).  Raises :class:`RuntimeError` with actionable
    instructions when the ns-3 tree, build artifacts or Frente B environment
    are unavailable.
    """

    ns3_path = Path(ns3_path)
    if not ns3_path.exists():
        raise RuntimeError(f"ns-3 path does not exist: {ns3_path}")

    if config_path is None:
        config_path = _default_scenario_config()
        if config_path is None:
            raise RuntimeError(
                "No scenario configuration was provided and no default "
                "paper_es_use_case.json / es_use_case.json was found."
            )
    config_path = Path(config_path)
    if not config_path.is_file():
        raise RuntimeError(f"Scenario configuration not found: {config_path}")

    try:
        from environments.es_env import EnergySavingEnv
    except ImportError as exc:
        raise RuntimeError(
            "Could not import environments.es_env.EnergySavingEnv. Install the "
            "nsoran package (`pip install -e .`) or run from the repository root. "
            f"Original error: {exc}"
        ) from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        scenario_configuration = json.load(handle)

    model_kwargs = dict(env_kwargs or {})
    try:
        return EnergySavingEnv(
            ns3_path=str(ns3_path),
            scenario_configuration=scenario_configuration,
            output_folder=str(output_folder),
            optimized=optimized,
            do_heuristic=False,
            **model_kwargs,
        )
    except TypeError as exc:
        if "do_heuristic" not in str(exc):
            raise RuntimeError(
                "Failed to create EnergySavingEnv. Build ns-3 first with "
                f"`cd {ns3_path} && ./ns3 build`. Original error: {exc}"
            ) from exc
        # Older/alternative Frente B signatures without ``do_heuristic``.
        try:
            return EnergySavingEnv(
                ns3_path=str(ns3_path),
                scenario_configuration=scenario_configuration,
                output_folder=str(output_folder),
                optimized=optimized,
                **model_kwargs,
            )
        except Exception as inner:
            raise RuntimeError(
                "Failed to create EnergySavingEnv. Build ns-3 first with "
                f"`cd {ns3_path} && ./ns3 build`. Original error: {inner}"
            ) from inner
    except Exception as exc:
        raise RuntimeError(
            "Failed to create EnergySavingEnv. Build ns-3 first with "
            f"`cd {ns3_path} && ./ns3 build`. Original error: {exc}"
        ) from exc


def make_dataset_env(
    dataset_path: Union[str, os.PathLike],
    *,
    mode: str = "deterministic",
    seed: int = 0,
) -> Any:
    from .offline import OfflineReplayEnv

    return OfflineReplayEnv(dataset_path, mode=mode, seed=seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_policy",
        description=(
            "Evaluate a PPO or REM-DQN policy on ns-3 or, as a fallback, on an "
            "offline .npz dataset. Aggregates throughput, energy, RLF and "
            "activation cost and optionally writes CDF/trade-off plots."
        ),
    )
    parser.add_argument(
        "--algo", required=True, choices=("ppo", "iql", "rem_dqn")
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Table I agent name (metadata only), e.g. PPO-1 or DQN.",
    )
    parser.add_argument("--model", required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--env",
        choices=("auto", "ns3", "dataset"),
        default="auto",
        help=(
            "Evaluation environment. 'auto' uses ns-3 when --ns3-path is given, "
            "otherwise falls back to --dataset."
        ),
    )
    parser.add_argument("--dataset", default=None, help="Offline .npz dataset.")
    parser.add_argument("--ns3-path", default=None, help="Path to ns-3 tree.")
    parser.add_argument(
        "--config",
        default=None,
        help="Scenario JSON (defaults to paper_es_use_case.json when available).",
    )
    parser.add_argument(
        "--output-folder",
        default="output",
        help="Output folder used by the ns-3 environment.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/evaluation",
        help="Directory for evaluation_summary.json and plots.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic/greedy inference.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000, help="Safety cap per episode."
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Do not generate matplotlib plots."
    )
    parser.add_argument(
        "--optimized", action="store_true", help="Use the optimized ns-3 build."
    )

    model_group = parser.add_argument_group("D1-D3 environment model (live ns-3)")
    model_group.add_argument(
        "--energy-model",
        choices=("watts", "proxy"),
        default="watts",
        help="Energy model used to recompute the KPI (default: watts).",
    )
    model_group.add_argument(
        "--es-cost-mode",
        choices=("dense", "impulse"),
        default="dense",
        help="Activation-cost semantics (default: dense).",
    )
    model_group.add_argument(
        "--per-cell-normalization",
        dest="per_cell_normalization",
        action="store_true",
        default=True,
        help="Normalise each reward component per cell (default: on).",
    )
    model_group.add_argument(
        "--no-per-cell-normalization",
        dest="per_cell_normalization",
        action="store_false",
        help="Ablation: normalise the pooled component only.",
    )
    model_group.add_argument(
        "--per-cell-aggregation",
        choices=("mean", "sum"),
        default="mean",
        help="How per-cell normalised components are aggregated (default: mean).",
    )
    model_group.add_argument("--p-tx", type=float, default=1.0)
    model_group.add_argument("--prb-total", type=float, default=139.0)
    model_group.add_argument("--p0-w", type=float, default=None)
    model_group.add_argument("--p-max-w", type=float, default=None)
    model_group.add_argument("--delta-p", type=float, default=None)
    model_group.add_argument("--p-sleep-w", type=float, default=None)
    model_group.add_argument("--n-antennas", type=float, default=None)
    return parser


def _resolve_environment(args: Any) -> Tuple[Any, str, List[str]]:
    notes: List[str] = []

    if args.env in ("auto", "ns3") and args.ns3_path:
        try:
            model_kwargs = _environment_model_kwargs(args)
            env = make_ns3_env(
                args.ns3_path,
                config_path=args.config,
                output_folder=args.output_folder,
                optimized=args.optimized,
                env_kwargs=model_kwargs,
            )
            notes.append(f"ns-3 metrics use the D1-D3 model {model_kwargs}.")
            return env, "ns3:EnergySavingEnv", notes
        except Exception as exc:
            if args.env == "ns3":
                raise
            notes.append(f"ns-3 environment unavailable, falling back: {exc}")
            print(
                "Warning: ns-3/Gym environment unavailable; falling back to "
                f"the offline dataset. Reason: {exc}"
            )

    if args.env in ("auto", "dataset") and args.dataset:
        env = make_dataset_env(args.dataset, mode="deterministic", seed=args.seed)
        notes.append(
            "Offline replay fallback: actions are recorded but do not affect the "
            "replayed transitions; metrics describe the dataset, not the policy."
        )
        return env, "offline:OfflineReplayEnv", notes

    if args.env == "ns3" and not args.ns3_path:
        raise RuntimeError("--env ns3 requires --ns3-path.")
    if args.env == "dataset" and not args.dataset:
        raise RuntimeError("--env dataset requires --dataset path.npz.")

    raise RuntimeError(
        "No evaluation environment available. Provide --dataset path.npz to "
        "replay an offline dataset, or --ns3-path /path/to/ns-3-mmwave-oran "
        "with a built ns-3 tree."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import sys

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.agent:
        try:
            get_agent_config(args.agent)
        except KeyError as exc:
            parser.error(str(exc))

    try:
        env, environment_label, notes = _resolve_environment(args)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = evaluate_policy(
            args.algo,
            args.model,
            env,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=not args.stochastic,
            max_steps_per_episode=args.max_steps,
            environment_label=environment_label,
            notes=notes,
        )
        payload = save_result(
            result, args.output_dir, make_plots=not args.no_plots
        )
    except ImportError as exc:
        print(f"error: missing optional dependency: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = payload["summary"]
    print(f"Algorithm: {payload['algo']}")
    print(f"Environment: {payload['environment']}")
    print(f"Episodes: {summary['episodes']}")
    for name in ("throughput", "energy", "rlf", "activation_cost", "bs_on_mean"):
        stats = summary["metrics"][name]
        print(f"{name}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    if summary["throughput_per_energy"] is not None:
        print(f"throughput/energy: {summary['throughput_per_energy']:.4f}")
    if payload.get("plots"):
        print("Plots:")
        for path in payload["plots"]:
            print(f"  {path}")
    print(f"Saved evaluation artifacts to: {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
