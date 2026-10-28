"""Shared CLI runner for the per-agent DRL example scripts (Table I).

``examples/ppo_1.py`` ... ``examples/ppo_5.py`` and ``examples/dqn.py`` are
thin wrappers around :func:`run_agent`, one per agent of Table I of the paper.
Each wrapper can:

* train the agent offline on a ``.npz`` dataset (or a synthetic one for smoke
  tests) using the agent's Table I reward configuration, mirroring
  ``examples/train_offline.py``;
* evaluate the resulting checkpoint -- or an existing one with
  ``--eval-only`` -- on the live ns-3 paper scenario or on an offline dataset
  replay, mirroring ``examples/evaluate_policy.py``.

Only the standard library is imported at module load time, so ``--help`` works
before the optional DRL dependencies (numpy, gymnasium, torch, Stable
Baselines3) are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # make ``nsoran`` importable without installing
    sys.path.insert(0, str(_SRC))

from nsoran.drl.config import (  # noqa: E402  (path setup must come first)
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOTAL_TIMESTEPS,
    get_agent_config,
)


def build_parser(agent: Any) -> argparse.ArgumentParser:
    """Build the CLI for one Table I agent."""

    program = f"{agent.name.lower().replace('-', '_')}.py"
    parser = argparse.ArgumentParser(
        prog=program,
        description=(
            f"Train and/or evaluate the {agent.name!r} agent of Table I "
            f"({agent.algo!r}). {agent.description}"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--dataset",
        default=None,
        help="Offline .npz dataset collected from ns-3 (see nsoran.drl.offline).",
    )
    data_group.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="Generate a synthetic dataset with N transitions (smoke tests only).",
    )
    data_group.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory (default: runs/<AGENT>).",
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="PPO environment timesteps or REM-DQN gradient updates.",
    )
    train_group.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument(
        "--device",
        default="auto",
        help="Torch/SB3 device: auto, cpu, cuda, ...",
    )
    train_group.add_argument(
        "--quiet", action="store_true", help="Suppress training logs."
    )
    train_group.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the Stable Baselines3 PPO progress bar.",
    )
    train_group.add_argument(
        "--no-apply-reward-config",
        dest="apply_reward_config",
        action="store_false",
        help=(
            "Use the stored dataset rewards instead of recomputing them from "
            "the reward_components array with the Table I weights."
        ),
    )
    if agent.algo == "ppo":
        train_group.add_argument(
            "--offline-algo",
            choices=("auto", "iql", "replay"),
            default="auto",
            help=(
                "Offline learner: 'iql' (default, D4) trains the approved "
                "discrete IQL agent; 'replay' keeps the legacy "
                "OfflineReplayEnv + SB3 PPO pipeline as an ablation."
            ),
        )
        train_group.add_argument(
            "--ent-coef",
            type=float,
            default=None,
            help=(
                "Override the PPO entropy coefficient (paper range "
                "[0.001, 0.003]); defaults to the per-agent Table I value."
            ),
        )
        train_group.add_argument(
            "--iql-tau",
            type=float,
            default=0.7,
            help="IQL expectile of the value loss (default 0.7).",
        )
        train_group.add_argument(
            "--iql-beta",
            type=float,
            default=1.0,
            help="IQL AWR temperature (default 1.0).",
        )
        train_group.add_argument(
            "--iql-lr",
            type=float,
            default=3e-4,
            help="IQL learning rate (default 3e-4).",
        )
        train_group.add_argument(
            "--iql-max-weight",
            type=float,
            default=100.0,
            help="IQL AWR weight clip (default 100).",
        )
        train_group.add_argument(
            "--iql-target-update",
            type=float,
            default=0.005,
            help="IQL soft target-value update rate (default 0.005).",
        )
        train_group.add_argument(
            "--iql-gamma",
            type=float,
            default=0.99,
            help="IQL discount factor (default 0.99).",
        )
    else:
        train_group.add_argument(
            "--cql-alpha",
            type=float,
            default=0.5,
            help="CQL conservative coefficient.",
        )
        train_group.add_argument(
            "--k-heads", type=int, default=3, help="Number of REM ensemble heads."
        )
        train_group.add_argument("--gamma", type=float, default=0.99)
        train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
        train_group.add_argument(
            "--target-update-freq",
            type=int,
            default=1000,
            help="Hard target-network update period (gradient steps).",
        )
        train_group.add_argument(
            "--checkpoint-freq",
            type=int,
            default=None,
            help="Optional periodic checkpoint period (gradient steps).",
        )

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the trained checkpoint after training.",
    )
    eval_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate --model instead.",
    )
    eval_group.add_argument(
        "--model",
        default=None,
        help=(
            "Checkpoint to evaluate (best_model.zip / best_model.pt). Defaults "
            "to the best checkpoint produced by the training run."
        ),
    )
    eval_group.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes."
    )
    eval_group.add_argument(
        "--ns3-path",
        default=None,
        help="Path to the ns-3 mmWave O-RAN tree for live evaluation.",
    )
    eval_group.add_argument(
        "--ns3-output-folder",
        default=None,
        help=(
            "Folder that receives the run folders of the live ns-3 "
            "evaluation (default: output)."
        ),
    )
    eval_group.add_argument(
        "--scenario-config",
        default=None,
        help="Scenario JSON for the live ns-3 environment.",
    )
    eval_group.add_argument(
        "--env",
        choices=("auto", "ns3", "dataset"),
        default="auto",
        help=(
            "Evaluation environment ('auto' prefers ns-3 when --ns3-path is "
            "given)."
        ),
    )
    eval_group.add_argument(
        "--eval-output-dir",
        default=None,
        help="Evaluation output directory (default: <output-dir>/evaluation).",
    )
    eval_group.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of deterministic/greedy inference.",
    )
    eval_group.add_argument(
        "--no-plots", action="store_true", help="Do not generate CDF/trade-off plots."
    )

    model_group = parser.add_argument_group("D1-D3 environment model (live ns-3)")
    model_group.add_argument(
        "--energy-model", choices=("watts", "proxy"), default="watts"
    )
    model_group.add_argument(
        "--es-cost-mode", choices=("dense", "impulse"), default="dense"
    )
    model_group.add_argument(
        "--per-cell-normalization",
        dest="per_cell_normalization",
        action="store_true",
        default=True,
    )
    model_group.add_argument(
        "--no-per-cell-normalization",
        dest="per_cell_normalization",
        action="store_false",
    )
    model_group.add_argument(
        "--per-cell-aggregation", choices=("mean", "sum"), default="mean"
    )
    model_group.add_argument("--p-tx", type=float, default=1.0)
    model_group.add_argument("--prb-total", type=float, default=139.0)
    model_group.add_argument("--p0-w", type=float, default=None)
    model_group.add_argument("--p-max-w", type=float, default=None)
    model_group.add_argument("--delta-p", type=float, default=None)
    model_group.add_argument("--p-sleep-w", type=float, default=None)
    model_group.add_argument("--n-antennas", type=float, default=None)

    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast with actionable messages before any heavy import."""

    if args.eval_only and not args.model:
        parser.error("--eval-only requires --model (the checkpoint to evaluate).")
    if not args.eval_only and not args.dataset and not args.synthetic:
        parser.error("training requires --dataset PATH or --synthetic N.")
    if args.dataset and args.synthetic:
        parser.error("use either --dataset or --synthetic, not both.")
    if args.synthetic is not None and args.synthetic <= 0:
        parser.error("--synthetic must be a positive number of transitions.")
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if args.eval_only and not args.ns3_path and not args.dataset and not args.synthetic:
        parser.error(
            "evaluation requires --ns3-path (live ns-3) or --dataset / "
            "--synthetic (offline replay)."
        )


def _make_synthetic(agent: Any, args: argparse.Namespace, output_dir: Path) -> Path:
    from nsoran.drl.offline import save_offline_dataset, synthetic_dataset

    dataset = synthetic_dataset(num_transitions=args.synthetic, seed=args.seed)
    path = output_dir / "synthetic_dataset.npz"
    save_offline_dataset(path, dataset)
    print(
        f"[{agent.name}] synthetic dataset saved to {path} "
        f"({args.synthetic} transitions)."
    )
    return path


def _training_argv(
    agent: Any,
    args: argparse.Namespace,
    dataset_path: Path,
    output_dir: Path,
) -> List[str]:
    """Translate the agent CLI into the ``nsoran.drl.train`` CLI."""

    argv = [
        "--algo",
        agent.algo,
        "--agent",
        agent.name,
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--total-timesteps",
        str(args.total_timesteps),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.quiet:
        argv.append("--quiet")
    if args.progress_bar:
        argv.append("--progress-bar")
    if not args.apply_reward_config:
        argv.append("--no-apply-reward-config")
    if agent.algo == "ppo":
        argv += ["--offline-algo", args.offline_algo]
        if args.offline_algo in ("auto", "iql"):
            argv += [
                "--iql-tau",
                str(args.iql_tau),
                "--iql-beta",
                str(args.iql_beta),
                "--iql-lr",
                str(args.iql_lr),
                "--iql-max-weight",
                str(args.iql_max_weight),
                "--iql-target-update",
                str(args.iql_target_update),
                "--gamma",
                str(args.iql_gamma),
            ]
        if args.ent_coef is not None:
            argv += ["--ent-coef", str(args.ent_coef)]
    else:
        argv += [
            "--cql-alpha",
            str(args.cql_alpha),
            "--k-heads",
            str(args.k_heads),
            "--gamma",
            str(args.gamma),
            "--lr",
            str(args.lr),
            "--target-update-freq",
            str(args.target_update_freq),
        ]
        if args.checkpoint_freq is not None:
            argv += ["--checkpoint-freq", str(args.checkpoint_freq)]
    return argv


def _train(
    agent: Any,
    args: argparse.Namespace,
    dataset_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    from nsoran.drl import train as train_cli

    print(
        f"[{agent.name}] training with weights {agent.weights}, "
        f"normalization '{agent.normalization}'."
    )
    namespace = train_cli.build_parser().parse_args(
        _training_argv(agent, args, dataset_path, output_dir)
    )
    return train_cli._run(namespace)


def _evaluation_argv(
    agent: Any,
    args: argparse.Namespace,
    model_path: Any,
    dataset_path: Optional[Path],
    output_dir: Path,
) -> List[str]:
    """Translate the agent CLI into the ``nsoran.drl.evaluate`` CLI."""

    eval_output_dir = (
        Path(args.eval_output_dir)
        if args.eval_output_dir
        else output_dir / "evaluation"
    )
    eval_algo = agent.algo
    if agent.algo == "ppo":
        if args.offline_algo in ("auto", "iql"):
            # Auto-detect the legacy Stable-Baselines3 .zip checkpoint; the
            # IQL/REM-DQN checkpoints are .pt files.
            eval_algo = "ppo" if str(model_path).endswith(".zip") else "iql"
        else:
            eval_algo = "ppo"

    argv = [
        "--algo",
        eval_algo,
        "--agent",
        agent.name,
        "--model",
        str(model_path),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(eval_output_dir),
    ]
    if args.env != "auto":
        argv += ["--env", args.env]
    if args.ns3_path:
        argv += ["--ns3-path", str(args.ns3_path)]
    if args.ns3_output_folder:
        argv += ["--output-folder", str(args.ns3_output_folder)]
    if dataset_path is not None:
        argv += ["--dataset", str(dataset_path)]
    if args.scenario_config:
        argv += ["--config", str(args.scenario_config)]
    if args.stochastic:
        argv.append("--stochastic")
    if args.no_plots:
        argv.append("--no-plots")

    # D1-D3 model used by the live ns-3 evaluation (ignored by the offline
    # replay, whose rewards are stored in the dataset).
    argv += [
        "--energy-model",
        args.energy_model,
        "--es-cost-mode",
        args.es_cost_mode,
        "--per-cell-aggregation",
        args.per_cell_aggregation,
        "--p-tx",
        str(args.p_tx),
        "--prb-total",
        str(args.prb_total),
    ]
    argv.append(
        "--per-cell-normalization"
        if args.per_cell_normalization
        else "--no-per-cell-normalization"
    )
    for flag, value in (
        ("--p0-w", args.p0_w),
        ("--p-max-w", args.p_max_w),
        ("--delta-p", args.delta_p),
        ("--p-sleep-w", args.p_sleep_w),
        ("--n-antennas", args.n_antennas),
    ):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def _evaluate(
    agent: Any,
    args: argparse.Namespace,
    model_path: Any,
    dataset_path: Optional[Path],
    output_dir: Path,
) -> int:
    from nsoran.drl import evaluate as evaluate_cli

    if model_path is None:
        raise ValueError(
            "No checkpoint to evaluate. Train first or pass --model with --eval-only."
        )
    print(f"[{agent.name}] evaluating {model_path}.")
    argv = _evaluation_argv(agent, args, str(model_path), dataset_path, output_dir)
    return evaluate_cli.main(argv)


def _print_training_summary(
    agent: Any, output_dir: Path, summary: Dict[str, Any]
) -> None:
    print("Training finished.")
    print(f"  agent: {agent.name}")
    print(f"  output dir: {output_dir}")
    print(f"  best model: {summary.get('best_model_path')}")
    print(f"  final model: {summary.get('final_model_path')}")
    print(
        "  hint: re-run with --evaluate (and --ns3-path ... on the paper "
        "scenario) to evaluate the policy."
    )


def _run_agent(agent_name: str, argv: Optional[Sequence[str]] = None) -> int:
    agent = get_agent_config(agent_name)
    parser = build_parser(agent)
    args = parser.parse_args(argv)
    _validate(parser, args)

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("runs") / agent.name
    )
    dataset_path = Path(args.dataset) if args.dataset else None
    model_path = args.model

    if not args.eval_only:
        if args.synthetic is not None:
            dataset_path = _make_synthetic(agent, args, output_dir)
        assert dataset_path is not None
        summary = _train(agent, args, dataset_path, output_dir)
        if model_path is None:
            model_path = summary.get("best_model_path") or summary.get(
                "final_model_path"
            )
        if not args.evaluate:
            _print_training_summary(agent, output_dir, summary)
            return 0

    return _evaluate(agent, args, model_path, dataset_path, output_dir)


def run_agent(agent_name: str, argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by the per-agent scripts; returns a process exit code.

    ``argv`` defaults to ``sys.argv[1:]`` (argparse behaviour).  ``--help``
    and early validation errors exit through ``SystemExit`` as usual.
    """

    try:
        return _run_agent(agent_name, argv)
    except ImportError as exc:
        print(f"error: missing optional dependency: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_agent(sys.argv[1] if len(sys.argv) > 1 else "PPO-1"))
