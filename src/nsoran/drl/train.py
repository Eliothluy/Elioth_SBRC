"""CLI orchestration for offline DRL training (PPO and REM-DQN + CQL).

Examples
--------
::

    python examples/train_offline.py --algo ppo --agent PPO-1 \
        --dataset data/offline.npz --output-dir runs/ppo1
    python examples/train_offline.py --algo rem_dqn --agent DQN \
        --dataset data/offline.npz --output-dir runs/dqn \
        --total-timesteps 50000

The dataset format is documented in :mod:`nsoran.drl.offline`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .config import (
    AGENT_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EVAL_FREQ,
    DEFAULT_MIN_TIMESTEPS,
    DEFAULT_TOTAL_TIMESTEPS,
    get_agent_config,
)

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_offline",
        description=(
            "Train PPO (Stable Baselines3) or REM-DQN + CQL (PyTorch) offline "
            "on an .npz dataset collected from ns-3 / ns-O-RAN."
        ),
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=("ppo", "rem_dqn"),
        help="Learning algorithm.",
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=AGENT_NAMES,
        help="Table I agent name (defines reward weights/normalization).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Offline dataset .npz (see nsoran.drl.offline for the schema).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/<algo>_<agent>_<timestamp>).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help=(
            "PPO environment timesteps (default 1,200,000) or REM-DQN gradient "
            "updates (same default; use a smaller value for smoke tests)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help=(
            "Override PPO entropy coefficient; must be in [0.001, 0.003]. "
            "Defaults to the per-agent Table I value."
        ),
    )
    parser.add_argument(
        "--offline-algo",
        choices=("auto", "replay", "iql"),
        default="auto",
        help=(
            "Offline learner for PPO-* agents: 'iql' trains the approved "
            "discrete IQL agent (D4, the default), 'replay' keeps the legacy "
            "OfflineReplayEnv + SB3 PPO pipeline as an ablation. DQN always "
            "uses REM-DQN + CQL."
        ),
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluation period in timesteps (default 10,000).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="PPO validation episodes per evaluation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop after N evaluations without improvement (default 4).",
    )
    parser.add_argument(
        "--min-timesteps",
        type=int,
        default=DEFAULT_MIN_TIMESTEPS,
        help="Do not early-stop before this many timesteps (default 10,000).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of episodes held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch/SB3 device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--tensorboard-log",
        default=None,
        help="Optional SB3 TensorBoard log directory.",
    )
    parser.add_argument(
        "--no-apply-reward-config",
        dest="apply_reward_config",
        action="store_false",
        help=(
            "Use the stored dataset rewards instead of recomputing them from "
            "the optional reward_components array with the Table I weights."
        ),
    )
    parser.set_defaults(apply_reward_config=True)
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=0.5,
        help="REM-DQN CQL conservative coefficient (default 0.5).",
    )
    parser.add_argument(
        "--k-heads",
        type=int,
        default=3,
        help="Number of REM ensemble heads (default 3).",
    )
    parser.add_argument(
        "--hidden-dims",
        default=None,
        help="Comma-separated dense sizes (default: 512,512,256).",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="REM-DQN learning rate."
    )
    parser.add_argument(
        "--iql-tau",
        type=float,
        default=0.7,
        help="IQL expectile for the value function (default 0.7).",
    )
    parser.add_argument(
        "--iql-beta",
        type=float,
        default=1.0,
        help="IQL advantage-weighting temperature (default 1.0).",
    )
    parser.add_argument(
        "--iql-lr",
        type=float,
        default=3e-4,
        help="IQL learning rate (default 3e-4).",
    )
    parser.add_argument(
        "--iql-max-weight",
        type=float,
        default=100.0,
        help="IQL AWR weight clip (default 100).",
    )
    parser.add_argument(
        "--iql-target-update",
        type=float,
        default=0.005,
        help="IQL soft target update rate (default 0.005).",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=1000,
        help="Hard target-network update period (gradient steps).",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=100_000
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1000,
        help="REM-DQN console log period (gradient steps).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Optional REM-DQN periodic checkpoint period.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the SB3 PPO progress bar.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress training logs."
    )
    # --- E1 / CMDP (docs/plano_tecnico/cmdp_design.md §6.3) -----------
    # Default --cmdp-mode none: a Eq. (2) do paper é usada intacta.
    parser.add_argument(
        "--cmdp-mode",
        choices=("none", "lagrangian", "barrier"),
        default="none",
        help=(
            "Modo de restrição QoS do E1: 'lagrangian' penaliza o alvo TD "
            "com lambda * custo hinge (REM-DQN e IQL, design §4.1); "
            "'barrier' (porta/gate da Eq. 5.1') ainda não está ligado aos "
            "treinadores offline (exige re-scoring do dataset, §6.2). "
            "Default 'none' mantém a Eq. (2)."
        ),
    )
    parser.add_argument(
        "--qos-throughput-min",
        type=float,
        default=None,
        help=(
            "τ_SLA: throughput mínimo médio por indicação, em bytes crus "
            "(design §2.1 recomenda 53680 = 70%% de μ_AO)."
        ),
    )
    parser.add_argument(
        "--qos-rlf-max",
        type=float,
        default=None,
        help="ε: máximo de UEs em RLF por indicação (design §2.1: p95 do AO = 14).",
    )
    parser.add_argument(
        "--qos-throughput-scale",
        type=float,
        default=None,
        help="Escala S_ρ do hinge (default: max(τ_SLA, 1); design §2.1: μ_AO = 76686).",
    )
    parser.add_argument(
        "--qos-rlf-scale",
        type=float,
        default=None,
        help="Escala S_ς do hinge (default: max(ε, 1); design §2.1: 18).",
    )
    parser.add_argument(
        "--qos-lambda-lr",
        type=float,
        default=0.01,
        help="η_λ do passo dual (design §4.1: 0.01).",
    )
    parser.add_argument(
        "--qos-lambda-init",
        type=float,
        default=0.0,
        help="λ inicial (design §4.1: 0).",
    )
    parser.add_argument(
        "--qos-lambda-max",
        type=float,
        default=10.0,
        help="Teto de λ (design §8.1-A4 reduz o default de 100 para 10).",
    )
    parser.add_argument(
        "--qos-tolerance",
        type=float,
        default=0.05,
        help="δ: custo médio admissível (design §2.1/§4.1: 0.05).",
    )
    return parser


def _parse_hidden_dims(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            f"--hidden-dims must be comma-separated integers, got {raw!r}."
        ) from exc
    if not values:
        raise ValueError("--hidden-dims must contain at least one size.")
    return values


def _augment_and_write_summary(
    output_dir: Path, paper_agent: Any, summary: Dict[str, Any]
) -> None:
    summary["paper_agent"] = paper_agent.as_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "paper_agent.json", "w", encoding="utf-8") as handle:
        json.dump(paper_agent.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    from .offline import load_offline_dataset

    paper_agent = get_agent_config(args.agent)
    if paper_agent.algo != args.algo:
        raise ValueError(
            f"Agent {paper_agent.name} is configured for {paper_agent.algo!r}, "
            f"but --algo {args.algo!r} was requested. Use --agent DQN with "
            "--algo rem_dqn, or a PPO-* agent with --algo ppo."
        )
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1).")

    # --- E1/CMDP (docs/plano_tecnico/cmdp_design.md §6.3) --------------
    # Validação antes de abrir o dataset / criar pastas, para falhar rápido.
    # Default --cmdp-mode none: qos_kwargs vazio, comportamento intacto.
    cmdp_mode = getattr(args, "cmdp_mode", "none")
    qos_kwargs: Dict[str, Any] = {}
    if cmdp_mode == "barrier":
        raise ValueError(
            "--cmdp-mode barrier (porta/gate da Eq. 5.1') ainda não está "
            "ligado aos treinadores offline: exige o re-scoring do dataset "
            "com raw_components (design §6.2). Use --cmdp-mode lagrangian."
        )
    if cmdp_mode == "lagrangian":
        if args.qos_throughput_min is None and args.qos_rlf_max is None:
            raise ValueError(
                "--cmdp-mode lagrangian exige --qos-throughput-min e/ou "
                "--qos-rlf-max (τ_SLA / ε em unidades cruas do ns-3)."
            )
        if args.algo == "ppo" and getattr(args, "offline_algo", "auto") == "replay":
            raise ValueError(
                "--cmdp-mode lagrangian com --offline-algo replay (SB3 PPO) "
                "ainda não está implementado (wrapper §8.1-A5 do design); "
                "use o IQL discreto (default) ou --algo rem_dqn."
            )
        qos_kwargs = {
            "qos_throughput_min": args.qos_throughput_min,
            "qos_rlf_max": args.qos_rlf_max,
            "qos_throughput_scale": args.qos_throughput_scale,
            "qos_rlf_scale": args.qos_rlf_scale,
            "qos_lambda_lr": args.qos_lambda_lr,
            "qos_lambda_init": args.qos_lambda_init,
            "qos_lambda_max": args.qos_lambda_max,
            "qos_tolerance": args.qos_tolerance,
        }

    offline_algo = getattr(args, "offline_algo", "auto")
    if offline_algo == "auto":
        # D4: PPO-* agents use discrete IQL by default; the replay pipeline is
        # kept as an explicit ablation.
        offline_algo = "iql"
    if args.algo != "ppo" and offline_algo == "iql":
        raise ValueError(
            "--offline-algo iql is only available for PPO-* agents; "
            "the DQN agent uses REM-DQN + CQL."
        )

    run_tag = f"{args.algo}_{paper_agent.name}"
    if offline_algo == "iql":
        run_tag += "_iql"
    output_dir = Path(
        args.output_dir
        or (f"runs/{run_tag}_{time.strftime('%Y%m%d-%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_offline_dataset(args.dataset)
    print(
        f"Dataset: {dataset.num_transitions} transitions, "
        f"{dataset.num_episodes} episodes"
    )

    if args.apply_reward_config:
        if dataset.reward_components is not None:
            dataset = dataset.with_rewards_from_components(paper_agent)
            print(
                f"Rewards recomputed with {paper_agent.name} weights "
                f"{paper_agent.weights} ({paper_agent.normalization})."
            )
        else:
            print(
                "Note: dataset has no 'reward_components'; using stored rewards. "
                "Use --no-apply-reward-config to silence this note."
            )

    verbose = 0 if args.quiet else 1

    if args.algo == "ppo" and offline_algo == "iql":
        try:
            from .iql import IQLConfig, train_iql
        except ImportError as exc:
            raise ImportError(
                "IQL training requires torch and numpy. Install the DRL "
                "extras with `pip install -e '.[drl]'`. "
                f"Original error: {exc}"
            ) from exc
        hidden_dims = _parse_hidden_dims(args.hidden_dims)
        iql_config = IQLConfig(
            hidden_dims=hidden_dims or (256, 256),
            gamma=args.gamma,
            lr=args.iql_lr,
            tau=args.iql_tau,
            beta=args.iql_beta,
            max_weight=args.iql_max_weight,
            target_update=args.iql_target_update,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            **qos_kwargs,
        )
        summary = train_iql(
            dataset,
            agent_config=paper_agent,
            iql_config=iql_config,
            output_dir=output_dir,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            patience=args.patience,
            min_timesteps=args.min_timesteps,
            val_fraction=args.val_fraction,
            seed=args.seed,
            log_freq=args.log_freq,
            checkpoint_freq=args.checkpoint_freq,
            verbose=verbose,
        )
    elif args.algo == "ppo":
        try:
            from .ppo import train_ppo
        except ImportError as exc:
            raise ImportError(
                "PPO training requires Stable Baselines3 and torch. Install the "
                "DRL extras with `pip install -e '.[drl]'`. "
                f"Original error: {exc}"
            ) from exc
        summary = train_ppo(
            dataset,
            agent_config=paper_agent,
            output_dir=output_dir,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            patience=args.patience,
            min_timesteps=args.min_timesteps,
            val_fraction=args.val_fraction,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            device=args.device,
            verbose=verbose,
            progress_bar=args.progress_bar,
        )
    else:
        try:
            from .rem_dqn import REMDQNConfig, train_rem_dqn
        except ImportError as exc:
            raise ImportError(
                "REM-DQN training requires torch and numpy. Install the DRL "
                "extras with `pip install -e '.[drl]'`. "
                f"Original error: {exc}"
            ) from exc
        hidden_dims = _parse_hidden_dims(args.hidden_dims)
        agent_config = REMDQNConfig(
            k_heads=args.k_heads,
            hidden_dims=hidden_dims or (512, 512, 256),
            cql_alpha=args.cql_alpha,
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            device=args.device,
            seed=args.seed,
            **qos_kwargs,
        )
        summary = train_rem_dqn(
            dataset,
            agent_config=agent_config,
            output_dir=output_dir,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            patience=args.patience,
            min_timesteps=args.min_timesteps,
            val_fraction=args.val_fraction,
            seed=args.seed,
            log_freq=args.log_freq,
            checkpoint_freq=args.checkpoint_freq,
            verbose=verbose,
        )

    summary["offline_algo"] = offline_algo
    _augment_and_write_summary(output_dir, paper_agent, summary)
    summary["output_dir"] = str(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = _run(args)
    except ImportError as exc:
        print(f"error: missing optional dependency: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = summary.get("output_dir", args.output_dir)
    print("Training finished.")
    print(f"  completed steps/timesteps: {summary.get('completed_steps', summary.get('completed_timesteps'))}")
    print(f"  early stopped: {summary.get('early_stopped')}")
    print(f"  best model: {summary.get('best_model_path')}")
    print(f"  final model: {summary.get('final_model_path')}")
    print(f"  output dir: {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
