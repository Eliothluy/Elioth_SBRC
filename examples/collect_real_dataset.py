#!/usr/bin/env python3
"""Convert one or more real ns-O-RAN (ns-3) runs into an offline ``.npz``.

This is the "Frente D" entry point of the paper pipeline: the runs produced by
``examples/energy_saving.py`` (a folder with ``database.db``, ``bsState.txt``
and ``EsActions.txt``) are replayed through the *same* code path as the live
environment and stored in the dataset format described in
:mod:`nsoran.drl.offline`.

Usage::

    # one run
    python examples/collect_real_dataset.py \
        --run-dir examples/output/<uuid> \
        --output data/paper_run_000.npz \
        --config src/environments/scenario_configurations/paper_es_use_case.json

    # every run under a folder, concatenated into a single dataset
    python examples/collect_real_dataset.py \
        --runs-dir examples/output/real_runs \
        --output data/paper_300k.npz \
        --config src/environments/scenario_configurations/paper_es_use_case.json

The rewards are recomputed with the Table I weights of the selected agent
(``--agent PPO-2`` by default); the raw component values are kept in the
``reward_components`` array so any other Table I row can be applied later with
``--apply-reward-config`` during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # make ``nsoran`` importable without installing
    sys.path.insert(0, str(_SRC))

from _model_cli import (  # noqa: E402
    add_environment_model_arguments,
    model_from_args,
)
from nsoran.drl.config import get_agent_config  # noqa: E402
from nsoran.drl.offline import save_offline_dataset  # noqa: E402
from nsoran.drl.real_data import (  # noqa: E402
    dataset_from_ns3_run,
    dataset_from_ns3_runs,
    load_scenario_config,
)


def _find_run_dirs(root: Path) -> list[Path]:
    """Return every run folder below ``root``.

    A run folder is a directory that holds a ``database.db``.  The search is
    recursive so that a campaign folder such as
    ``examples/output/real_runs/<policy>_<seed>/<uuid>/`` is fully covered.
    """

    if (root / "database.db").is_file():
        return [root]
    return sorted(
        child.parent
        for child in root.rglob("database.db")
        if child.parent.is_dir()
    )


def _has_indications(run_dir: Path) -> bool:
    """Return True when the datalake of ``run_dir`` holds at least one E2 indication."""

    import sqlite3

    try:
        connection = sqlite3.connect(f"file:{run_dir / 'database.db'}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        for table in ("grafana", "du"):
            try:
                count = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            except sqlite3.Error:
                continue
            if count:
                return True
        return False
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_real_dataset.py",
        description=(
            "Convert real ns-3 / ns-O-RAN run folders into the offline .npz "
            "dataset used by the PPO and REM-DQN agents."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="A ns-O-RAN run folder (repeatable).",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Folder whose sub-folders are ns-O-RAN runs (all of them are used).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Campaign manifest.json (e.g. examples/output/real_runs/manifest.json). "
            "When given, only jobs with status='ok' are converted and their "
            "policy labels are stored in the dataset metadata."
        ),
    )
    parser.add_argument(
        "--state-variant",
        choices=("full", "paper85", "compact", "compact78"),
        default="full",
        help=(
            "State layout: 'full'/'paper85' is the paper's 85-dim state "
            "(default); 'compact' is the 70-dim E5 vector of "
            "docs/plano_tecnico/kpm_audit.md (drops mac_pdu_64qam and "
            "prb_percentage per cell plus the global bs_on); 'compact78' "
            "is the legacy 78-dim ablation."
        ),
    )
    parser.add_argument(
        "--output",
        default="offline_dataset.npz",
        help="Output .npz path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Scenario JSON of the runs (e.g. paper_es_use_case.json). It "
            "supplies 'ues' and the E2 indication periodicity."
        ),
    )
    add_environment_model_arguments(parser)
    parser.add_argument(
        "--agent",
        default="PPO-2",
        help="Table I agent whose reward weights are used (default: PPO-2).",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help=(
            "Reward calibration .npz (see examples/fit_reward_calibration.py). "
            "Without it the generic fallback ranges of the reward normaliser "
            "are used, which compresses the real KPI values."
        ),
    )
    parser.add_argument(
        "--write-calibration",
        default=None,
        metavar="PATH",
        help=(
            "Also fit the reward calibration on the given runs and write it to "
            "PATH (use it in a second pass to convert the dataset with it)."
        ),
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip the dataset contract validation.",
    )
    parser.set_defaults(validate=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labelled_runs: list[tuple[Path, str]] = []

    def add_runs(paths: list[Path], label: str) -> None:
        for path in paths:
            if not any(existing == path for existing, _ in labelled_runs):
                labelled_runs.append((path, label))

    for item in args.run_dir:
        path = Path(item)
        if not path.is_dir():
            print(f"error: --run-dir is not a directory: {path}", file=sys.stderr)
            return 2
        add_runs(_find_run_dirs(path), "unknown")
    if args.runs_dir:
        root = Path(args.runs_dir)
        if not root.is_dir():
            print(f"error: --runs-dir does not exist: {root}", file=sys.stderr)
            return 2
        add_runs(_find_run_dirs(root), "unknown")
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"error: --manifest does not exist: {manifest_path}", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text())
        for job in manifest.get("jobs", []):
            if str(job.get("status", "")).lower() != "ok":
                continue
            policy = str(job.get("policy", "unknown"))
            for raw_path in job.get("run_dirs") or []:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    for base in (_REPO_ROOT, _REPO_ROOT.parent):
                        resolved = base / candidate
                        if (resolved / "database.db").is_file():
                            candidate = resolved
                            break
                if (candidate / "database.db").is_file():
                    add_runs([candidate], policy)

    run_dirs = [path for path, _ in labelled_runs]
    run_labels = [label for _, label in labelled_runs]

    if not run_dirs:
        print(
            "error: no run folders given (use --run-dir and/or --runs-dir)",
            file=sys.stderr,
        )
        return 2

    missing = [item for item in run_dirs if not (item / "database.db").is_file()]
    if missing:
        for item in missing:
            print(f"error: no database.db in {item}", file=sys.stderr)
        return 2

    # A run that is still in flight (or that crashed before the first E2
    # indication) holds no indication at all: skip it instead of failing.
    empty = [item for item in run_dirs if not _has_indications(item)]
    for item in empty:
        print(f"warning: skipping {item} (no E2 indication yet)")
    kept = [
        (path, label)
        for path, label in zip(run_dirs, run_labels)
        if path not in empty
    ]
    run_dirs = [path for path, _ in kept]
    run_labels = [label for _, label in kept]
    if not run_dirs:
        print("error: none of the given runs holds an E2 indication", file=sys.stderr)
        return 2

    agent = get_agent_config(args.agent)
    config = load_scenario_config(args.config) if args.config else None
    if config is None:
        config = {"simTime": [10], "ues": [9], "indicationPeriodicity": [0.1]}
    model = model_from_args(args)

    if args.write_calibration:
        from nsoran.drl.real_data import raw_reward_components_from_ns3_runs

        samples = raw_reward_components_from_ns3_runs(
            run_dirs,
            scenario_configuration=config,
            model=model,
            include_cells=args.per_cell_normalization,
        )
        output = Path(args.write_calibration)
        output.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            output, **{k: v for k, v in samples.items() if v.size}
        )
        print(f"Reward calibration written to {output}")

    print(
        f"Converting {len(run_dirs)} run(s) with {agent.name} weights "
        f"{agent.weights}, model {model.summary()}, state={args.state_variant}"
    )

    if len(run_dirs) == 1:
        dataset = dataset_from_ns3_run(
            run_dirs[0],
            scenario_configuration=config,
            reward_weights=agent.weights,
            normalization=agent.normalization,
            calibration_file=args.calibration_file,
            validate=args.validate,
            model=model,
            state_variant=args.state_variant,
        )
    else:
        dataset = dataset_from_ns3_runs(
            run_dirs,
            scenario_configuration=config,
            reward_weights=agent.weights,
            normalization=agent.normalization,
            calibration_file=args.calibration_file,
            validate=args.validate,
            model=model,
            state_variant=args.state_variant,
            run_labels=run_labels,
        )

    output = Path(args.output)
    save_offline_dataset(output, dataset)

    print(f"Transitions: {dataset.num_transitions}")
    print(f"Episodes:    {dataset.num_episodes}")
    print(f"Observations: {dataset.observations.shape}")
    print(f"Actions in [{dataset.actions.min()}, {dataset.actions.max()}]")
    print(f"Reward mean: {float(dataset.rewards.mean()):.6f}")
    print(f"Saved dataset to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
