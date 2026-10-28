#!/usr/bin/env python3
"""Fit the reward-component calibration on real ns-3 runs.

The paper normalises every component of Eq. (2) with the quantiles of its
reference distribution before applying the weights ``w1..w4``.  This script
collects the *raw* component samples (throughput, energy, RLF, activation cost
and number of active cells) of one or more real ns-3 runs and writes them as the
calibration file expected by ``EnergySavingEnv(calibration_file=...)``.

Usage::

    python examples/fit_reward_calibration.py \
        --runs-dir examples/output/real_runs \
        --output examples/output/reward_calibration.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from _model_cli import (  # noqa: E402
    add_environment_model_arguments,
    model_from_args,
)
from nsoran.drl.real_data import (  # noqa: E402
    REWARD_COMPONENT_ORDER,
    load_scenario_config,
    raw_reward_components_from_ns3_runs,
)

DEFAULT_CONFIG = (
    _REPO_ROOT / "src/environments/scenario_configurations/paper_es_use_case.json"
)


def _find_run_dirs(root: Path) -> list[Path]:
    if (root / "database.db").is_file():
        return [root]
    return sorted(child.parent for child in root.rglob("database.db"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fit_reward_calibration.py",
        description=(
            "Fit the reward-component calibration of Eq. (2) on real ns-3 runs."
        ),
    )
    parser.add_argument(
        "--run-dir", action="append", default=[], help="A run folder (repeatable)."
    )
    parser.add_argument(
        "--runs-dir", default=None, help="Folder that contains the run folders."
    )
    parser.add_argument(
        "--output", default="reward_calibration.npz", help="Output .npz path."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Scenario JSON of the runs."
    )
    add_environment_model_arguments(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dirs = [Path(item) for item in args.run_dir]
    if args.runs_dir:
        run_dirs.extend(_find_run_dirs(Path(args.runs_dir)))
    if not run_dirs:
        print("error: no run folders given", file=sys.stderr)
        return 2

    config = load_scenario_config(args.config)
    model = model_from_args(args)
    print(
        "Fitting the calibration on "
        f"{len(run_dirs)} run(s) with model {model.summary()} ..."
    )
    samples = raw_reward_components_from_ns3_runs(
        run_dirs,
        scenario_configuration=config,
        model=model,
        include_cells=args.per_cell_normalization,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: v for k, v in samples.items() if v.size})

    print(f"Wrote {output}")
    for name in REWARD_COMPONENT_ORDER:
        values = samples.get(name)
        if values is None or values.size == 0:
            continue
        print(
            f"  {name:<11} n={values.size:<6} min={values.min():.4g} "
            f"median={np.median(values):.4g} max={values.max():.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
