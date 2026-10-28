#!/usr/bin/env python3
"""Run the paper energy-saving scenario with a scripted (non-learned) policy.

The script drives the live ns-3 scenario through the Gymnasium environment and
leaves the run folder (``database.db``, ``bsState.txt``, ``EsActions.txt``,
``cu-*-cell-*.txt``, ``du-cell-*.txt``) in place, so that
``examples/collect_real_dataset.py`` can turn it into an offline dataset.

Policies
--------
``keep``
    Re-apply the current ON/OFF state (all cells stay ON).  This is the
    *Always On* baseline and the safest choice for a smoke test.
``random``
    Uniformly sample one of the 128 actions at every E2 indication.  This is
    the exploration policy used to collect the offline dataset.
``random-seconds``
    The paper's *Random* baseline: every second (10 indications at the paper
    E2 periodicity) a uniformly distributed number of cells is turned off and
    the remaining ones are turned on.

Usage::

    python examples/collect_run.py --policy random \
        --config src/environments/scenario_configurations/paper_es_use_case.json \
        --ns3_path ../ns-3-mmwave-oran --output_folder examples/output/real_runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # make ``environments`` importable
    sys.path.insert(0, str(_SRC))

from environments.es_env import N_CELLS  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "src/environments/scenario_configurations/paper_es_use_case.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_run.py",
        description="Run the paper ns-3 scenario with a scripted policy.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Scenario JSON (default: paper_es_use_case.json).",
    )
    parser.add_argument(
        "--ns3_path",
        default="/mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran/",
        help="Path to the ns-3 mmWave O-RAN tree.",
    )
    parser.add_argument(
        "--output_folder",
        default="output/real_runs",
        help="Folder that receives the run folders.",
    )
    parser.add_argument(
        "--policy",
        choices=("keep", "random", "random-seconds"),
        default="random",
        help="Scripted policy applied to the seven NR cells.",
    )
    parser.add_argument("--seed", type=int, default=555, help="ns-3 RngRun seed.")
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="Maximum number of E2 steps."
    )
    parser.add_argument(
        "--optimized", action="store_true", help="Use the optimized ns-3 build."
    )
    parser.add_argument(
        "--agent",
        default="PPO-2",
        help="Table I agent whose reward weights the environment uses.",
    )
    return parser


def _random_action(rng, keep_all_on_probability: float = 0.0) -> int:
    if keep_all_on_probability and rng.random() < keep_all_on_probability:
        return (1 << N_CELLS) - 1
    return int(rng.integers(0, 1 << N_CELLS))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    import numpy as np

    from environments.es_env import EnergySavingEnv
    from nsoran.drl.config import get_agent_config

    with open(args.config, "r", encoding="utf-8") as handle:
        scenario_configuration = json.load(handle)

    agent = get_agent_config(args.agent)
    rng = np.random.default_rng(args.seed)

    env = EnergySavingEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        reward_weights=agent.weights,
        normalization=agent.normalization,
    )

    print(f"Policy: {args.policy} | agent: {agent.name} | seed: {args.seed}")
    print("Launching the ns-3 scenario (this takes tens of minutes) ...")

    started = time.time()
    observation, info = env.reset(seed=args.seed)
    print(f"Run folder: {env.sim_path}")

    # The number of indications per second depends on the E2 periodicity.
    periodicity = float(env.scenario_configuration.get("indicationPeriodicity", 0.1))
    indications_per_second = max(1, int(round(1.0 / periodicity)))

    step = 0
    action = int((1 << N_CELLS) - 1)
    try:
        for step in range(1, args.num_steps + 1):
            if args.policy == "keep":
                action = int(
                    sum(
                        1 << index
                        for index in range(N_CELLS)
                        if env.bs_states_list()[index]
                    )
                )
            elif args.policy == "random":
                action = _random_action(rng)
            else:  # random-seconds
                if step % indications_per_second == 1:
                    off_count = int(rng.integers(0, N_CELLS + 1))
                    off_cells = set(
                        int(cell)
                        for cell in rng.choice(
                            np.arange(2, 2 + N_CELLS), size=off_count, replace=False
                        )
                    )
                    action = 0
                    for index in range(N_CELLS):
                        if (2 + index) not in off_cells:
                            action |= 1 << index

            observation, reward, terminated, truncated, info = env.step(action)
            bs_on = info.get("bs_on", -1)
            print(
                f"step {step:4d} | t={info.get('timestamps', {}).get('simulation_ms', 0)} "
                f"| action={action:3d} bs_on={bs_on} | reward={reward:+.4f} "
                f"| elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            if terminated or truncated:
                print("Simulation finished.")
                break
    finally:
        env.close()

    print(f"Run folder: {env.sim_path}")
    print(f"Total wall time: {time.time() - started:.0f}s over {step} indications")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
