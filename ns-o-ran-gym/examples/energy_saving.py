"""Run the energy-saving environment against ns-3.

This is the reference driver of the paper pipeline: it launches the
``scenario-three`` ns-3 simulation through the Gymnasium environment
(``environments.es_env.EnergySavingEnv``), keeps the RF front-ends in the
requested state and reports the paper KPIs at every E2 indication.

It is also the entry point of the parallel baseline campaign
(``scripts/run_paper_baselines.py``), which spawns one process of this script
per (policy, tuple, seed) combination.

Examples
--------
Paper scenario, Always On baseline (no external action, ns-3 keeps every cell
powered on)::

    python examples/energy_saving.py \\
        --config src/environments/scenario_configurations/paper_es_use_case.json \\
        --seed 555 --no-action

Paper scenario, Static baseline with the ``(4, 2, 0, 1)`` tuple::

    python examples/energy_saving.py \\
        --config src/environments/scenario_configurations/paper_es_use_case.json \\
        --seed 555 --no-action --heuristic 1 --bs-on 4 --bs-idle 2 --bs-off 1

DRL-style run (the environment applies the current cell state every step)::

    python examples/energy_saving.py --config ... --seed 555
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # make ``environments`` importable without installing
    sys.path.insert(0, str(_SRC))

from environments.es_env import EnergySavingEnv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the energy saving environment")
    parser.add_argument("--config", type=str, default="/mnt/legitbet/elioth_sbrc/ns-o-ran-gym/src/environments/scenario_configurations/es_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")

    # --- paper campaign knobs -----------------------------------------
    parser.add_argument("--seed", type=int, default=None,
                        help="ns-3 RngRun seed (defaults to the value of the config file)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name of the sub-folder of --output_folder that holds this run")
    parser.add_argument("--no-action", action="store_true",
                        help="Do not write any energy-saving action: ns-3 keeps the "
                             "ownership of the cell states (used by the Always On, "
                             "Random, Static and Dynamic baselines)")
    parser.add_argument("--heuristic", type=int, default=None,
                        help="ns-3 heuristicType (-1 none, 0 random sleeping, 1 static, "
                             "2 dynamic, 3 paper random policy)")
    parser.add_argument("--bs-on", type=int, default=None, help="bsOn (static/dynamic)")
    parser.add_argument("--bs-idle", type=int, default=None, help="bsIdle (static/dynamic)")
    parser.add_argument("--bs-sleep", type=int, default=None, help="bsSleep (static/dynamic)")
    parser.add_argument("--bs-off", type=int, default=None, help="bsOff (static/dynamic)")
    parser.add_argument("--sinr-th", type=float, default=None,
                        help="SINR threshold of the static/dynamic heuristic (73 == 13 dB)")
    parser.add_argument("--sim-time", type=float, default=None,
                        help="Override simTime (s) of the configuration file")
    parser.add_argument("--ues", type=int, default=None,
                        help="Override the number of UEs per BS of the configuration file")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse the existing ns-3 build instead of configuring and "
                             "compiling it (mandatory when several runs are launched in "
                             "parallel: ns-3's CMake cache cannot be regenerated "
                             "concurrently)")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the step/reward summary instead of the full "
                             "observation and info dictionaries")
    return parser


def _apply_overrides(scenario_configuration, args):
    """Apply the CLI overrides to the scenario configuration.

    The ns-3 side expects a ``{key: [value]}`` mapping: every value is
    forwarded to the binary as ``--key=value``.
    """
    overrides = {
        "RngRun": args.seed,
        "heuristicType": args.heuristic,
        "bsOn": args.bs_on,
        "bsIdle": args.bs_idle,
        "bsSleep": args.bs_sleep,
        "bsOff": args.bs_off,
        "sinrTh": args.sinr_th,
        "simTime": args.sim_time,
        "ues": args.ues,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key in ("RngRun", "heuristicType", "bsOn", "bsIdle", "bsSleep", "bsOff", "ues"):
            scenario_configuration[key] = [int(value)]
        else:
            scenario_configuration[key] = [value]
    return scenario_configuration


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps
    optimized = args.optimized

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        return -1

    scenario_configuration = _apply_overrides(json.loads(params), args)

    if args.run_name:
        output_folder = os.path.join(output_folder, args.run_name)

    print('Creating ES Environment')
    env = EnergySavingEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration,
                          output_folder=output_folder, optimized=optimized,
                          skip_build=args.skip_build)

    print('Environment Created!')

    started = time.time()
    steps = 0
    outcome = "ok"
    try:
        print('Launch reset ', end='', flush=True)
        obs, info = env.reset()
        print('done')

        if not args.quiet:
            print(f'First set of observations {obs}')
            print(f'Info {info}')

        # Action logic
        for step in range(2, num_steps):
            # Keep the current ON/OFF state (the environment accepts both the
            # paper's integer action and a sequence of seven ON/OFF values).
            # The baselines let ns-3 decide instead: no action is written to
            # the control file, the environment only releases the semaphore.
            if args.no_action:
                model_action = None
            else:
                model_action = env.bs_states_list()

            print(f'Step {step} ', end='', flush=True)
            if args.no_action:
                obs, reward, terminated, truncated, info = env.step_baseline()
            else:
                obs, reward, terminated, truncated, info = env.step(model_action)

            print('done', flush=True)
            steps = step

            print(f'Status t = {step}')
            print(f'Reward {reward}')
            print(f'Terminated {terminated}')
            print(f'Truncated {truncated}')
            if not args.quiet:
                print(f'Observations {obs}')
                print(f'Info {info}')

            # If the environment is over, exit
            if terminated:
                break

            # If the episode is up (environment still running), then start another one
            if truncated:
                # The episode was cut short (timeout or a crashed ns-3): the
                # run does not cover the whole simulated window.
                outcome = "truncated"
                break # We don't want this outside the training
                obs, info = env.reset()
    finally:
        elapsed = time.time() - started
        print(f'Run finished after {steps} steps / {elapsed:.1f} s wall time')
        exitcode = (env.sim_result.get('meta') or {}).get('exitcode')
        env.close()
        if exitcode is not None and exitcode != 0:
            # ns-3 died (abort, assertion, ...): the run is not a valid result.
            print(f'ns-3 exited with code {exitcode}: the run is invalid', file=sys.stderr)
            return 1
        if outcome != "ok":
            print(f'the episode ended early ({outcome}): the run is incomplete', file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
