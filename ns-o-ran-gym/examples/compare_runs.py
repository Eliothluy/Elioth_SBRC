#!/usr/bin/env python3
"""Aggregate real ns-3 runs into the paper's comparison metrics.

For every run folder the script rebuilds the per-indication 85-dimensional state
with the very same code path as the live environment and derives the four KPMs
the paper reports:

* **throughput** -- sum of ``QosFlow.PdcpPduVolumeDL_Filter`` over the NR cells;
* **energy**     -- sum of the EARTH macro power of each cell (watts, D1);
  ``--energy-model proxy`` recovers the legacy ``TB.TotNbrDl.1 * P_TX`` proxy;
* **RLF**        -- number of UEs whose serving SINR is below -5 dB;
* **activation** -- the ``0.9 ** (0.01 * T_on_ms)`` cost of the cells that are ON
  (dense, D2; ``--es-cost-mode impulse`` for the transition-only ablation).

Every run is then compared with a reference run (the *Always On* baseline of the
paper): the report contains the relative energy saving and the relative
throughput change, i.e. the two numbers of Table VI of the paper.

Usage::

    python examples/compare_runs.py \
        --run "Always On=examples/output/real_runs/mine_alwayson_seed558" \
        --run "Random=examples/output/real_runs/mine_random_seed557" \
        --reference "Always On"
"""

from __future__ import annotations

import argparse
import csv
import json
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
    EnvironmentModelConfig,
    load_scenario_config,
    raw_reward_components_from_ns3_runs,
)

DEFAULT_CONFIG = (
    _REPO_ROOT / "src/environments/scenario_configurations/paper_es_use_case.json"
)

def _expand_runs(directory: Path) -> list[Path]:
    """Return the run folders inside ``directory``.

    ``directory`` may be a run folder itself (it holds a ``database.db``) or a
    parent folder that contains one run folder per simulation.
    """

    if (directory / "database.db").is_file():
        return [directory]
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and (child / "database.db").is_file()
    )


def pdcp_latency_stats(run_dirs: list[Path]) -> dict:
    """Median/p95 of the per-line downlink PDCP SDU delay (ms, TS 28.552).

    Reads the raw ``cu-up-cell-*.txt`` traces directly: the legacy
    ``gnb_cu_up`` SQLite table used to drop these two columns, so this works
    for every run already on disk (6.237 samples per complete run =
    63 UEs x 99 indications).
    """

    column = "DRB.PdcpSduDelayDl(cellAverageLatency)"
    values: list[float] = []
    for run_dir in run_dirs:
        for file_path in sorted(run_dir.glob("cu-up-cell-*.txt")):
            with open(file_path, "r") as handle:
                for row in csv.DictReader(handle):
                    raw = (row.get(column) or "").strip()
                    if raw:
                        values.append(float(raw))
    if not values:
        return {
            "pdcp_latency_median": None,
            "pdcp_latency_p95": None,
            "pdcp_latency_samples": 0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "pdcp_latency_median": float(np.median(arr)),
        "pdcp_latency_p95": float(np.percentile(arr, 95)),
        "pdcp_latency_samples": int(arr.size),
    }


def run_metrics(
    run_dirs: list[Path],
    config: dict,
    window: int | None = None,
    model: EnvironmentModelConfig | None = None,
) -> dict:
    """Aggregate one or more ns-3 runs into the paper's KPMs.

    The raw per-indication totals are used (not the min-max normalised state),
    so the numbers are directly comparable with the paper's KPMs.  ``model``
    selects the D1-D3 energy/activation-cost model; the defaults are the
    approved main configuration (EARTH watts + dense activation cost).
    """

    samples = raw_reward_components_from_ns3_runs(
        run_dirs, scenario_configuration=config, model=model
    )

    if window is not None and window > 0:
        # Compare the runs over the same time window: a run that was aborted
        # by ns-3 covers fewer E2 indications and its first indications are
        # the ramp-up of the scenario.
        samples = {
            name: np.asarray(values, dtype=float)[:window]
            for name, values in samples.items()
        }

    throughput = np.asarray(samples["throughput"], dtype=float)
    energy = np.asarray(samples["energy"], dtype=float)
    rlf = np.asarray(samples["rlf"], dtype=float)
    activation = np.asarray(samples["es_on_cost"], dtype=float)
    bs_on = np.asarray(samples["bs_on"], dtype=float)

    return {
        "run_dir": [str(item) for item in run_dirs],
        "indications": int(throughput.size),
        "throughput_total": float(throughput.sum()),
        "throughput_mean": float(throughput.mean()) if throughput.size else 0.0,
        "energy_total": float(energy.sum()),
        "energy_mean": float(energy.mean()) if energy.size else 0.0,
        "rlf_mean": float(rlf.mean()) if rlf.size else 0.0,
        "rlf_max": float(rlf.max()) if rlf.size else 0.0,
        "rlf_steps": int((rlf > 0).sum()),
        "activation_total": float(activation.sum()),
        "activation_mean": float(activation.mean()) if activation.size else 0.0,
        "bs_on_mean": float(bs_on.mean()) if bs_on.size else 0.0,
        "bs_on_min": float(bs_on.min()) if bs_on.size else 0.0,
        **pdcp_latency_stats(run_dirs),
    }


def compare(metrics: dict, reference: dict) -> dict:
    """Relative energy saving / throughput change with respect to the reference.

    The comparison uses the *mean per indication* so that runs of different
    length (a run that was interrupted by an ns-3 abort covers fewer E2
    indications) stay comparable.
    """

    energy_ref = reference["energy_mean"]
    throughput_ref = reference["throughput_mean"]
    energy_saving = (
        100.0 * (energy_ref - metrics["energy_mean"]) / energy_ref
        if energy_ref
        else None
    )
    throughput_change = (
        100.0 * (metrics["throughput_mean"] - throughput_ref) / throughput_ref
        if throughput_ref
        else None
    )
    out = dict(metrics)
    out["energy_saving_pct"] = energy_saving
    out["throughput_change_pct"] = throughput_change
    out["rlf_delta_vs_reference"] = metrics["rlf_max"] - reference["rlf_max"]
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Aggregate real ns-3 runs into the paper's comparison metrics.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="A run folder with its label (repeatable).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Label of the reference run (default: the first one).",
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Scenario JSON of the runs."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help=(
            "Compare every run over the same number of first E2 indications "
            "(default: the whole run of each policy)."
        ),
    )
    parser.add_argument(
        "--json", default=None, help="Write the report to this JSON file."
    )
    add_environment_model_arguments(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.run:
        print("error: at least one --run LABEL=DIR is required", file=sys.stderr)
        return 2

    runs: dict[str, list[Path]] = {}
    for item in args.run:
        if "=" not in item:
            print(f"error: --run expects LABEL=DIR, got {item!r}", file=sys.stderr)
            return 2
        label, directory = item.split("=", 1)
        path = Path(directory)
        if not path.is_dir():
            print(f"error: {path} is not a directory", file=sys.stderr)
            return 2
        if not _expand_runs(path):
            print(f"error: no database.db found in {path}", file=sys.stderr)
            return 2
        # The same label may be given several times: the runs are then
        # aggregated together (e.g. the three seeds of a baseline).
        runs.setdefault(label, []).extend(_expand_runs(path))

    config = load_scenario_config(args.config)
    model = model_from_args(args)
    print(f"Environment model: {model.summary()}")

    metrics = {}
    for label, paths in runs.items():
        print(f"Aggregating {label} ({len(paths)} run(s)) ...")
        metrics[label] = run_metrics(paths, config, args.window, model=model)

    reference_label = args.reference or next(iter(runs))
    if reference_label not in metrics:
        print(
            f"error: reference {reference_label!r} is not one of {list(runs)}",
            file=sys.stderr,
        )
        return 2

    report = {
        label: compare(values, metrics[reference_label])
        for label, values in metrics.items()
    }

    header = (
        f"{'policy':<16}{'indic.':>7}{'energy':>16}{'throughput':>16}"
        f"{'RLF max':>8}{'RLF mean':>10}{'bs_on':>8}{'dEnergy%':>10}{'dThrpt%':>10}"
    )
    print()
    print(f"Reference run: {reference_label}")
    print(header)
    print("-" * len(header))
    for label, values in report.items():
        energy_saving = values["energy_saving_pct"]
        throughput_change = values["throughput_change_pct"]
        print(
            f"{label:<16}{values['indications']:>7d}"
            f"{values['energy_mean']:>16.1f}{values['throughput_mean']:>16.1f}"
            f"{values['rlf_max']:>8.0f}{values['rlf_mean']:>10.2f}"
            f"{values['bs_on_mean']:>8.2f}"
            f"{energy_saving:>10.1f}{throughput_change:>10.1f}"
        )
    print(
        "\nRLF max is the largest number of UEs below -5 dB observed in a single "
        "E2 indication (the paper's 'max 2 UEs in RLF' metric)."
    )

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reference": reference_label,
                    "environment_model": model.summary(),
                    "runs": report,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
