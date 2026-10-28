#!/usr/bin/env python3
"""Summarise a baseline campaign into the paper's energy/throughput trade-off.

The metrics are recomputed with the *canonical* code path of the project
(``nsoran.drl.real_data.raw_reward_components_from_ns3_runs``), i.e. exactly
the same reconstruction of the 85-dimensional state that the live environment
and ``examples/compare_runs.py`` use.  For every run the script reports:

* the mean downlink throughput per E2 indication;
* the mean energy proxy per E2 indication (``TB.TotNbrDl.1 * P_TX``);
* the number of UEs in radio link failure (SINR < -5 dB);
* the number of RF front-ends that stayed powered on;
* the activation cost ``0.9 ** (0.01 * T_on_ms)``.

Every policy is then compared with the *Always On* baseline, which is the
reference used by the paper ("PPO-2 saves 24.2% energy vs Always On at
-52.3% throughput").

``--window`` compares every run over the same number of first indications.
This matters because the ns-3 fork currently aborts in the IP stack during the
paper scenario: the runs of a campaign do not all cover the same number of
indications, and comparing them over their whole (different) length would be
biased.

Usage::

    python scripts/summarize_baselines.py --campaign-dir examples/output/real_runs
    python scripts/summarize_baselines.py --campaign-dir examples/output/real_runs \\
        --window 20 --csv baselines.csv --markdown baselines.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from nsoran.drl.real_data import (  # noqa: E402
    load_scenario_config,
    raw_reward_components_from_ns3_runs,
)

#: Baselines of the campaign, as implemented by ``scripts/run_paper_baselines.py``.
POLICIES = ("always_on", "random", "static", "dynamic")

DEFAULT_CONFIG = _REPO_ROOT / "src/environments/scenario_configurations/paper_es_use_case.json"


def _policy_of(job_stem: str) -> str:
    """Return the policy of a job name such as ``static_4_2_0_1``."""
    for policy in POLICIES:
        if job_stem == policy or job_stem.startswith(policy + "_"):
            return policy
    return job_stem.split("_")[0]


def _tuple_of(job_stem: str) -> list[int] | None:
    """Return the ``(bsOn, bsIdle, bsSleep, bsOff)`` tuple of a job name."""
    parts = job_stem.split("_")
    if len(parts) == 5 and all(part.isdigit() for part in parts[1:]):
        return [int(part) for part in parts[1:]]
    return None


def load_manifest(campaign_dir: Path) -> list[dict]:
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload.get("jobs", [])


def discover_runs(campaign_dir: Path) -> list[dict]:
    """Find the run folders of a campaign that are not in the manifest yet.

    The manifest is written when a run finishes, so a campaign that is still
    running has folders that are not listed yet.  The job name carries the
    policy, the tuple and the seed, e.g. ``static_4_2_0_1_seed555``.
    """
    jobs = []
    for job_dir in sorted(campaign_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        for run_dir in sorted(job_dir.glob("*")):
            if not (run_dir / "database.db").is_file():
                continue
            name = job_dir.name
            policy, _, seed = name.rpartition("_seed")
            jobs.append({
                "name": name,
                "policy": _policy_of(policy),
                "tuple": _tuple_of(policy),
                "seed": int(seed) if seed.isdigit() else None,
                "status": "ok",
                "run_dirs": [str(run_dir)],
            })
    return jobs


def summarise_run(run_dir: Path, config: dict, window: int | None = None) -> dict | None:
    """Reduce one run folder to the KPIs used by the paper."""
    components = raw_reward_components_from_ns3_runs(
        [run_dir], scenario_configuration=config
    )
    if not components or not len(components.get("throughput", [])):
        return None

    def series(name: str) -> np.ndarray:
        values = np.asarray(components.get(name, []), dtype=float)
        return values[:window] if window else values

    throughput = series("throughput")
    energy = series("energy")
    rlf = series("rlf")
    on_cost = series("es_on_cost")
    bs_on = series("bs_on")

    return {
        "run_dir": str(run_dir),
        "indications": int(throughput.size),
        "throughput_mean": float(throughput.mean()),
        "throughput_total": float(throughput.sum()),
        "energy_mean": float(energy.mean()),
        "energy_total": float(energy.sum()),
        "rlf_mean": float(rlf.mean()),
        "rlf_max": float(rlf.max()) if rlf.size else 0.0,
        "on_cost_mean": float(on_cost.mean()) if on_cost.size else 0.0,
        "bs_on_mean": float(bs_on.mean()) if bs_on.size else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize_baselines.py",
        description="Summarise a baseline campaign into the paper's trade-off table.",
    )
    parser.add_argument("--campaign-dir", required=True,
                        help="Folder that holds the campaign run folders and manifest.json.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Scenario JSON of the runs (default: the paper scenario).")
    parser.add_argument("--window", type=int, default=None,
                        help="Compare every run over the same number of first E2 "
                             "indications (default: the whole run of each policy).")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path.")
    parser.add_argument("--include-crashed", action="store_true",
                        help="Also summarise the runs whose ns-3 process aborted "
                             "(their window is shorter than the simulated window).")
    args = parser.parse_args(argv)

    campaign_dir = Path(args.campaign_dir)
    config = load_scenario_config(args.config)

    manifest_jobs = load_manifest(campaign_dir)
    manifest_names = {job.get("name") for job in manifest_jobs}

    selected: list[dict] = []
    seen_runs: set[str] = set()
    rejected: list[dict] = []

    def _add(job, run_dir):
        if run_dir in seen_runs:
            return
        seen_runs.add(run_dir)
        selected.append({**job, "run_dirs": [run_dir]})

    for job in manifest_jobs:
        if job.get("status") != "ok" and not args.include_crashed:
            rejected.append(job)
            continue
        for run_dir in job.get("run_dirs", []):
            _add(job, run_dir)
    # A campaign that is still running has folders that are not in the manifest
    # yet.  They are only used when there is no manifest at all: the manifest
    # is the authority on which runs are results.
    if not manifest_jobs:
        for job in discover_runs(campaign_dir):
            for run_dir in job["run_dirs"]:
                _add(job, run_dir)

    if not selected:
        print(f"error: no valid run folder found in {campaign_dir}", file=sys.stderr)
        return 2

    records = []
    for job in selected:
        for run_dir in job.get("run_dirs", []):
            summary = summarise_run(Path(run_dir), config, args.window)
            if summary is None:
                continue
            summary.update({
                "policy": job.get("policy"),
                "tuple": job.get("tuple"),
                "seed": job.get("seed"),
                "name": job.get("name"),
                "status": job.get("status"),
            })
            records.append(summary)

    if not records:
        print("error: no run with data was found", file=sys.stderr)
        return 1

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_policy[record["policy"]].append(record)

    reference = by_policy.get("always_on")
    reference_throughput = statistics.fmean([r["throughput_mean"] for r in reference]) if reference else None
    reference_energy = statistics.fmean([r["energy_mean"] for r in reference]) if reference else None

    print(f"window: {args.window if args.window else 'whole run'} indications")
    print(f"{'policy':<28} {'n':>2} {'indic':>6} {'throughput':>12} {'energy':>12} {'rlf':>7} "
          f"{'bs_on':>6} {'dThr%':>8} {'dEnergy%':>9}")
    rows = []
    for policy, group in sorted(by_policy.items()):
        thr = statistics.fmean([r["throughput_mean"] for r in group])
        en = statistics.fmean([r["energy_mean"] for r in group])
        rlf = statistics.fmean([r["rlf_mean"] for r in group])
        bs_on = statistics.fmean([r["bs_on_mean"] for r in group])
        ind = statistics.fmean([r["indications"] for r in group])
        d_thr = (thr - reference_throughput) / reference_throughput * 100.0 if reference_throughput else float("nan")
        d_en = (en - reference_energy) / reference_energy * 100.0 if reference_energy else float("nan")
        print(f"{policy:<28} {len(group):>2} {ind:>6.1f} {thr:>12.1f} {en:>12.1f} {rlf:>7.2f} "
              f"{bs_on:>6.2f} {d_thr:>8.2f} {d_en:>9.2f}")
        rows.append({
            "policy": policy,
            "runs": len(group),
            "indications_mean": ind,
            "throughput_mean": thr,
            "energy_mean": en,
            "rlf_mean": rlf,
            "rlf_max": max(r["rlf_max"] for r in group),
            "bs_on_mean": bs_on,
            "on_cost_mean": statistics.fmean([r["on_cost_mean"] for r in group]),
            "throughput_delta_pct_vs_always_on": d_thr,
            "energy_delta_pct_vs_always_on": d_en,
        })

    if rejected:
        print(f"\n{len(rejected)} run(s) excluded (ns-3 did not complete the simulated window):")
        for job in rejected:
            print(f"  {job.get('name')}: {job.get('status')} ({job.get('crash', 'no crash marker')})")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv}")

    if args.markdown:
        lines = ["| policy | runs | indications | throughput | energy | RLF | BS on | dThr % | dEnergy % |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(
                f"| {row['policy']} | {row['runs']} | {row['indications_mean']:.1f} | "
                f"{row['throughput_mean']:.1f} | {row['energy_mean']:.1f} | "
                f"{row['rlf_mean']:.2f} | {row['bs_on_mean']:.2f} | "
                f"{row['throughput_delta_pct_vs_always_on']:.2f} | "
                f"{row['energy_delta_pct_vs_always_on']:.2f} |"
            )
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Markdown written to {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
