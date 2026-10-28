#!/usr/bin/env python3
"""Run the paper baselines of ``scenario-three`` in parallel.

The campaign reproduces the baselines of the CCNC 2025 paper:

============ ========================================================
policy       ns-3 ``heuristicType``
============ ========================================================
always_on    ``-1`` -- no heuristic, every RF front-end stays powered on
random       ``3``  -- uniform number of cells turned off every second
static       ``1``  -- static sleeping of [3], driven by the (bsOn, bsIdle,
                      bsSleep, bsOff) tuple
dynamic      ``2``  -- dynamic sleeping of [3], same tuple
============ ========================================================

Each (policy, tuple, seed) combination is one independent ns-3 simulation
launched through the Gymnasium environment, so it produces the same
artifacts as ``examples/energy_saving.py``: a run folder with
``database.db``, ``bsState.txt``, ``EsActions.txt`` and the ns-3 KPI traces.
Those folders are the input of ``examples/collect_real_dataset.py``.

Every combination is a separate process of ``examples/energy_saving.py``:
the ns-3 process, its POSIX semaphores (``/sem_metrics_<uuid>``) and its
control file live in the run folder, so the runs are fully isolated and can
be executed concurrently.

Usage::

    # 4 policies x 3 tuples x 3 seeds = 24 runs, 6 at a time
    python scripts/run_paper_baselines.py --jobs 6 --seeds 555 556 557

    # only two policies, with an explicit output folder
    python scripts/run_paper_baselines.py --baselines always_on static \\
        --output-dir examples/output/baselines

The ns-3 build is done **once** by this script: ns-3's CMake cache cannot be
regenerated concurrently, so the workers are launched with ``--skip-build``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO_ROOT / "examples" / "energy_saving.py"
_DEFAULT_CONFIG = _REPO_ROOT / "src" / "environments" / "scenario_configurations" / "paper_es_use_case.json"

#: Paper baselines and the ns-3 ``heuristicType`` that implements them.
POLICIES: dict[str, int] = {
    "always_on": -1,
    "random": 3,
    "static": 1,
    "dynamic": 2,
}

#: ``(bsOn, bsIdle, bsSleep, bsOff)`` tuples of the paper for the static and
#: dynamic sleeping heuristics.  All of them add up to the 7 controlled cells.
TUPLES: tuple[tuple[int, int, int, int], ...] = (
    (4, 2, 0, 1),
    (3, 2, 0, 2),
    (2, 2, 0, 3),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_paper_baselines.py",
        description="Run the paper baselines of scenario-three in parallel.",
    )
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG),
                        help="Scenario JSON forwarded to ns-3 (default: the paper scenario).")
    parser.add_argument("--ns3-path", default="/mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN checkout.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "examples" / "output" / "paper_baselines"),
                        help="Folder that holds the run folders, the logs and the manifest.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[555, 556, 557],
                        help="ns-3 RngRun seeds (default: 555 556 557).")
    parser.add_argument("--baselines", nargs="+", default=list(POLICIES),
                        choices=list(POLICIES),
                        help="Baselines to run (default: all of them).")
    parser.add_argument("--tuples", nargs="+", default=None,
                        help="(bsOn,bsIdle,bsSleep,bsOff) tuples, e.g. 4,2,0,1 (default: the three "
                             "tuples of the paper).")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Maximum number of concurrent ns-3 simulations (default: 4). "
                             "A paper simulation grows to ~1.8 GB of resident memory, so "
                             "this is bounded by the RAM of the machine.")
    parser.add_argument("--min-free-mb", type=int, default=3000,
                        help="Do not start a simulation while less than this many MiB are "
                             "available (default: 3000).")
    parser.add_argument("--sim-time", type=float, default=None,
                        help="Override simTime (s) of the configuration file.")
    parser.add_argument("--ues", type=int, default=None,
                        help="Override the number of UEs per BS of the configuration file.")
    parser.add_argument("--timeout-min", type=float, default=180.0,
                        help="Wall-clock timeout of a single simulation, in minutes.  A "
                             "10 s paper simulation costs ~90 min of CPU and much more "
                             "when several of them run in parallel.")
    parser.add_argument("--only", nargs="+", default=None, metavar="JOB",
                        help="Run only the listed jobs (their names are printed by "
                             "--dry-run), e.g. --only always_on_seed557 random_seed557.")
    parser.add_argument("--manifest", default=None,
                        help="Manifest path (default: <output-dir>/manifest.json).  An "
                             "existing manifest is updated, never overwritten.")
    parser.add_argument("--dataset", default=None,
                        help="After the campaign, convert the runs that completed into "
                             "this offline dataset (.npz) for the DRL agents.")
    parser.add_argument("--agent", default="PPO-2",
                        help="Table I agent whose reward weights the dataset uses "
                             "(default: PPO-2).")
    parser.add_argument("--skip-build", action="store_true",
                        help="Do not configure/compile ns-3 (reuse the current build).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Run the jobs again even if they already completed "
                             "successfully in the manifest.")
    parser.add_argument("--reclassify", action="store_true",
                        help="Do not run anything: re-scan the logs of the runs already "
                             "listed in the manifest and update their status (a run whose "
                             "ns-3 process aborted is not a result).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used to run examples/energy_saving.py.")
    return parser


def parse_tuples(raw: list[str] | None) -> list[tuple[int, int, int, int]]:
    if not raw:
        return list(TUPLES)
    tuples = []
    for item in raw:
        parts = [int(part) for part in item.replace(" ", ",").split(",") if part != ""]
        if len(parts) != 4:
            raise ValueError(f"a tuple must have 4 values, got {item!r}")
        tuples.append(tuple(parts))
    return tuples


def build_ns3(ns3_path: str) -> None:
    """Configure and compile ns-3 once, before any parallel run starts."""
    build_program = "./ns3" if os.path.exists(os.path.join(ns3_path, "ns3")) else "./waf"
    print(f"[build] configuring ns-3 in {ns3_path}")
    subprocess.run(
        ["python3", build_program, "configure", "--enable-examples", "--disable-gtk", "--disable-werror"],
        cwd=ns3_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print("[build] compiling ns-3")
    subprocess.run(
        ["python3", build_program, "-j", str(os.cpu_count()), "build"],
        cwd=ns3_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print("[build] done")


def make_jobs(args, tuples):
    """Enumerate the (policy, tuple, seed) combinations of the campaign."""
    jobs = []
    for policy in args.baselines:
        heuristic = POLICIES[policy]
        # Only the static/dynamic heuristics are parametrized by a tuple.
        policy_tuples = tuples if policy in ("static", "dynamic") else [None]
        for bs_tuple in policy_tuples:
            for seed in args.seeds:
                suffix = "" if bs_tuple is None else "_{}_{}_{}_{}".format(*bs_tuple)
                name = f"{policy}{suffix}_seed{seed}"
                command = [
                    args.python, str(_EXAMPLE),
                    "--config", args.config,
                    "--output_folder", str(Path(args.output_dir) / name),
                    "--ns3_path", args.ns3_path,
                    "--seed", str(seed),
                    "--heuristic", str(heuristic),
                    "--skip-build",
                    "--quiet",
                    # The baselines decide inside ns-3: the control file must stay
                    # empty, otherwise the environment would override the
                    # heuristic and would evict the users of the cells it turned
                    # off at every indication.
                    "--no-action",
                ]
                if args.sim_time is not None:
                    command += ["--sim-time", str(args.sim_time)]
                if args.ues is not None:
                    command += ["--ues", str(args.ues)]
                if bs_tuple is not None:
                    command += [
                        "--bs-on", str(bs_tuple[0]),
                        "--bs-idle", str(bs_tuple[1]),
                        "--bs-sleep", str(bs_tuple[2]),
                        "--bs-off", str(bs_tuple[3]),
                    ]
                jobs.append({
                    "name": name,
                    "policy": policy,
                    "tuple": list(bs_tuple) if bs_tuple is not None else None,
                    "seed": seed,
                    "heuristic": heuristic,
                    "command": command,
                })
    if args.only:
        wanted = set(args.only)
        known = {job["name"] for job in jobs}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"error: unknown job(s): {', '.join(sorted(unknown))}")
        jobs = [job for job in jobs if job["name"] in wanted]
    return jobs


def completed_jobs(manifest_path: Path) -> set[str]:
    """Return the names of the runs that already completed successfully."""
    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        job.get("name")
        for job in manifest.get("jobs", [])
        if job.get("status") == "ok"
    }


#: Markers that identify an ns-3 run that died instead of completing.
CRASH_MARKERS = (
    "Simulation exited with an error",
    "ns-3 exited with code",
    "the episode ended early",
    "NS_FATAL, terminating",
    "NS_ASSERT failed",
    "terminate called without an active exception",
)


def free_memory_mb() -> int:
    """Return the available memory of the machine, in MiB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def wait_for_memory(min_free_mb: int, timeout_s: float = 3600.0) -> int:
    """Block until the machine has enough free memory to start another run.

    A paper simulation of this scenario grows to ~1.8 GB of resident memory
    (the fork accumulates state over the simulated window), so starting too
    many of them at once makes the kernel OOM-kill them: a killed run is not a
    result.  The gate keeps the campaign below the memory limit of the
    machine.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        free = free_memory_mb()
        if free >= min_free_mb or time.monotonic() >= deadline:
            return free
        print(f"[wait] only {free} MiB free, waiting for {min_free_mb} MiB", flush=True)
        time.sleep(30)


def classify_log(log_path: Path) -> str | None:
    """Return the crash signature of a finished run, if any.

    ns-3 aborts (assertions, ``NS_FATAL_ERROR``, ...) leave their message in
    the stdout/stderr of the run folder and the Python driver reports it with
    "Simulation exited with an error".  A run that died is *not* a result.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for marker in CRASH_MARKERS:
        if marker in text:
            return marker
    return None


def run_job(job, log_dir: Path, timeout_s: float, min_free_mb: int = 0) -> dict:
    """Run one simulation and collect its metadata."""
    log_path = log_dir / f"{job['name']}.log"
    started = time.time()
    record = {
        "name": job["name"],
        "policy": job["policy"],
        "tuple": job["tuple"],
        "seed": job["seed"],
        "heuristic": job["heuristic"],
        "command": " ".join(shlex.quote(part) for part in job["command"]),
        "log": str(log_path),
        "status": "running",
    }
    print(f"[start] {job['name']}", flush=True)
    free = wait_for_memory(min_free_mb)
    print(f"[start] {job['name']} ({free} MiB free)", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                job["command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                cwd=str(_REPO_ROOT),
            )
            record["returncode"] = completed.returncode
            record["status"] = "ok" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            record["returncode"] = None
            record["status"] = "timeout"
    record["duration_s"] = round(time.time() - started, 1)

    crash = classify_log(log_path)
    if crash is not None:
        record["status"] = "crashed"
        record["crash"] = crash

    # The ns-3 run folder is the sub-folder of the job folder that holds the datalake.
    # A resumed job may already contain the folder of a previous crashed attempt, so
    # record only the newest one: the manifest must describe this attempt, not mix
    # the transitions of an old aborted run with the new one.
    job_dir = Path(job["command"][job["command"].index("--output_folder") + 1])
    candidates = sorted(
        child for child in job_dir.glob("*") if (child / "database.db").is_file()
    )
    run_dirs = (
        [max(candidates, key=lambda child: child.stat().st_mtime)]
        if candidates
        else []
    )
    record["run_dirs"] = [str(child) for child in run_dirs]
    if run_dirs:
        record["run_dir"] = str(run_dirs[-1])
        bs_state = run_dirs[-1] / "bsState.txt"
        if bs_state.is_file():
            lines = [line for line in bs_state.read_text().splitlines() if line.strip()]
            record["bsstate_rows"] = len(lines)
            if lines:
                # First column of bsState.txt is the simulation time in seconds.
                record["sim_time_covered_s"] = float(lines[-1].split()[0])
    print(f"[done] {job['name']}: {record['status']} in {record['duration_s']} s", flush=True)
    return record


def reclassify(manifest_path: Path) -> int:
    """Re-scan the logs of the recorded runs and fix their status."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = 0
    for job in manifest.get("jobs", []):
        log_path = Path(job.get("log") or "")
        if not log_path.is_file():
            continue
        crash = classify_log(log_path)
        status = job.get("status")
        if crash is not None and status != "crashed":
            job["status"] = "crashed"
            job["crash"] = crash
            changed += 1
        elif crash is None and status == "crashed":
            job["status"] = "ok"
            job.pop("crash", None)
            changed += 1
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Run finished" in text:
            for line in text.splitlines():
                if line.startswith("Run finished"):
                    job["summary"] = line.strip()
                    break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"reclassified {changed} run(s) in {manifest_path}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tuples = parse_tuples(args.tuples)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"

    jobs = make_jobs(args, tuples)
    if not args.force:
        done = completed_jobs(manifest_path)
        if done:
            skipped = [job["name"] for job in jobs if job["name"] in done]
            jobs = [job for job in jobs if job["name"] not in done]
            print(f"skipping {len(skipped)} run(s) that already completed: "
                  f"{', '.join(skipped)}")
    if not jobs:
        print("nothing to do")
        return 0

    print(f"{len(jobs)} simulation(s) planned, {args.jobs} at a time")
    for job in jobs:
        print(f"  {job['name']:<28} {job['policy']:<10} tuple={job['tuple']} seed={job['seed']}")
    if args.dry_run:
        return 0

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.reclassify:
        if not manifest_path.is_file():
            print(f"error: no manifest at {manifest_path}", file=sys.stderr)
            return 2
        return reclassify(manifest_path)

    if not args.skip_build:
        build_ns3(args.ns3_path)

    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ns3_path": args.ns3_path,
        "config": args.config,
        "seeds": args.seeds,
        "tuples": [list(item) for item in tuples],
        "jobs": [],
    }
    if manifest_path.is_file():
        # A resumed campaign must not lose the results of the previous runs.
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        merged = {job.get("name"): job for job in previous.get("jobs", [])}
        for job in jobs:
            merged.setdefault(job["name"], {"name": job["name"], "status": "pending"})
        manifest["jobs"] = list(merged.values())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    timeout_s = args.timeout_min * 60.0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_job, job, log_dir, timeout_s, args.min_free_mb): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
            except Exception as error:  # pragma: no cover - defensive
                record = {"name": job["name"], "status": "error", "error": repr(error)}
            manifest["jobs"] = [item for item in manifest["jobs"] if item.get("name") != job["name"]]
            manifest["jobs"].append(record)
            manifest["jobs"].sort(key=lambda item: item["name"])
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.time() - started
    ok = sum(1 for job in manifest["jobs"] if job.get("status") == "ok")
    print(f"\n{ok}/{len(jobs)} run(s) succeeded in {elapsed / 60.0:.1f} min")
    print(f"manifest: {manifest_path}")
    for job in manifest["jobs"]:
        if job.get("status") != "ok":
            print(f"  FAILED {job['name']}: {job.get('status')} (log: {job.get('log')})")

    if args.dataset:
        build_dataset(manifest, args)
    return 0 if ok == len(manifest["jobs"]) else 1


def build_dataset(manifest: dict, args) -> None:
    """Convert the runs that completed into one offline dataset."""
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from nsoran.drl.config import get_agent_config
    from nsoran.drl.offline import save_offline_dataset
    from nsoran.drl.real_data import dataset_from_ns3_runs, load_scenario_config

    run_dirs = [
        run_dir
        for job in manifest.get("jobs", [])
        if job.get("status") == "ok"
        for run_dir in job.get("run_dirs", [])
    ]
    if not run_dirs:
        print("no completed run to convert into a dataset", file=sys.stderr)
        return
    agent = get_agent_config(args.agent)
    config = load_scenario_config(args.config)
    dataset = dataset_from_ns3_runs(
        run_dirs,
        scenario_configuration=config,
        reward_weights=agent.weights,
        normalization=agent.normalization,
    )
    save_offline_dataset(args.dataset, dataset)
    print(f"\ndataset: {args.dataset} "
          f"({dataset.num_transitions} transitions, {dataset.num_episodes} episodes)")


if __name__ == "__main__":
    raise SystemExit(main())
