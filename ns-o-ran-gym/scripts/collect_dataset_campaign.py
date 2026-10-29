#!/usr/bin/env python3
"""Tier M dataset campaign driver (experiment-configs/dataset_plan.json).

Despacha a coleta de comportamento para treino offline:
- policies scripted (keep / random / random-seconds): um job por run via
  examples/collect_run.py;
- policies de heuristicas ns-3 (static / dynamic): um lote por familia via
  scripts/run_paper_baselines.py (que sabe montar as flags --heuristic e
  --no-action), com as seeds do plano.

Retomavel: jobs scripted com pasta concluida sao pulados; o manifest e escrito
incrementalmente. Use --dry-run para conferir os 400 jobs antes de disparar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
SCRIPTED = {"keep", "random", "random-seconds"}


def scripted_jobs(entry: dict, output_dir: Path) -> list[tuple[str, list[str], Path]]:
    jobs = []
    for i in range(entry["runs"]):
        seed = entry["seed_start"] + i
        run_dir = output_dir / f"{entry['name']}_seed{seed}"
        cmd = [
            str(PYTHON), "examples/collect_run.py",
            "--config", "src/environments/scenario_configurations/paper_es_use_case.json",
            "--ns3_path", str(REPO.parent / "ns-3-mmwave-oran"),
            "--output_folder", str(run_dir),
            "--policy", entry["policy"],
            "--seed", str(seed),
        ]
        jobs.append((f"{entry['name']}#{i}", cmd, run_dir))
    return jobs


def heuristic_batches(entry: dict, output_dir: Path) -> list[tuple[str, list[str], Path]]:
    seeds = [str(entry["seed_start"] + i) for i in range(entry["runs"])]
    tuples = [f"{a},{b},{c},{d}" for a, b, c, d in entry.get("tuples", [])]
    batch_dir = output_dir / entry["name"]
    cmd = [
        str(PYTHON), "scripts/run_paper_baselines.py",
        "--config", "src/environments/scenario_configurations/paper_es_use_case.json",
        "--ns3-path", str(REPO.parent / "ns-3-mmwave-oran"),
        "--output-dir", str(batch_dir),
        "--seeds", *seeds,
        "--baselines", entry["name"],
        "--tuples", *tuples,
        "--jobs", "6",
        "--skip-build",
    ]
    return [(f"{entry['name']} ({len(seeds)} runs)", cmd, batch_dir)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="../experiment-configs/dataset_plan.json")
    parser.add_argument("--output-dir", default="examples/output/dataset_runs")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "campaign_manifest.jsonl"

    jobs: list[tuple[str, list[str], Path]] = []
    for entry in plan["behavior_policies"]:
        if entry["policy"] in SCRIPTED:
            jobs += scripted_jobs(entry, output_dir)
        else:
            jobs += heuristic_batches(entry, output_dir)

    total_runs = plan["runs_total"]
    print(f"{len(jobs)} jobs despachaveis para {total_runs} runs "
          f"(~{plan['expected_transitions']} transicoes esperadas)")
    for name, cmd, out in jobs:
        print(f"  - {name} -> {out}")
        if args.dry_run:
            print("      " + " ".join(cmd))
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.touch()

    def run_one(item):
        name, cmd, out = item
        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO)
        return name, out, res.returncode, round(time.time() - t0, 1)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            name, out, code, wall = fut.result()
            record = {"job": name, "output": str(out), "exit_code": code,
                      "wall_s": wall, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(manifest_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            if code != 0:
                failures += 1
            print(f"{name}: exit={code} wall={wall}s")

    print(f"campanha concluida: {len(jobs) - failures}/{len(jobs)} jobs ok "
          f"(manifest: {manifest_path})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
