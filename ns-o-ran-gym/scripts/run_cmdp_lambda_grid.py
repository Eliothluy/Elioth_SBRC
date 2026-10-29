#!/usr/bin/env python3
"""CMDP Lagrangian grid (Ajuste A3 do cmdp_design.md, Rev. 1).

Treina um agente da Tabela I para cada valor fixo de lambda (dual congelado com
--qos-lambda-lr 0) e cada seed de treino: cada (lambda, seed) e um ponto
candidato da fronteira de Pareto. Retomavel: jobs com training_summary.json
existente sao pulados.

Exemplo (grade principal):
    .venv/bin/python scripts/run_cmdp_lambda_grid.py \
        --dataset examples/output/real_runs/baselines_dataset_d1d3.npz \
        --lambdas 0 0.5 1 2 5 10 --seeds 0 1 2 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
AGENT_SCRIPT = {f"PPO-{i}": f"ppo_{i}.py" for i in range(1, 6)}
AGENT_SCRIPT["DQN"] = "dqn.py"


def build_command(args: argparse.Namespace, lam: float, seed: int) -> tuple[list[str], Path]:
    output_dir = Path(args.output_root) / f"{args.agent}_lambda{lam:g}_s{seed}"
    cmd = [
        str(PYTHON),
        f"examples/{AGENT_SCRIPT[args.agent]}",
        "--dataset", args.dataset,
        "--total-timesteps", str(args.total_timesteps),
        "--batch-size", str(args.batch_size),
        "--seed", str(seed),
        "--quiet",
        "--output-dir", str(output_dir),
        "--cmdp-mode", "lagrangian",
        # A3: dual congelado — lambda fixo por treino; A4: teto 10.
        "--qos-lambda-init", str(lam),
        "--qos-lambda-lr", str(args.lambda_lr),
        "--qos-lambda-max", str(args.lambda_max),
    ]
    if args.qos_throughput_min is not None:
        cmd += ["--qos-throughput-min", str(args.qos_throughput_min)]
    if args.qos_rlf_max is not None:
        cmd += ["--qos-rlf-max", str(args.qos_rlf_max)]
    return cmd, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="PPO-2", choices=sorted(AGENT_SCRIPT))
    parser.add_argument(
        "--dataset",
        default="examples/output/real_runs/baselines_dataset_d1d3.npz",
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.5, 1, 2, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=1_200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lambda-lr", type=float, default=0.0,
                        help="0 (default) freezes lambda: the A3 fixed grid.")
    parser.add_argument("--lambda-max", type=float, default=10.0, help="A4 cap.")
    parser.add_argument("--qos-throughput-min", type=float, default=53_680.0,
                        help="tau_SLA in raw ns-3 units (cmdp_design.md Sec. 2).")
    parser.add_argument("--qos-rlf-max", type=float, default=None)
    parser.add_argument("--output-root", default="examples/output/training/cmdp_grid")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = []
    for lam in args.lambdas:
        for seed in args.seeds:
            cmd, out = build_command(args, lam, seed)
            done = (out / "training_summary.json").exists()
            jobs.append((lam, seed, cmd, out, done))

    print(f"{len(jobs)} jobs ({sum(j[4] for j in jobs)} ja concluidos, serao pulados)")
    for lam, seed, cmd, out, done in jobs:
        tag = "SKIP" if done else "RUN "
        print(f"[{tag}] lambda={lam:g} seed={seed} -> {out}")
        if args.dry_run and not done:
            print("      " + " ".join(cmd))
    if args.dry_run:
        return 0

    def run_one(cmd: list[str]) -> tuple[subprocess.CompletedProcess, float]:
        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO)
        return res, round(time.time() - t0, 1)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for lam, seed, cmd, out, done in jobs:
            if done:
                continue
            futures[pool.submit(run_one, cmd)] = (lam, seed, out)
        for fut in as_completed(futures):
            lam, seed, out = futures[fut]
            res, wall = fut.result()
            rows.append({
                "agent": args.agent, "lambda": lam, "seed": seed,
                "exit_code": res.returncode,
                "wall_s": wall,
                "output_dir": str(out),
            })
            print(f"lambda={lam:g} seed={seed}: exit={res.returncode} wall={wall}s")

    csv_path = Path(args.output_root) / "grid_runs.csv"
    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"custo dos novos treinos: {csv_path}")
    failures = [r for r in rows if r["exit_code"] != 0]
    if failures:
        print(f"{len(failures)} treinos falharam", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
