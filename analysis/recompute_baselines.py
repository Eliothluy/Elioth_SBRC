#!/usr/bin/env python3
"""Recompute the paper baselines from the real ns-3 run folders.

The script does **not** read any pre-aggregated table: it replays every run
folder through :func:`examples.compare_runs.run_metrics`, which rebuilds the
85-dimensional state with the same code path as the live environment and
returns the raw per-indication totals.  The output therefore has a single
provenance chain:

    ns-3 run folder (database.db, bsState.txt)
        -> compare_runs.run_metrics(window=99)
        -> analysis/baseline_per_run.csv
        -> analysis/baseline_summary.csv / .md
        -> analysis/baseline_by_tuple.csv

Runs aborted by ns-3 are reported, flagged ``comparable=no`` and excluded from
the primary aggregate.  They are still shown so that the loss is visible.

Usage::

    cd /mnt/legitbet/elioth_sbrc
    ns-o-ran-gym/.venv/bin/python analysis/recompute_baselines.py

Outputs are written next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GYM = ROOT / "ns-o-ran-gym"
sys.path.insert(0, str(GYM / "src"))
sys.path.insert(0, str(GYM / "examples"))

from _model_cli import (  # noqa: E402
    add_environment_model_arguments,
    model_from_args,
)
from compare_runs import run_metrics  # noqa: E402
from nsoran.drl.real_data import load_scenario_config  # noqa: E402

DEFAULT_RUNS = GYM / "examples/output/real_runs"
DEFAULT_CONFIG = GYM / "src/environments/scenario_configurations/paper_es_use_case.json"
PAPER_INDICATIONS = 99  # 10 s / 100 ms, minus the first report
POLICY_ORDER = ("always_on", "static", "dynamic", "random")


# ---------------------------------------------------------------------------
# Statistics without a hard scipy dependency
# ---------------------------------------------------------------------------
def _t_critical(n: int) -> float:
    """Two-sided 95% Student-t critical value for ``n - 1`` degrees of freedom."""

    if n < 2:
        return float("nan")
    try:  # scipy is installed in the project venv
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:  # pragma: no cover - fallback for a bare interpreter
        table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                 8: 2.365, 9: 2.306, 10: 2.262}
        return table.get(n, 1.96)


def summarise(values: list[float]) -> dict[str, float]:
    """Mean, sample standard deviation and 95% confidence interval."""

    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "ci95": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = float(_t_critical(n) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95": ci,
            "min": float(arr.min()), "max": float(arr.max())}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}"


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------
def collect_runs(
    runs_dir: Path, config: dict, model: Any = None
) -> list[dict[str, Any]]:
    manifest_path = runs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []

    for job in manifest["jobs"]:
        run_dirs = [Path(p) for p in job.get("run_dirs") or []]
        resolved: list[Path] = []
        for path in run_dirs:
            if path.is_absolute():
                resolved.append(path)
            else:
                # The manifest stores paths relative to the Gym repo
                # (``examples/output/real_runs/...``); fall back to the
                # workspace root for older manifests.
                for base in (GYM, ROOT):
                    candidate = base / path
                    if (candidate / "database.db").is_file():
                        resolved.append(candidate)
                        break
        run_dirs = [p for p in resolved if (p / "database.db").is_file()]
        status = job.get("status", "unknown")
        manifest_run_dir = Path(job.get("run_dir", ""))
        if manifest_run_dir and not manifest_run_dir.is_absolute():
            manifest_run_dir = GYM / manifest_run_dir
        row: dict[str, Any] = {
            "run_name": job["name"],
            "policy": job["policy"],
            "tuple": "-".join(str(v) for v in job["tuple"]) if job.get("tuple") else "",
            "seed": job["seed"],
            "status": status,
            "duration_s": job.get("duration_s"),
            "sim_time_covered_s": job.get("sim_time_covered_s"),
            "run_dir": str(manifest_run_dir),
        }
        if not run_dirs:
            row.update({"indications": 0, "comparable": "no", "note": "run folder missing"})
            rows.append(row)
            continue

        metrics = run_metrics(run_dirs, config, window=None, model=model)
        row.update({
            "indications": metrics["indications"],
            "throughput_mean": metrics["throughput_mean"],
            "energy_mean": metrics["energy_mean"],
            "rlf_mean": metrics["rlf_mean"],
            "rlf_max": metrics["rlf_max"],
            "bs_on_mean": metrics["bs_on_mean"],
            "activation_mean": metrics["activation_mean"],
            "pdcp_latency_median": metrics.get("pdcp_latency_median"),
            "pdcp_latency_p95": metrics.get("pdcp_latency_p95"),
        })
        complete = metrics["indications"] >= PAPER_INDICATIONS and status == "ok"
        row["comparable"] = "yes" if complete else "no"
        if not complete:
            row["note"] = (
                f"crashed at {row['sim_time_covered_s']} s, "
                f"{metrics['indications']} of {PAPER_INDICATIONS} indications"
            )
        else:
            row["note"] = "complete 10 s scenario"
        # Energy per unit of delivered throughput: the paper's energy saving is
        # only meaningful next to the traffic that was actually carried.
        row["energy_per_throughput"] = (
            metrics["energy_mean"] / metrics["throughput_mean"]
            if metrics["throughput_mean"] else float("nan")
        )
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    by_policy: list[dict[str, Any]] = []
    by_tuple: list[dict[str, Any]] = []

    always_on = [r for r in rows if r["policy"] == "always_on" and r["comparable"] == "yes"]
    ref_thr = summarise([r["throughput_mean"] for r in always_on])["mean"]
    ref_energy = summarise([r["energy_mean"] for r in always_on])["mean"]
    ref_intensity = ref_energy / ref_thr if ref_thr else float("nan")
    ref_lat_p95 = summarise(
        [r["pdcp_latency_p95"] for r in always_on if r.get("pdcp_latency_p95") is not None]
    )["mean"]

    for policy in POLICY_ORDER:
        subset = [r for r in rows if r["policy"] == policy and r["comparable"] == "yes"]
        if not subset and policy != "always_on":
            continue
        thr = summarise([r["throughput_mean"] for r in subset])
        energy = summarise([r["energy_mean"] for r in subset])
        rlf = summarise([r["rlf_mean"] for r in subset])
        rlf_max = summarise([r["rlf_max"] for r in subset])
        bs_on = summarise([r["bs_on_mean"] for r in subset])
        intensity_mean = energy["mean"] / thr["mean"] if thr["mean"] else float("nan")
        by_policy.append({
            "policy": policy,
            "runs": len(subset),
            "indications": PAPER_INDICATIONS,
            "throughput_mean": thr["mean"],
            "throughput_sd": thr["sd"],
            "throughput_ci95": thr["ci95"],
            "energy_mean": energy["mean"],
            "energy_sd": energy["sd"],
            "energy_ci95": energy["ci95"],
            "rlf_mean": rlf["mean"],
            "rlf_max": rlf_max["max"],
            "bs_on_mean": bs_on["mean"],
            "pdcp_latency_median": summarise(
                [r["pdcp_latency_median"] for r in subset if r.get("pdcp_latency_median") is not None]
            )["mean"],
            "pdcp_latency_p95": summarise(
                [r["pdcp_latency_p95"] for r in subset if r.get("pdcp_latency_p95") is not None]
            )["mean"],
            "energy_per_throughput": intensity_mean,
            "throughput_delta_pct": 100.0 * (thr["mean"] - ref_thr) / ref_thr if ref_thr else float("nan"),
            "energy_delta_pct": 100.0 * (energy["mean"] - ref_energy) / ref_energy if ref_energy else float("nan"),
            "intensity_ratio_vs_always_on": intensity_mean / ref_intensity if ref_intensity else float("nan"),
            "pdcp_latency_p95_ratio_vs_always_on": (
                summarise([r["pdcp_latency_p95"] for r in subset if r.get("pdcp_latency_p95") is not None])["mean"] / ref_lat_p95
                if ref_lat_p95 else float("nan")
            ),
            "seeds": ",".join(str(r["seed"]) for r in sorted(subset, key=lambda x: x["seed"])),
            "tuples": ",".join(sorted({r["tuple"] for r in subset if r["tuple"]})) or "—",
        })

    for policy in POLICY_ORDER:
        subset = [r for r in rows if r["policy"] == policy and r["comparable"] == "yes"]
        tuples = sorted({r["tuple"] for r in subset if r["tuple"]})
        for tup in tuples:
            group = [r for r in subset if r["tuple"] == tup]
            thr = summarise([r["throughput_mean"] for r in group])
            energy = summarise([r["energy_mean"] for r in group])
            intensity_mean = energy["mean"] / thr["mean"] if thr["mean"] else float("nan")
            by_tuple.append({
                "policy": policy,
                "tuple": tup,
                "runs": len(group),
                "throughput_mean": thr["mean"],
                "energy_mean": energy["mean"],
                "rlf_mean": summarise([r["rlf_mean"] for r in group])["mean"],
                "bs_on_mean": summarise([r["bs_on_mean"] for r in group])["mean"],
                "pdcp_latency_p95": summarise(
                    [r["pdcp_latency_p95"] for r in group if r.get("pdcp_latency_p95") is not None]
                )["mean"],
                "energy_per_throughput": intensity_mean,
                "throughput_delta_pct": 100.0 * (thr["mean"] - ref_thr) / ref_thr if ref_thr else float("nan"),
                "energy_delta_pct": 100.0 * (energy["mean"] - ref_energy) / ref_energy if ref_energy else float("nan"),
                "intensity_ratio_vs_always_on": intensity_mean / ref_intensity if ref_intensity else float("nan"),
            })

    return by_policy, by_tuple


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
PER_RUN_FIELDS = [
    "run_name", "policy", "tuple", "seed", "status", "duration_s",
    "sim_time_covered_s", "indications", "throughput_mean", "energy_mean",
    "rlf_mean", "rlf_max", "bs_on_mean", "activation_mean",
    "pdcp_latency_median", "pdcp_latency_p95",
    "energy_per_throughput", "comparable", "note", "run_dir",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def write_markdown(
    path: Path,
    rows,
    by_policy,
    by_tuple,
    manifest_note: str,
    model_note: str = "",
) -> None:
    out: list[str] = []
    out.append("# Baselines recompostos a partir das execuções reais em ns-3")
    out.append(
        "Gerado por `analysis/recompute_baselines.py`. Cada linha vem de uma pasta de "
        "execução (`database.db` + `bsState.txt`) reprocessada por "
        "`examples/compare_runs.py::run_metrics` (janela integral; as estatísticas "
        "usam as 99 indicações do cenário de 10 s). Nenhum número desta tabela foi "
        "copiado de tabela agregada."
    )
    out.append(manifest_note)
    if model_note:
        out.append(model_note)

    out.append("## Por política")
    table = [
        "| política | runs | throughput (média ± IC95) | energia (média ± IC95) | RLF médio | "
        "RLF máx | BS ligadas | Δthr % | Δenergia % | intensidade E/thr vs Always-On | "
        "latência PDCP p95 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in by_policy:
        table.append(
            f"| {row['policy']} | {row['runs']} | "
            f"{_fmt(row['throughput_mean'], 0)} ± {_fmt(row['throughput_ci95'], 0)} | "
            f"{_fmt(row['energy_mean'], 1)} ± {_fmt(row['energy_ci95'], 1)} | "
            f"{_fmt(row['rlf_mean'], 2)} | {_fmt(row['rlf_max'], 0)} | "
            f"{_fmt(row['bs_on_mean'], 2)} | {_fmt(row['throughput_delta_pct'], 2)} | "
            f"{_fmt(row['energy_delta_pct'], 2)} | "
            f"{_fmt(row['intensity_ratio_vs_always_on'], 3)} | "
            f"{_fmt(row.get('pdcp_latency_p95'), 1)} |"
        )
    out.append("\n".join(table))
    out.append(
        "`intensidade E/thr` = energia média por unidade de throughput médio, "
        "normalizada pelo valor do Always-On. Valores > 1 significam que a política "
        "gasta **mais** energia por bit entregue: a poupança absoluta vem de transportar "
        "menos tráfego, não de operar a rede de forma mais eficiente."
    )
    if model_note.startswith("Modelo D1"):
        out.append(
            "Com o modelo D1 a célula ligada consome `130 + 94·x` W (x = PRB/139) "
            "e a desligada 75 W; desligar uma célula poupa no máximo 66,5% "
            "(carga máxima) e no mínimo 42,3% (carga zero). Por isso a poupança "
            "de energia é menor que a perda de throughput no Random, embora a "
            "intensidade E/thr ainda piore em todas as políticas. O artigo relata "
            "+0,7%/+4,3% de poupança para Static/Dynamic com −23,5%/−27,6% de "
            "throughput; esses valores não são reproduzíveis com a Eq. (1) literal "
            "nem com o modelo EARTH, e ficam registados como ambiguidade (ver "
            "`mathematical-formulation.md`)."
        )

    out.append("## Por tupla (Static/Dynamic)")
    out.append(
        "As tuplas do artigo têm efeitos muito diferentes; a média por política esconde "
        "essa dispersão."
    )
    table = [
        "| política | tupla (N_On,P1,N_On,P2,N_Off) | runs | throughput | energia | RLF | "
        "BS ligadas | Δthr % | Δenergia % | E/thr vs Always-On | p95 lat (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in by_tuple:
        table.append(
            f"| {row['policy']} | {row['tuple']} | {row['runs']} | "
            f"{_fmt(row['throughput_mean'], 0)} | {_fmt(row['energy_mean'], 1)} | "
            f"{_fmt(row['rlf_mean'], 2)} | {_fmt(row['bs_on_mean'], 2)} | "
            f"{_fmt(row['throughput_delta_pct'], 2)} | {_fmt(row['energy_delta_pct'], 2)} | "
            f"{_fmt(row['intensity_ratio_vs_always_on'], 3)} | "
            f"{_fmt(row.get('pdcp_latency_p95'), 1)} |"
        )
    out.append("\n".join(table))

    out.append("## Execuções abortadas")
    crashed = [r for r in rows if r["comparable"] == "no"]
    if not crashed:
        out.append("Nenhuma.")
    else:
        table = [
            "| run | política | tupla | seed | sim time | indicações | nota |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in crashed:
            table.append(
                f"| {row['run_name']} | {row['policy']} | {row.get('tuple') or '—'} | "
                f"{row['seed']} | {_fmt(row.get('sim_time_covered_s'), 3)} | "
                f"{row.get('indications', 0)} | {row.get('note', '')} |"
            )
        out.append("\n".join(table))
        out.append(
            "As execuções abortadas ficam fora da agregação principal. "
            "As execuções `dynamic_2_2_0_3_seed556` e `static_2_2_0_3_seed556` têm "
            "impressão digital idêntica (mesmo throughput, energia, RLF e bs_on): "
            "não são amostras independentes."
        )

    out.append("## Leitura")
    out.append(
        "\n".join([
            "- O Always-On tem enorme variância entre seeds (ver `baseline_per_run.csv`), "
            "o que fixa um piso de incerteza para qualquer comparação de políticas.",
            "- Nenhuma política preserva throughput; a ordenação por energia é quase a "
            "ordenação por throughput perdido. A coluna de intensidade mostra que a "
            "eficiência por bit não melhora de forma sistemática.",
            "- As tuplas 4-2-0-1 são as menos agressivas e as 2-2-0-3 as mais agressivas, "
            "como esperado pela definição (N_On, N_Off).",
            "- O custo de activação médio das políticas que desligam células é próximo do "
            "das Always-On porque a métrica `0.9**(0.01*T_on_ms)` recompensa tempo ligado, "
            "não transições OFF→ON (ver `mathematical-formulation.md`, R2).",
        ])
    )
    path.write_text("\n\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffix appended to each output file name, e.g. '_proxy' writes "
            "baseline_summary_proxy.md. Useful for keeping the Eq. (1) proxy "
            "tables next to the D1 main tables."
        ),
    )
    add_environment_model_arguments(parser)
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    config = load_scenario_config(Path(args.config))
    model = model_from_args(args)
    rows = collect_runs(runs_dir, config, model=model)
    by_policy, by_tuple = aggregate(rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    write_csv(out / f"baseline_per_run{suffix}.csv", rows, PER_RUN_FIELDS)
    write_csv(
        out / f"baseline_summary{suffix}.csv", by_policy, list(by_policy[0].keys())
    )
    write_csv(
        out / f"baseline_by_tuple{suffix}.csv", by_tuple, list(by_tuple[0].keys())
    )

    n_jobs = len(rows)
    n_ok = sum(1 for r in rows if r["comparable"] == "yes")
    n_crash = n_jobs - n_ok
    manifest_note = (
        f"Manifesto: `{runs_dir / 'manifest.json'}` — {n_jobs} jobs, "
        f"{n_ok} completos e comparáveis, {n_crash} abortados/excluídos."
    )
    if model.energy_model == "watts":
        power = model.summary()["power_model"]
        model_note = (
            "Modelo D1 (principal): potência EARTH macro por setor "
            f"(P_0={power['p0_w']} W, P_tx,max={power['p_max_w']} W, "
            f"Δp={power['delta_p']}, N={power['n_antennas']}, "
            f"P_sleep={power['p_sleep_w']} W; P_on=130+94x W), custo de ativação "
            f"`{model.es_cost_mode}` (D2) e normalização por célula (D3; afeta o "
            "reward, não a energia/throughput acima)."
        )
    else:
        model_note = (
            "Modelo proxy (ablação da Eq. (1)): energia = "
            f"`TB.TotNbrDl.1 * {model.p_tx}`, custo de ativação "
            f"`{model.es_cost_mode}` e normalização por célula (D3; afeta o "
            "reward, não a energia/throughput acima)."
        )
    write_markdown(
        out / f"baseline_summary{suffix}.md",
        rows,
        by_policy,
        by_tuple,
        manifest_note,
        model_note,
    )

    print(manifest_note)
    for row in by_policy:
        print(
            f"{row['policy']:>10}  n={row['runs']}  "
            f"thr={row['throughput_mean']:.1f}  energy={row['energy_mean']:.1f}  "
            f"dThr={row['throughput_delta_pct']:+.2f}%  "
            f"dE={row['energy_delta_pct']:+.2f}%  "
            f"E/thr={row['intensity_ratio_vs_always_on']:.3f}x"
        )
    print(f"\nEscrito em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
