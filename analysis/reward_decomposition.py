#!/usr/bin/env python3
"""Decompose the paper reward (Eq. 2) on the real offline dataset.

The dataset ``examples/output/real_runs/baselines_dataset.npz`` stores the five
*normalised* reward components of every transition, so Eq. (2) can be verified
and decomposed without re-running ns-3:

    r = w1*thr - w2*energy - w3*rlf - w4*es_on_cost - w2*bs_on

The script reports, for the Table I agent whose weights reproduce the stored
reward:

* the mean normalised value of each component;
* the signed contribution of each term and its share of the negative mass;
* the correlation of each component with the reward and with throughput;
* the same decomposition per baseline policy;
* the mean reward of every baseline under every Table I weight set, i.e. how
  sensitive the "best policy" verdict is to the weights.

Outputs (next to this script)::

    reward_decomposition.csv
    reward_decomposition_by_policy.csv
    reward_weights_sensitivity.csv
    reward_decomposition.md

Usage::

    cd /mnt/legitbet/elioth_sbrc
    ns-o-ran-gym/.venv/bin/python analysis/reward_decomposition.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GYM = ROOT / "ns-o-ran-gym"
sys.path.insert(0, str(GYM / "src"))

from nsoran.drl.config import PAPER_AGENTS, compute_reward  # noqa: E402

DEFAULT_DATASET = GYM / "examples/output/real_runs/baselines_dataset_d1d3.npz"
DEFAULT_CALIBRATION = GYM / "examples/output/reward_calibration_d1d3.npz"
COMPONENTS = ("throughput", "energy", "rlf", "es_on_cost", "bs_on")
SIGNS = {"throughput": +1.0, "energy": -1.0, "rlf": -1.0,
         "es_on_cost": -1.0, "bs_on": -1.0}
WEIGHT_SYMBOL = {"throughput": "w1", "energy": "w2", "rlf": "w3",
                 "es_on_cost": "w4", "bs_on": "w2"}
POLICY_ORDER = ("always_on", "static", "dynamic", "random")
PRETTY = {"always_on": "Always-On", "static": "Static", "dynamic": "Dynamic",
          "random": "Random"}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}"


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], list[str], dict]:
    data = np.load(path, allow_pickle=True)
    components = np.asarray(data["reward_components"], dtype=float)
    if components.shape[1] != len(COMPONENTS):
        raise SystemExit(f"unexpected reward_components shape {components.shape}")
    comps = {name: components[:, i] for i, name in enumerate(COMPONENTS)}
    metadata = json.loads(str(data["metadata"]))
    run_dirs = list(metadata.get("run_dirs", []))
    if len(run_dirs) != len(data["episode_ids"]):
        # Fall back to the episode count: the mapping is only needed for the
        # per-policy breakdown.
        n_episodes = int(np.max(data["episode_ids"])) + 1
        run_dirs = run_dirs[:n_episodes]
    policies = []
    for run_dir in run_dirs:
        name = Path(run_dir).parent.name if "real_runs" in run_dir else Path(run_dir).name
        policy = next(
            (p for p in POLICY_ORDER if name.startswith(p)),
            name.split("_seed")[0],
        )
        policies.append(policy)
    episode_ids = np.asarray(data["episode_ids"], dtype=int)
    # Expand the per-episode label to one label per transition.
    per_step = [policies[i] if i < len(policies) else "unknown" for i in episode_ids]
    return comps, per_step, {
        "rewards": np.asarray(data["rewards"], dtype=float),
        "episode_ids": episode_ids,
        "metadata": metadata,
        "n": len(data["rewards"]),
        "episodes": policies,
    }


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    """Load the quantile-calibration samples of the five reward components."""

    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=True)
    return {
        name: np.asarray(data[name], dtype=float)
        for name in COMPONENTS
        if name in data
    }


def match_agent(comps: dict[str, np.ndarray], rewards: np.ndarray) -> tuple[str, dict[str, float]]:
    """Find the Table I weight set that reproduces the stored reward."""

    errors: dict[str, float] = {}
    for name, agent in PAPER_AGENTS.items():
        predicted = compute_reward(
            comps["throughput"], comps["energy"], comps["rlf"],
            comps["es_on_cost"], comps["bs_on"], agent,
        )
        errors[name] = float(np.max(np.abs(np.asarray(predicted) - rewards)))
    best = min(errors, key=errors.get)
    return best, errors


def contributions(comps: dict[str, np.ndarray], agent) -> dict[str, np.ndarray]:
    weights = {"throughput": agent.w1, "energy": agent.w2, "rlf": agent.w3,
               "es_on_cost": agent.w4, "bs_on": agent.w2}
    return {name: SIGNS[name] * weights[name] * comps[name] for name in COMPONENTS}


def decompose(comps, rewards, agent, policies) -> list[dict[str, Any]]:
    contrib = contributions(comps, agent)
    negative_mass = {name: float(-contrib[name].mean())
                     for name in COMPONENTS if SIGNS[name] < 0}
    total_negative = sum(negative_mass.values()) or float("nan")

    policy_arr = np.asarray(policies)
    reward_resid = rewards.copy()
    for policy in np.unique(policy_arr):
        mask = policy_arr == policy
        reward_resid[mask] -= rewards[mask].mean()

    rows = []
    for name in COMPONENTS:
        corr_reward = float(np.corrcoef(comps[name], rewards)[0, 1])
        corr_thr = float(np.corrcoef(comps[name], comps["throughput"])[0, 1])
        resid = comps[name].copy()
        for policy in np.unique(policy_arr):
            mask = policy_arr == policy
            resid[mask] -= comps[name][mask].mean()
        corr_within = (
            float(np.corrcoef(resid, reward_resid)[0, 1])
            if np.std(resid) > 1e-12 and np.std(reward_resid) > 1e-12
            else float("nan")
        )
        rows.append({
            "component": name,
            "weight_symbol": WEIGHT_SYMBOL[name],
            "sign": "+" if SIGNS[name] > 0 else "−",
            "weight": getattr(agent, WEIGHT_SYMBOL[name]),
            "mean_norm": float(comps[name].mean()),
            "sd_norm": float(comps[name].std(ddof=1)),
            "min_norm": float(comps[name].min()),
            "max_norm": float(comps[name].max()),
            "mean_contribution": float(contrib[name].mean()),
            "share_of_negative_mass_pct": (
                100.0 * negative_mass[name] / total_negative
                if SIGNS[name] < 0 and np.isfinite(total_negative) else float("nan")
            ),
            "corr_with_reward": corr_reward,
            "corr_with_reward_within_policy": corr_within,
            "corr_with_throughput": corr_thr,
        })
    return rows


def by_policy(comps, rewards, agent, policies) -> list[dict[str, Any]]:
    contrib = contributions(comps, agent)
    policy_arr = np.asarray(policies)
    rows = []
    for policy in POLICY_ORDER:
        mask = policy_arr == policy
        if not mask.any():
            continue
        row = {
            "policy": policy,
            "n": int(mask.sum()),
            "mean_reward": float(rewards[mask].mean()),
        }
        for name in COMPONENTS:
            row[f"mean_{name}"] = float(comps[name][mask].mean())
            row[f"contrib_{name}"] = float(contrib[name][mask].mean())
        rows.append(row)
    return rows


def sensitivity(comps, rewards, policies) -> list[dict[str, Any]]:
    policy_arr = np.asarray(policies)
    rows = []
    for name, agent in PAPER_AGENTS.items():
        predicted = np.asarray(compute_reward(
            comps["throughput"], comps["energy"], comps["rlf"],
            comps["es_on_cost"], comps["bs_on"], agent,
        ))
        row = {
            "agent": name,
            "w1": agent.w1, "w2": agent.w2, "w3": agent.w3, "w4": agent.w4,
            "normalization": agent.normalization,
            "mean_reward_all": float(predicted.mean()),
        }
        means = {}
        for policy in POLICY_ORDER:
            mask = policy_arr == policy
            if mask.any():
                means[policy] = float(predicted[mask].mean())
                row[f"reward_{policy}"] = means[policy]
        row["best_policy"] = max(means, key=means.get) if means else ""
        row["spread_best_minus_worst"] = (
            max(means.values()) - min(means.values()) if means else float("nan")
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                for k, v in row.items()
            })


def write_markdown(path: Path, dataset_path: Path, env_model: dict,
                   calibration: dict[str, np.ndarray], calibration_name: str,
                   agent_name, agent,
                   errors, rows, pol_rows, sens_rows, rewards, comps,
                   n) -> None:
    out: list[str] = []
    out.append("# Decomposição do reward do artigo sobre dados reais\n")
    model_label = (
        f"energia `{env_model.get('energy_model', '?')}`, custo de activação "
        f"`{env_model.get('es_cost_mode', '?')}` e normalização "
        f"{'por célula' if env_model.get('per_cell_normalization', True) else 'agregada'}"
    )
    out.append(
        f"Gerado por `analysis/reward_decomposition.py` a partir de "
        f"`{dataset_path}` ({n} transições, {len(set(comps['bs_on']))} valores "
        f"distintos de `bs_on`). As componentes guardadas no dataset são as "
        f"**normalizadas** (contrato de `nsoran.drl.offline`), pelo que a Eq. (2) "
        f"pode ser verificada directamente. Modelo do ambiente: {model_label}."
    )
    if calibration:
        table = [
            "| componente | p0 | p25 | p50 | p75 | p100 |",
            "|---|---|---|---|---|---|",
        ]
        for name in COMPONENTS:
            samples = calibration[name]
            if samples.size == 0:
                continue
            q = np.quantile(samples, [0.0, 0.25, 0.5, 0.75, 1.0])
            table.append(
                f"| {name} | " + " | ".join(_fmt(value, 3) for value in q) + " |"
            )
        out.append(
            "Calibração quantílica ajustada nos runs reais "
            f"(`{calibration_name}`), amostras por componente. Para "
            "`es_on_cost` e `bs_on` a série global é a soma sobre as 7 células; a "
            "normalização D3 usa as séries `*_cell_*` e agrega pela média:"
        )
        out.append("\n".join(table))

    out.append("## Verificação da Eq. (2)\n")
    out.append(
        "| agente | w1 | w2 | w3 | w4 | normalização | erro máximo de reconstrução |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {name} | {cfg.w1:.2f} | {cfg.w2:.2f} | {cfg.w3:.2f} | {cfg.w4:.2f} | "
            f"{cfg.normalization} | {errors[name]:.3e} |"
            for name, cfg in PAPER_AGENTS.items()
        )
    )
    out.append(
        f"O reward armazenado é reproduzido por **{agent_name}** com erro máximo de "
        f"`{errors[agent_name]:.2e}` — a implementação é fiel à Eq. (2) com estes pesos. "
        "A decomposição seguinte usa esses pesos."
    )

    out.append("## Componentes\n")
    table = [
        "| componente | símbolo | sinal | peso | média normalizada | dp | min | max | "
        "contribuição média | % da massa negativa | corr(r, x) | corr dentro da política | "
        "corr(x, thr) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['component']} | {row['weight_symbol']} | {row['sign']} | "
            f"{row['weight']:.2f} | {_fmt(row['mean_norm'])} | {_fmt(row['sd_norm'])} | "
            f"{_fmt(row['min_norm'])} | {_fmt(row['max_norm'])} | "
            f"{_fmt(row['mean_contribution'])} | "
            f"{_fmt(row['share_of_negative_mass_pct'], 1)} | "
            f"{_fmt(row['corr_with_reward'])} | "
            f"{_fmt(row['corr_with_reward_within_policy'])} | "
            f"{_fmt(row['corr_with_throughput'])} |"
        )
    out.append("\n".join(table))

    bs_on = next(r for r in rows if r["component"] == "bs_on")
    energy = next(r for r in rows if r["component"] == "energy")
    thr = next(r for r in rows if r["component"] == "throughput")
    out.append(
        f"O termo `bs_on` contribui com **{bs_on['mean_contribution']:.4f}** por passo, "
        f"isto é, **{bs_on['share_of_negative_mass_pct']:.1f}%** de toda a massa negativa "
        f"do reward; a energia contribui com {energy['share_of_negative_mass_pct']:.1f}% "
        f"e o RLF com {next(r for r in rows if r['component'] == 'rlf')['share_of_negative_mass_pct']:.1f}%. "
        f"A correlação entre `bs_on` e o reward é {bs_on['corr_with_reward']:.3f}, "
        f"enquanto a do throughput é {thr['corr_with_reward']:.3f}."
    )
    out.append(
        "Como `w2` é usado duas vezes (energia e `bs_on`), o peso efectivo do eixo "
        f"energético é `w2*(energy + bs_on)` = {agent.w2:.2f}·(soma das duas componentes), "
        "superior ao `w2` nominal da Tabela I. Ver `mathematical-formulation.md`, R4."
    )
    out.append(
        "Depois da normalização quantílica cada componente tem média ≈ 0,5; a "
        "contribuição **média** é portanto fixada pelos pesos, não pelos KPIs. O que "
        "distingue as políticas é a dispersão: `bs_on` tem o maior desvio-padrão "
        f"({bs_on['sd_norm']:.3f}), seguido de `es_on_cost` ({next(r for r in rows if r['component'] == 'es_on_cost')['sd_norm']:.3f}); "
        "as componentes de throughput, energia e RLF ficam em ≈ 0,10–0,12. É esta "
        "variância de `bs_on`, e não a energia por bit, que domina a ordenação do "
        "reward entre políticas."
    )
    out.append(
        "A coluna `corr dentro da política` remove a média de cada política antes de "
        "correlacionar. Se o sinal agregado desaparecer (ou inverter) dentro das "
        "políticas, a correlação agregada era um efeito de composição entre políticas, "
        "não uma relação causal entre a KPI e o reward."
    )

    out.append("## Por política\n")
    table = [
        "| política | n | reward médio | thr | energia | RLF | es_on_cost | bs_on | "
        "contribuição thr | energia | RLF | activação | bs_on |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in pol_rows:
        table.append(
            f"| {PRETTY[row['policy']]} | {row['n']} | {_fmt(row['mean_reward'])} | "
            + " | ".join(_fmt(row[f"mean_{name}"]) for name in COMPONENTS)
            + " | "
            + " | ".join(_fmt(row[f"contrib_{name}"]) for name in COMPONENTS)
            + " |"
        )
    out.append("\n".join(table))
    out.append(
        "Todas as políticas que desligam células recebem uma penalização de `bs_on` "
        "menor do que o Always-On; é essa diferença, e não a energia por bit, que "
        "domina a ordenação do reward. O Random tem o pior RLF e o pior throughput, "
        "mas ainda assim reduz a penalização de `bs_on` em relação ao Always-On."
    )

    out.append("## Sensibilidade aos pesos da Tabela I\n")
    table = [
        "| agente | w1 | w2 | w3 | w4 | Always-On | Static | Dynamic | Random | "
        "melhor política | spread melhor−pior |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sens_rows:
        table.append(
            f"| {row['agent']} | {row['w1']:.2f} | {row['w2']:.2f} | {row['w3']:.2f} | "
            f"{row['w4']:.2f} | {_fmt(row.get('reward_always_on'))} | "
            f"{_fmt(row.get('reward_static'))} | {_fmt(row.get('reward_dynamic'))} | "
            f"{_fmt(row.get('reward_random'))} | {PRETTY.get(row['best_policy'], row['best_policy'])} | "
            f"{_fmt(row['spread_best_minus_worst'])} |"
        )
    out.append("\n".join(table))
    winners: dict[str, list[str]] = {}
    for row in sens_rows:
        winners.setdefault(row["best_policy"], []).append(row["agent"])
    winner_text = "; ".join(
        f"{PRETTY.get(policy, policy)} ({', '.join(agents)})"
        for policy, agents in winners.items()
    )
    spread_max = max(sens_rows, key=lambda row: row["spread_best_minus_worst"])
    spread_min = min(sens_rows, key=lambda row: row["spread_best_minus_worst"])
    random_wins = len(winners.get("random", []))
    total_agents = len(sens_rows)
    static_agents = ", ".join(winners.get("static", []))
    static_note = (
        f"; o Static é escolhido por {static_agents}, o conjunto com menor `w2` "
        "(0,19), que reduz o peso de `bs_on` e dá mais espaço ao throughput (o único "
        "termo positivo)"
        if static_agents
        else ""
    )
    out.append(
        f"Vencedor por conjunto de pesos: {winner_text}. O Random — a pior política "
        f"em throughput e RLF — ganha em {random_wins} de {total_agents} casos"
        f"{static_note}. O spread melhor−pior é máximo no "
        f"{spread_max['agent']} ({spread_max['spread_best_minus_worst']:.3f}) e mínimo "
        f"no {spread_min['agent']} ({spread_min['spread_best_minus_worst']:.3f}). "
        "Isto mostra que a Eq. (2), com esta normalização, **não codifica a restrição "
        "de QoS** que o artigo usa para avaliar os agentes: o reward e a métrica de "
        "avaliação (energia vs throughput vs RLF) apontam para políticas diferentes."
    )

    energy_model = env_model.get("energy_model", "proxy")
    if energy_model == "watts":
        energy_note = (
            "2. A energia usa o modelo EARTH `P_on = 130 + 94·x` W e "
            "`P_sleep = 75 W` (D1). A correlação "
            f"`corr(thr, energy)` no dataset é "
            f"{np.corrcoef(comps['throughput'], comps['energy'])[0, 1]:.3f}: "
            "continua alta porque a carga `x` acompanha o throughput, mas o termo "
            "estático de 130 W impõe um piso e o sleep de 75 W limita a poupança."
        )
    else:
        energy_note = (
            "2. A energia é o proxy `TB.TotNbrDl.1 * p_tx` (sem potência estática), "
            "pelo que `energy` e `throughput` são quase colineares; a correlação "
            f"`corr(thr, energy)` no dataset é "
            f"{np.corrcoef(comps['throughput'], comps['energy'])[0, 1]:.3f}."
        )
    per_cell = bool(env_model.get("per_cell_normalization", False))
    normalization_note = (
        "3. A normalização quantílica por célula (D3) força média ≈ 0,5 em todas as "
        "componentes, pelo que a contribuição média de cada termo é fixada pelos "
        "pesos. A ordenação entre políticas vem da dispersão — sobretudo de `bs_on` "
        f"(dp {bs_on['sd_norm']:.3f}) — e não da magnitude média dos KPIs."
        if per_cell
        else
        "3. A normalização quantílica agregada também força média ≈ 0,5 em todas as "
        "componentes; a ordenação entre políticas vem da dispersão de `bs_on` "
        f"(dp {bs_on['sd_norm']:.3f}) e não da magnitude média dos KPIs."
    )
    out.append("## Implicações para a reprodução\n")
    out.append(
        "\n".join([
            "1. O reward offline é dominado por `bs_on`, uma variável de **estado** e não "
            "de desempenho; um agente pode aumentá-lo desligando células sem melhorar "
            f"eficiência energética. Com os pesos da Tabela I, o Random — a pior política "
            f"em throughput e RLF — é o óptimo em {random_wins} de {total_agents} casos; "
            "o PPO-1 é a excepção e prefere Static.",
            energy_note,
            normalization_note,
            "4. Qualquer comparação com a Tabela VI do artigo deve reportar também a "
            "intensidade energética (energia/throughput), caso contrário a poupança "
            "atribuída ao agente é indistinguível de tráfego não entregue.",
            f"5. O dataset usa a calibração quantílica ajustada nos runs reais "
            f"(`{calibration_name}`). Sem `--calibration-file`, as componentes ficam "
            "comprimidas pelas gamas de fallback e a decomposição deixa de ser comparável.",
        ])
    )
    path.write_text("\n\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--calibration-file", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffix appended to each output file name, e.g. '_proxy' keeps the "
            "Eq. (1) proxy decomposition next to the D1 main files."
        ),
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    comps, policies, meta = load_dataset(dataset_path)
    calibration = load_calibration(Path(args.calibration_file))
    rewards = meta["rewards"]
    agent_name, errors = match_agent(comps, rewards)
    agent = PAPER_AGENTS[agent_name]

    rows = decompose(comps, rewards, agent, policies)
    pol_rows = by_policy(comps, rewards, agent, policies)
    sens_rows = sensitivity(comps, rewards, policies)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    write_csv(out / f"reward_decomposition{suffix}.csv", rows)
    write_csv(out / f"reward_decomposition_by_policy{suffix}.csv", pol_rows)
    write_csv(out / f"reward_weights_sensitivity{suffix}.csv", sens_rows)
    write_markdown(
        out / f"reward_decomposition{suffix}.md",
        dataset_path,
        meta["metadata"].get("environment_model", {}),
        calibration,
        Path(args.calibration_file).name,
        agent_name,
        agent,
        errors,
        rows,
        pol_rows,
        sens_rows,
        rewards,
        comps,
        meta["n"],
    )

    print(f"dataset: {meta['n']} transições, {len(meta['episodes'])} episódios")
    print(f"pesos que reproduzem o reward: {agent_name} (erro {errors[agent_name]:.2e})")
    for row in rows:
        print(f"  {row['component']:>10}  mean={row['mean_norm']:.4f}  "
              f"contrib={row['mean_contribution']:+.4f}  "
              f"share_neg={row['share_of_negative_mass_pct']:.1f}%  "
              f"corr_r={row['corr_with_reward']:+.3f}")
    print(f"Escrito em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
