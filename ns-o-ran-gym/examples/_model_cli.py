"""Shared CLI flags for the D1-D3 energy/reward model.

Only the standard library is imported at module load time, so callers such as
``collect_real_dataset.py`` and ``fit_reward_calibration.py`` keep working
before numpy is imported (the model dataclass is imported lazily inside
:func:`model_from_args`).
"""

from __future__ import annotations

import argparse
from typing import Any


def add_environment_model_arguments(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """Add the energy-model, activation-cost and normalisation flags."""

    group = parser.add_argument_group("D1-D3 environment model")
    group.add_argument(
        "--energy-model",
        choices=("watts", "proxy"),
        default="watts",
        help=(
            "Energy KPI: 'watts' is the EARTH macro model of Table 1 of "
            "computers-15-00050 (default); 'proxy' is the legacy TB*p_tx "
            "ablation."
        ),
    )
    group.add_argument(
        "--es-cost-mode",
        choices=("dense", "impulse"),
        default="dense",
        help=(
            "'dense' charges 0.9**(0.01*T_on_ms) every ON step (default); "
            "'impulse' charges only the OFF->ON transition."
        ),
    )
    group.add_argument(
        "--per-cell-normalization",
        dest="per_cell_normalization",
        action="store_true",
        default=True,
        help="Normalise each reward component per cell and aggregate the mean.",
    )
    group.add_argument(
        "--no-per-cell-normalization",
        dest="per_cell_normalization",
        action="store_false",
        help="Ablation: normalise the pooled component only.",
    )
    group.add_argument(
        "--per-cell-aggregation",
        choices=("mean", "sum"),
        default="mean",
        help="How per-cell normalised components are aggregated (default: mean).",
    )
    group.add_argument(
        "--p-tx",
        type=float,
        default=1.0,
        help="Legacy proxy transmission factor (only with --energy-model proxy).",
    )
    group.add_argument(
        "--prb-total", type=float, default=139.0, help="PRBs per indication."
    )
    group.add_argument("--p0-w", type=float, default=None, help="EARTH P_0 (W).")
    group.add_argument("--p-max-w", type=float, default=None, help="EARTH P_max (W).")
    group.add_argument("--delta-p", type=float, default=None, help="EARTH delta_p.")
    group.add_argument(
        "--p-sleep-w", type=float, default=None, help="EARTH sleep power (W)."
    )
    group.add_argument(
        "--n-antennas",
        type=float,
        default=None,
        help="EARTH antenna count (default 1, i.e. per sector).",
    )
    return group


def model_from_args(args: Any) -> Any:
    """Build an :class:`EnvironmentModelConfig` from parsed CLI arguments."""

    from nsoran.drl.real_data import EnvironmentModelConfig

    power_model = {}
    for key, value in (
        ("p0_w", getattr(args, "p0_w", None)),
        ("p_max_w", getattr(args, "p_max_w", None)),
        ("delta_p", getattr(args, "delta_p", None)),
        ("p_sleep_w", getattr(args, "p_sleep_w", None)),
        ("n_antennas", getattr(args, "n_antennas", None)),
    ):
        if value is not None:
            power_model[key] = float(value)
    return EnvironmentModelConfig(
        energy_model=getattr(args, "energy_model", "watts"),
        power_model=power_model or None,
        es_cost_mode=getattr(args, "es_cost_mode", "dense"),
        per_cell_normalization=getattr(args, "per_cell_normalization", True),
        per_cell_aggregation=getattr(args, "per_cell_aggregation", "mean"),
        p_tx=float(getattr(args, "p_tx", 1.0)),
        prb_total=float(getattr(args, "prb_total", 139.0)),
    )


__all__ = ["add_environment_model_arguments", "model_from_args"]
