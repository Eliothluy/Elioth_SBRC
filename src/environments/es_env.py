"""Gymnasium environment for the energy-saving use case of the paper.

The environment controls the seven NR cells ``[2, 3, 4, 5, 6, 7, 8]`` of the
``scenario-three`` ns-3 scenario.  Its contract follows the paper and
``PAPER_IMPLEMENTATION.md``:

* action space: ``Discrete(128)``.  Bit ``i`` of the integer action (LSB =
  cell 2) is ``1`` when the cell must be ON and ``0`` when it must be OFF.
* observation space: ``Box(0, 1, shape=(85,), dtype=float32)`` by default
  (``state_variant="paper85"``).  ``state_variant="compact"`` selects the
  70-dimensional E5 compact vector of ``docs/plano_tecnico/kpm_audit.md``
  (drops ``mac_pdu_64qam`` and ``prb_percentage`` per cell plus the global
  ``bs_on``); the default keeps the paper contract.
* state: 12 KPIs per cell (cell-major order) plus the global number of active
  cells (``bs_on``).
* reward::

      r =  w1 * throughput_norm
         - w2 * energy_norm
         - w3 * rlf_norm
         - w4 * es_on_cost_norm
         - w2 * bs_on_norm

The environment is intentionally usable with a fake datalake / fabricated
``pandas.DataFrame`` for unit tests: all state and reward maths lives in the
module level helpers below and does not need a running ns-3 process.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import numbers
import os
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from collections import deque
from gymnasium import spaces
from typing_extensions import override

from nsoran.drl.config import CmdpConfig, state_variant_indices
from nsoran.ns_env import NsOranEnv

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

CELL_IDS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
N_CELLS = len(CELL_IDS)

#: Order of the 12 per-cell features in the 85-dimensional state vector.
STATE_FEATURES_PER_CELL: tuple[str, ...] = (
    "throughput_dl",       # QosFlow.PdcpPduVolumeDL_Filter
    "energy",              # TB.TotNbrDl.1 * P_TX
    "ee_ratio",            # throughput / energy
    "rlf_count",           # UEs with L3 serving SINR < -5 dB
    "rlf_percentage",      # rlf_count / UEs_in_cell * 100
    "prb_used",            # RRU.PrbUsedDl
    "prb_percentage",      # RRU.PrbUsedDl / 139 * 100
    "tb_64qam",            # TB.TotNbrDlInitial.64Qam
    "tb_64qam_ratio",      # 64QAM / (QPSK + 16QAM + 64QAM)
    "mac_pdu_64qam",       # proxy: TB.TotNbrDlInitial.64Qam
    "es_on_cost",          # 0.9 ** (0.01 * T_on_ms)
    "cell_state",          # 1 = ON, 0 = OFF
)

GLOBAL_STATE_FEATURES: tuple[str, ...] = ("bs_on",)
STATE_DIM = N_CELLS * len(STATE_FEATURES_PER_CELL) + len(GLOBAL_STATE_FEATURES)  # 85

STATE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{feature}_{cell}" for cell in CELL_IDS for feature in STATE_FEATURES_PER_CELL
) + GLOBAL_STATE_FEATURES

#: Names of the reward components (keys used by :class:`RewardNormalizer`).
REWARD_COMPONENTS: tuple[str, ...] = ("throughput", "energy", "rlf", "es_on_cost", "bs_on")

#: Table I of the paper.
PAPER_REWARD_WEIGHTS: dict[str, dict[str, float | str]] = {
    "PPO-1": {"w1": 0.51, "w2": 0.19, "w3": 0.20, "w4": 0.10, "normalization": "normal"},
    "DQN": {"w1": 0.40, "w2": 0.40, "w3": 0.10, "w4": 0.10, "normalization": "uniform"},
    "PPO-2": {"w1": 0.40, "w2": 0.40, "w3": 0.10, "w4": 0.10, "normalization": "uniform"},
    "PPO-3": {"w1": 0.20, "w2": 0.40, "w3": 0.20, "w4": 0.20, "normalization": "uniform"},
    "PPO-4": {"w1": 0.40, "w2": 0.32, "w3": 0.18, "w4": 0.10, "normalization": "uniform"},
    "PPO-5": {"w1": 0.45, "w2": 0.20, "w3": 0.25, "w4": 0.10, "normalization": "normal"},
}

DEFAULT_REWARD_WEIGHTS = {
    "w1": PAPER_REWARD_WEIGHTS["PPO-1"]["w1"],
    "w2": PAPER_REWARD_WEIGHTS["PPO-1"]["w2"],
    "w3": PAPER_REWARD_WEIGHTS["PPO-1"]["w3"],
    "w4": PAPER_REWARD_WEIGHTS["PPO-1"]["w4"],
}

# The old implementation exposed a list of 64 "allowed" action indices.  It is
# kept as an attribute for source compatibility, but it no longer constrains
# the action space: the paper uses all 128 combinations.
LEGACY_ACTION_LIST = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    21, 22, 24, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 48,
    49, 50, 52, 56, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 80, 81, 82,
    84, 88, 96, 97, 98, 100, 104, 112,
]

#: KPIs read from the ns-O-RAN datalake (UE level; the datalake joins tables).
UE_KPMS = [
    "nrCellId",
    "QosFlow.PdcpPduVolumeDL_Filter",
    "TB.TotNbrDl.1",
    "L3 serving SINR",
    "RRU.PrbUsedDl",
    "TB.TotNbrDlInitial.64Qam",
    "TB.TotNbrDlInitial.Qpsk",
    "TB.TotNbrDlInitial.16Qam",
]
#: Same as :data:`UE_KPMS` without the CU-CP SINR column (fallback query).
DU_KPMS = [kpm for kpm in UE_KPMS if kpm != "L3 serving SINR"]

#: Default per-KPI ranges used by the state min-max normaliser.  They are
#: deliberately generous: clipping to ``[0, 1]`` is part of the contract, so an
#: out-of-range sample saturates instead of breaking the observation space.
_DEFAULT_KPI_RANGES: dict[str, tuple[float, float]] = {
    "throughput_dl": (0.0, 1.0e7),
    "energy": (0.0, 1.0e5),
    "ee_ratio": (0.0, 1.0e6),
    "rlf_count": (0.0, float(N_CELLS * 9)),
    "rlf_percentage": (0.0, 100.0),
    "prb_used": (0.0, 139.0),
    "prb_percentage": (0.0, 100.0),
    "tb_64qam": (0.0, 1.0e5),
    "tb_64qam_ratio": (0.0, 1.0),
    "mac_pdu_64qam": (0.0, 1.0e5),
    "es_on_cost": (0.0, 1.0),
    "cell_state": (0.0, 1.0),
}

#: Deterministic fallback calibration for the reward quantile normaliser.
#: When no calibration file is supplied, each component is fitted on a uniform
#: grid spanning its plausible range.  This reduces to robust (clipped) min-max
#: normalisation while keeping the same transform/inverse API as a calibrated
#: quantile normaliser.
_DEFAULT_REWARD_RANGES: dict[str, tuple[float, float]] = {
    "throughput": (0.0, 1.0e7),
    "energy": (0.0, 1.0e5),
    "rlf": (0.0, float(N_CELLS * 9)),
    "es_on_cost": (0.0, float(N_CELLS)),
    "bs_on": (0.0, float(N_CELLS)),
}


#: EARTH macro-cell power model, per sector with one logical antenna.
#: ``P_on(x) = N * (P_0 + delta_p * P_max * x)`` and ``P_off = N * P_sleep``,
#: with ``x = PRB_used / PRB_total`` in ``[0, 1]``.  The parameters are the
#: macro row of the EARTH table (Auer et al., 2011), reproduced in the MDPI
#: review ``computers-15-00050``: P_0 = 130 W, P_max = 20 W, delta_p = 4.7,
#: P_sleep = 75 W.  ``N = 1`` because the KPI is per cell/sector; ``N = 6``
#: (the EARTH antenna count) is used as a sensitivity check.
DEFAULT_POWER_MODEL: dict[str, float] = {
    "p0_w": 130.0,
    "p_max_w": 20.0,
    "delta_p": 4.7,
    "p_sleep_w": 75.0,
    "n_antennas": 1.0,
    "prb_total": 139.0,
}

ENERGY_MODELS: tuple[str, ...] = ("proxy", "watts")
ES_COST_MODES: tuple[str, ...] = ("dense", "impulse")


def _resolve_power_model(
    power_model: Mapping[str, float] | None = None,
    **overrides: float,
) -> dict[str, float]:
    """Validate and merge an EARTH power-model mapping."""
    resolved = dict(DEFAULT_POWER_MODEL)
    if power_model:
        resolved.update({str(key): float(value) for key, value in power_model.items()})
    resolved.update({str(key): float(value) for key, value in overrides.items() if value is not None})
    unknown = set(resolved) - set(DEFAULT_POWER_MODEL)
    if unknown:
        raise ValueError(f"unknown power-model keys: {sorted(unknown)}")
    if resolved["p0_w"] <= 0:
        raise ValueError("power_model['p0_w'] must be positive")
    if resolved["p_max_w"] < 0 or resolved["delta_p"] < 0:
        raise ValueError("power_model['p_max_w'] and ['delta_p'] must be >= 0")
    if not 0.0 <= resolved["p_sleep_w"] <= resolved["p0_w"]:
        raise ValueError("power_model['p_sleep_w'] must be in [0, p0_w]")
    if resolved["n_antennas"] <= 0:
        raise ValueError("power_model['n_antennas'] must be positive")
    if resolved["prb_total"] <= 0:
        raise ValueError("power_model['prb_total'] must be positive")
    return resolved


def max_cell_power_w(power_model: Mapping[str, float] | None = None) -> float:
    """Maximum on-state power of one cell (W) under the EARTH model."""
    model = _resolve_power_model(power_model)
    return float(
        model["n_antennas"]
        * (model["p0_w"] + model["delta_p"] * model["p_max_w"])
    )


def default_state_ranges(
    energy_max: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Return the default min/max range for every one of the 85 state KPIs.

    ``energy_max`` overrides the per-cell energy range; the watts model passes
    the EARTH maximum (W) so the normalised state is not compressed by the old
    transport-block proxy range.
    """
    ranges: dict[str, tuple[float, float]] = {}
    for cell in CELL_IDS:
        for feature in STATE_FEATURES_PER_CELL:
            if feature == "energy" and energy_max is not None:
                ranges[f"{feature}_{cell}"] = (0.0, float(energy_max))
            else:
                ranges[f"{feature}_{cell}"] = _DEFAULT_KPI_RANGES[feature]
    ranges["bs_on"] = (0.0, float(N_CELLS))
    return ranges


def _default_reward_references(
    energy_max: float | None = None,
    per_cell: bool = False,
) -> dict[str, np.ndarray]:
    ranges = dict(_DEFAULT_REWARD_RANGES)
    if energy_max is not None:
        ranges["energy"] = (0.0, float(energy_max))
    if per_cell:
        # Per-cell components are ~1/N of the network totals.  Dividing the
        # fallback ranges by N keeps the *mean* over cells on the same [0, 1]
        # scale as the global normalisation.
        for key in ("throughput", "energy", "rlf", "es_on_cost"):
            low, high = ranges[key]
            ranges[key] = (low / N_CELLS, high / N_CELLS)
    return {
        key: np.linspace(low, high, 101, dtype=float)
        for key, (low, high) in ranges.items()
    }


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

_STANDARD_NORMAL = statistics.NormalDist()


def _normal_cdf(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.ravel()
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flat),
        dtype=float,
        count=flat.size,
    )
    return result.reshape(array.shape)


def _normal_ppf(values: Any, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.ravel()
    result = np.fromiter(
        (_STANDARD_NORMAL.inv_cdf(float(np.clip(value, eps, 1.0 - eps))) for value in flat),
        dtype=float,
        count=flat.size,
    )
    return result.reshape(array.shape)


def _as_scalar_or_array(result: np.ndarray, scalar: bool):
    if scalar:
        return float(np.asarray(result).reshape(-1)[0])
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _is_integer_action(action: Any) -> bool:
    if isinstance(action, numbers.Integral):
        return True
    if isinstance(action, np.ndarray) and action.ndim == 0:
        return np.issubdtype(action.dtype, np.integer)
    return False


def action_to_cell_bits(action: int | np.integer) -> list[tuple[int, int]]:
    """Convert an integer action to ``[(cell_id, bit), ...]``.

    Bit ``0`` (LSB) maps to cell 2, bit ``6`` maps to cell 8.  ``1`` means ON.
    ``0`` and ``127`` therefore mean "all OFF" and "all ON", respectively.
    """
    if not _is_integer_action(action):
        raise TypeError(f"action must be an integer, got {type(action)!r}")
    value = int(action)
    if not 0 <= value < 2 ** N_CELLS:
        raise ValueError(f"action must be in [0, {2 ** N_CELLS - 1}], got {value}")
    return [(cell, (value >> index) & 1) for index, cell in enumerate(CELL_IDS)]


def cell_bits_to_action(bits: Sequence[int]) -> int:
    """Inverse of :func:`action_to_cell_bits`."""
    values = list(bits)
    if len(values) != N_CELLS:
        raise ValueError(f"expected {N_CELLS} bits, got {len(values)}")
    action = 0
    for index, bit in enumerate(values):
        if int(bit):
            action |= 1 << index
    return action


def action_to_bitstring(action: int) -> str:
    """Return the 7-bit representation (cell 2 first, cell 8 last)."""
    return "".join(str(bit) for _, bit in action_to_cell_bits(action))


def normalize_action(action: Any) -> list[tuple[int, int]]:
    """Accept either the paper's integer action or a 7-value ON/OFF sequence.

    The sequence form keeps ``examples/energy_saving.py`` working: it passes the
    ``State`` column of ``bsState.txt`` (1 = ON) directly to ``step``.
    """
    if _is_integer_action(action):
        return action_to_cell_bits(action)

    array = np.asarray(action)
    if array.ndim != 1 or array.size != N_CELLS:
        raise ValueError(
            f"action must be an integer in [0, 127] or a sequence of {N_CELLS} "
            f"ON/OFF values, got shape {array.shape}"
        )
    return [(cell, 1 if int(value) else 0) for cell, value in zip(CELL_IDS, array)]


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


class QuantileNormalizer:
    """Empirical quantile normaliser used for the reward components.

    The transform is a monotone mapping from the fitted reference distribution
    to ``[0, 1]``:

    * ``method="uniform"``: the output is the empirical percentile.
    * ``method="normal"``: the empirical percentile is first converted to a
      standard-normal score and then passed through a Gaussian CDF fitted on
      the reference normal scores.  This stretches the tails while keeping the
      output in ``[0, 1]``.

    ``fit`` stores the reference values and their mid-rank percentiles, so
    ``transform``/``inverse_transform`` are deterministic and round-trip
    (up to interpolation error).

    A calibration file is an ``.npz`` file containing one array per component
    (or a JSON object mapping component names to reference value lists).
    """

    def __init__(self, method: str = "uniform", clip_eps: float = 1e-6):
        method = str(method).lower()
        if method not in ("uniform", "normal"):
            raise ValueError("normalization must be 'uniform' or 'normal'")
        self.method = method
        self.clip_eps = float(clip_eps)
        self.reference_: np.ndarray | None = None
        self.percentiles_: np.ndarray | None = None
        self.z_mean_ = 0.0
        self.z_std_ = 1.0

    # -- fitting ---------------------------------------------------------
    def fit(self, values: Any) -> "QuantileNormalizer":
        array = np.asarray(values, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size == 0:
            raise ValueError("cannot fit a QuantileNormalizer on an empty array")

        size = array.size
        # Ranks must be computed on the *sorted* sample: averaging the
        # position-based ranks of the original order (the previous
        # implementation) gives the wrong percentile whenever a value repeats
        # in several places of the array.  With the EARTH energy model a whole
        # cell sits at 75 W whenever it sleeps, so those ties are the rule, not
        # the exception; the bug made ``transform`` non-monotone (75 W mapped
        # above 130 W).
        order = np.argsort(array, kind="mergesort")
        sorted_array = array[order]
        ranks = (np.arange(size, dtype=float) + 0.5) / size
        unique, first_index, counts = np.unique(
            sorted_array, return_index=True, return_counts=True
        )
        percentiles = np.add.reduceat(ranks, first_index) / counts
        percentiles = np.clip(percentiles, self.clip_eps, 1.0 - self.clip_eps)

        self.reference_ = unique.astype(float)
        self.percentiles_ = percentiles.astype(float)

        if self.method == "normal":
            z_scores = _normal_ppf(self.percentiles_, self.clip_eps)
            self.z_mean_ = float(np.mean(z_scores))
            self.z_std_ = float(np.std(z_scores)) or 1.0
        return self

    # -- transforms ------------------------------------------------------
    def _check_fitted(self):
        if self.reference_ is None or self.percentiles_ is None:
            raise RuntimeError("QuantileNormalizer must be fitted before use")

    def transform(self, values: Any):
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        scalar = array.ndim == 0
        array = np.atleast_1d(array)

        percentiles = np.interp(
            array,
            self.reference_,
            self.percentiles_,
            left=float(self.percentiles_[0]),
            right=float(self.percentiles_[-1]),
        )
        percentiles = np.clip(percentiles, 0.0, 1.0)

        if self.method == "normal":
            z_scores = _normal_ppf(percentiles, self.clip_eps)
            percentiles = _normal_cdf((z_scores - self.z_mean_) / self.z_std_)

        result = np.clip(percentiles, 0.0, 1.0)
        return _as_scalar_or_array(result, scalar)

    def inverse_transform(self, values: Any):
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        scalar = array.ndim == 0
        array = np.clip(np.atleast_1d(array), 0.0, 1.0)

        if self.method == "normal":
            z_scores = self.z_mean_ + self.z_std_ * _normal_ppf(array, self.clip_eps)
            percentiles = _normal_cdf(z_scores)
        else:
            percentiles = array

        result = np.interp(percentiles, self.percentiles_, self.reference_)
        return _as_scalar_or_array(result, scalar)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        self._check_fitted()
        np.savez_compressed(
            path,
            method=np.array(self.method),
            reference=self.reference_,
            percentiles=self.percentiles_,
            z_mean=np.array(self.z_mean_),
            z_std=np.array(self.z_std_),
        )

    @classmethod
    def load(cls, path: str) -> "QuantileNormalizer":
        with np.load(path, allow_pickle=False) as data:
            method = str(data["method"].item()) if "method" in data else "uniform"
            normalizer = cls(method=method)
            normalizer.reference_ = np.asarray(data["reference"], dtype=float)
            if "percentiles" in data:
                normalizer.percentiles_ = np.asarray(data["percentiles"], dtype=float)
            else:  # pragma: no cover - files written by this module always have it
                size = normalizer.reference_.size
                normalizer.percentiles_ = (np.arange(size) + 0.5) / size
            normalizer.z_mean_ = float(data["z_mean"]) if "z_mean" in data else 0.0
            normalizer.z_std_ = float(data["z_std"]) if "z_std" in data else 1.0
        normalizer.percentiles_ = np.clip(
            normalizer.percentiles_, normalizer.clip_eps, 1.0 - normalizer.clip_eps
        )
        return normalizer


class MinMaxNormalizer:
    """Feature-wise min-max normaliser for the 85-dimensional state."""

    def __init__(
        self,
        feature_ranges: Mapping[str, tuple[float, float]] | None = None,
        feature_names: Sequence[str] | None = None,
        clip: bool = True,
    ):
        ranges = feature_ranges or default_state_ranges()
        self.feature_names = list(feature_names or ranges.keys())
        self.min_ = np.array([float(ranges[name][0]) for name in self.feature_names], dtype=float)
        self.max_ = np.array([float(ranges[name][1]) for name in self.feature_names], dtype=float)
        self.clip = bool(clip)
        self._refresh_span()

    def _refresh_span(self):
        span = self.max_ - self.min_
        span[span == 0] = 1.0
        self.span_ = span

    def fit(self, data: Any) -> "MinMaxNormalizer":
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {array.shape[1]}"
            )
        with np.errstate(invalid="ignore"):
            mins = np.nanmin(array, axis=0)
            maxs = np.nanmax(array, axis=0)
        mins = np.where(np.isfinite(mins), mins, self.min_)
        maxs = np.where(np.isfinite(maxs), maxs, self.max_)
        self.min_ = mins.astype(float)
        self.max_ = maxs.astype(float)
        self._refresh_span()
        return self

    def transform(self, data: Any) -> np.ndarray:
        array = np.asarray(data, dtype=float)
        result = (array - self.min_) / self.span_
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        if self.clip:
            result = np.clip(result, 0.0, 1.0)
        return result

    def inverse_transform(self, data: Any) -> np.ndarray:
        array = np.asarray(data, dtype=float)
        return array * self.span_ + self.min_

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            feature_names=np.array(self.feature_names),
            min=self.min_,
            max=self.max_,
        )

    @classmethod
    def load(cls, path: str) -> "MinMaxNormalizer":
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["feature_names"]]
            ranges = {
                name: (float(low), float(high))
                for name, low, high in zip(names, data["min"], data["max"])
            }
        return cls(ranges, names)


def _load_reward_references(path: str) -> dict[str, np.ndarray]:
    """Load reference arrays for the reward normaliser from ``.npz``/``.json``."""
    references: dict[str, np.ndarray] = {}
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key in ("method", "percentiles", "z_mean", "z_std"):
                    continue
                references[key] = np.asarray(data[key], dtype=float).ravel()
    else:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        components = payload.get("components", payload)
        for key, values in components.items():
            references[str(key)] = np.asarray(values, dtype=float).ravel()
    return references


class RewardNormalizer:
    """One :class:`QuantileNormalizer` per reward component.

    When ``per_cell`` is true, an additional normaliser is built for every
    component/cell pair whose calibration is present (keys such as
    ``throughput_cell_2``).  Components without a per-cell calibration fall
    back to the pooled normaliser, so old calibration files keep working.
    """

    def __init__(
        self,
        method: str = "uniform",
        calibration_file: str | None = None,
        energy_max: float | None = None,
        per_cell: bool = False,
    ):
        self.method = str(method).lower()
        if self.method not in ("uniform", "normal"):
            raise ValueError("normalization must be 'uniform' or 'normal'")
        self.per_cell = bool(per_cell)

        references = _default_reward_references(energy_max=energy_max, per_cell=per_cell)
        if calibration_file:
            references.update(_load_reward_references(calibration_file))

        self.normalizers: dict[str, QuantileNormalizer] = {}
        for component in REWARD_COMPONENTS:
            values = references.get(component)
            if values is None or np.asarray(values).size == 0:
                values = _default_reward_references(
                    energy_max=energy_max, per_cell=per_cell
                )[component]
            self.normalizers[component] = QuantileNormalizer(self.method).fit(values)

        # Optional per-cell normalisers (D3): ``<component>_cell_<id>``.
        self.cell_normalizers: dict[tuple[str, int], QuantileNormalizer] = {}
        if self.per_cell:
            for component in REWARD_COMPONENTS:
                if component == "bs_on":
                    continue
                for cell in CELL_IDS:
                    values = references.get(f"{component}_cell_{cell}")
                    if values is not None and np.asarray(values).size:
                        self.cell_normalizers[(component, cell)] = QuantileNormalizer(
                            self.method
                        ).fit(values)

    def transform(self, raw: Mapping[str, float]) -> dict[str, float]:
        return {
            component: float(self.normalizers[component].transform(raw.get(component, 0.0)))
            for component in REWARD_COMPONENTS
        }

    def transform_cell(self, component: str, cell: int, value: float) -> float:
        """Normalise one cell's component, using its own calibration if any."""
        normalizer = self.cell_normalizers.get((component, int(cell)))
        if normalizer is None:
            normalizer = self.normalizers[component]
        return float(normalizer.transform(value))

    def inverse_transform(self, normalized: Mapping[str, float]) -> dict[str, float]:
        return {
            component: float(self.normalizers[component].inverse_transform(normalized[component]))
            for component in REWARD_COMPONENTS
        }

    def save(self, directory: str) -> None:
        """Save one ``<component>.npz`` calibration file per component."""
        os.makedirs(directory, exist_ok=True)
        for component, normalizer in self.normalizers.items():
            normalizer.save(os.path.join(directory, f"{component}.npz"))


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cell_kpis_from_dataframe(
    df: pd.DataFrame | None,
    cell_ids: Sequence[int] = CELL_IDS,
    timestamp: int | float | None = None,
    rlf_sinr_threshold: float = -5.0,
    prb_total: float = 139.0,
    p_tx: float = 1.0,
    total_ues_per_cell: Mapping[int, int] | None = None,
    aggregation: str = "max",
    *,
    energy_model: str = "proxy",
    power_model: Mapping[str, float] | None = None,
    cell_states: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Aggregate UE-level datalake rows into the 12 per-cell KPIs.

    Parameters
    ----------
    df:
        Rows with at least ``nrCellId``.  The expected columns are the ones in
        :data:`UE_KPMS`; missing columns are treated as zero.  A ``None`` or
        empty frame produces an all-zero frame (apart from ``total_ues``).
    aggregation:
        How cell-level counters are collapsed when the datalake contains one
        row per UE.  ns-O-RAN DU counters such as ``TB.TotNbrDl.1`` and
        ``RRU.PrbUsedDl`` are cell-wide values repeated on every UE row, so the
        default ``"max"`` avoids over-counting.  Use ``"sum"`` if a deployment
        emits genuinely per-UE counters.
    total_ues_per_cell:
        Fallback denominator for ``rlf_percentage`` when SINR rows are absent.
    energy_model:
        ``"proxy"`` (legacy ``TB.TotNbrDl.1 * p_tx``) or ``"watts"`` (EARTH
        ``P = N * (P_0 + delta_p * P_max * x)``, with ``P_sleep`` when the cell
        is OFF).  The watts model needs ``cell_states``; omitted states are
        treated as ON.
    """
    if energy_model not in ENERGY_MODELS:
        raise ValueError(f"energy_model must be one of {ENERGY_MODELS}")
    cell_ids = list(cell_ids)
    if aggregation not in ("max", "sum"):
        raise ValueError("aggregation must be 'max' or 'sum'")

    feature_columns = list(STATE_FEATURES_PER_CELL)
    total_ues_fallback = {cell: int((total_ues_per_cell or {}).get(cell, 0)) for cell in cell_ids}

    if df is None or len(df) == 0 or "nrCellId" not in df.columns:
        if df is not None and len(df) and "nrCellId" not in df.columns:
            raise ValueError("dataframe must contain a 'nrCellId' column")
        result = pd.DataFrame(0.0, index=pd.Index(cell_ids, name="nrCellId"),
                              columns=feature_columns)
        result["total_ues"] = [total_ues_fallback[cell] for cell in cell_ids]
        if energy_model == "watts":
            model = _resolve_power_model(power_model)
            states = cell_states or {}
            result["energy"] = [
                model["n_antennas"]
                * (
                    model["p0_w"]
                    if int(states.get(cell, 1)) == 1
                    else model["p_sleep_w"]
                )
                for cell in cell_ids
            ]
        if timestamp is not None:
            result["timestamp"] = timestamp
        return result

    work = df.copy()
    work["nrCellId"] = _numeric(work["nrCellId"]).astype(int)
    work = work[work["nrCellId"].isin(cell_ids)]

    def grouped(column: str) -> pd.Series:
        if column not in work.columns:
            return pd.Series(0.0, index=pd.Index(cell_ids, dtype=int))
        values = _numeric(work[column])
        work["_metric"] = values
        return work.groupby("nrCellId")["_metric"].agg(aggregation)

    def grouped_with_ue_fallback(column: str, ue_column: str) -> pd.Series:
        main = grouped(column).reindex(cell_ids, fill_value=0.0)
        if column not in work.columns and ue_column in work.columns:
            work["_metric"] = _numeric(work[ue_column])
            return work.groupby("nrCellId")["_metric"].sum().reindex(cell_ids, fill_value=0.0)
        if column in work.columns and float(main.sum()) == 0.0 and ue_column in work.columns:
            work["_metric"] = _numeric(work[ue_column])
            return work.groupby("nrCellId")["_metric"].sum().reindex(cell_ids, fill_value=0.0)
        return main

    throughput = grouped_with_ue_fallback(
        "QosFlow.PdcpPduVolumeDL_Filter", "QosFlow.PdcpPduVolumeDL_Filter.UEID"
    )
    transport_blocks = grouped_with_ue_fallback("TB.TotNbrDl.1", "TB.TotNbrDl.1.UEID")
    prb_used = grouped_with_ue_fallback("RRU.PrbUsedDl", "RRU.PrbUsedDl.UEID")
    tb_64qam = grouped_with_ue_fallback(
        "TB.TotNbrDlInitial.64Qam", "TB.TotNbrDlInitial.64Qam.UEID"
    )
    tb_qpsk = grouped_with_ue_fallback(
        "TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.Qpsk.UEID"
    )
    tb_16qam = grouped_with_ue_fallback(
        "TB.TotNbrDlInitial.16Qam", "TB.TotNbrDlInitial.16Qam.UEID"
    )

    # RLF: count unique UEs whose serving SINR is below the threshold.  When
    # several rows exist per UE (one per QCI), use the minimum SINR.
    rlf_count = pd.Series(0.0, index=cell_ids, dtype=float)
    total_ues = pd.Series(
        [float(total_ues_fallback[cell]) for cell in cell_ids],
        index=cell_ids,
        dtype=float,
    )
    if "L3 serving SINR" in work.columns:
        sinr = work[["nrCellId", "L3 serving SINR"]].copy()
        sinr["L3 serving SINR"] = pd.to_numeric(
            sinr["L3 serving SINR"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        if "ueImsiComplete" in work.columns:
            sinr["_ue"] = work["ueImsiComplete"].astype(str)
        else:
            sinr["_ue"] = np.arange(len(work)).astype(str)

        per_ue = (
            sinr.groupby(["nrCellId", "_ue"], dropna=False)["L3 serving SINR"]
            .min()
            .reset_index()
        )
        rlf_series = (
            per_ue.assign(_rlf=(per_ue["L3 serving SINR"] < float(rlf_sinr_threshold)).astype(float))
            .groupby("nrCellId")["_rlf"]
            .sum()
            .reindex(cell_ids, fill_value=0.0)
        )
        ue_series = (
            per_ue.groupby("nrCellId")["_ue"]
            .nunique()
            .reindex(cell_ids, fill_value=0)
            .astype(float)
        )
        rlf_count = rlf_series
        total_ues = ue_series.where(ue_series > 0, total_ues)

    if energy_model == "proxy":
        energy = transport_blocks.astype(float) * float(p_tx)
    else:  # watts (EARTH macro model)
        model = _resolve_power_model(power_model)
        states = cell_states or {}
        on = np.asarray(
            [1.0 if int(states.get(cell, 1)) == 1 else 0.0 for cell in cell_ids],
            dtype=float,
        )
        load = np.clip(
            prb_used.reindex(cell_ids, fill_value=0.0).to_numpy(dtype=float)
            / float(model["prb_total"]),
            0.0,
            1.0,
        )
        p_on = float(model["n_antennas"]) * (
            float(model["p0_w"])
            + float(model["delta_p"]) * float(model["p_max_w"]) * load
        )
        p_off = float(model["n_antennas"]) * float(model["p_sleep_w"])
        energy = pd.Series(
            np.where(on > 0.0, p_on, p_off),
            index=pd.Index(cell_ids, name="nrCellId"),
            dtype=float,
        )
    safety = 1e-9
    ee_ratio = throughput.astype(float) / np.maximum(energy.astype(float), safety)
    prb_percentage = prb_used.astype(float) / float(prb_total) * 100.0
    modulation_total = tb_qpsk.astype(float) + tb_16qam.astype(float) + tb_64qam.astype(float)
    tb_64qam_ratio = np.where(
        modulation_total > 0.0,
        tb_64qam.astype(float) / np.maximum(modulation_total, safety),
        0.0,
    )
    rlf_percentage = np.where(
        total_ues.to_numpy(dtype=float) > 0.0,
        rlf_count.to_numpy(dtype=float) / total_ues.to_numpy(dtype=float) * 100.0,
        0.0,
    )

    result = pd.DataFrame(index=pd.Index(cell_ids, name="nrCellId"))
    result["throughput_dl"] = throughput.reindex(cell_ids).astype(float)
    result["energy"] = energy.reindex(cell_ids).astype(float)
    result["ee_ratio"] = ee_ratio.reindex(cell_ids).astype(float)
    result["rlf_count"] = rlf_count.reindex(cell_ids).astype(float)
    result["rlf_percentage"] = rlf_percentage
    result["prb_used"] = prb_used.reindex(cell_ids).astype(float)
    result["prb_percentage"] = prb_percentage.reindex(cell_ids).astype(float)
    result["tb_64qam"] = tb_64qam.reindex(cell_ids).astype(float)
    result["tb_64qam_ratio"] = np.asarray(tb_64qam_ratio, dtype=float)
    result["mac_pdu_64qam"] = tb_64qam.reindex(cell_ids).astype(float)
    result["es_on_cost"] = 0.0
    result["cell_state"] = 1.0
    result["total_ues"] = total_ues.reindex(cell_ids).astype(float)
    result = result.fillna(0.0)

    if timestamp is not None:
        result["timestamp"] = timestamp
    return result


def build_observation(
    cell_df: pd.DataFrame,
    cell_states: Mapping[int, int],
    cell_ids: Sequence[int] = CELL_IDS,
) -> np.ndarray:
    """Flatten a cell-centric KPI frame into the 85-dimensional state vector."""
    cell_ids = list(cell_ids)
    values: list[float] = []
    for cell in cell_ids:
        row = cell_df.loc[cell] if cell in cell_df.index else {}
        for feature in STATE_FEATURES_PER_CELL:
            if feature == "cell_state":
                value = float(cell_states.get(cell, 1))
            elif feature == "es_on_cost":
                value = float(row.get("es_on_cost", 0.0)) if hasattr(row, "get") else 0.0
            else:
                value = float(row.get(feature, 0.0)) if hasattr(row, "get") else 0.0
            values.append(value)

    bs_on = sum(1 for cell in cell_ids if int(cell_states.get(cell, 1)) == 1)
    values.append(float(bs_on))

    observation = np.asarray(values, dtype=float)
    observation = np.nan_to_num(
        observation,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    return observation.astype(np.float32)


def compute_reward(
    raw: Mapping[str, float],
    weights: Mapping[str, float],
    normalizer: RewardNormalizer,
    per_cell: Mapping[str, Sequence[tuple[int, float]]] | None = None,
    aggregation: str = "mean",
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Compute Eq. (2) from raw component totals.

    When ``per_cell`` is given (D3), every component except ``bs_on`` is
    normalised cell by cell and then aggregated (``mean`` by default, which is
    ``sum / N`` and keeps the ``[0, 1]`` scale of the weights).  ``bs_on`` is a
    network-level count and stays global.
    """
    if per_cell:
        normalized: dict[str, float] = {}
        for component in REWARD_COMPONENTS:
            values = per_cell.get(component)
            if component == "bs_on" or not values:
                normalized[component] = float(
                    normalizer.transform({component: raw.get(component, 0.0)})[component]
                )
                continue
            cell_values = [
                normalizer.transform_cell(component, int(cell), float(value))
                for cell, value in values
            ]
            if aggregation == "sum":
                normalized[component] = float(np.sum(cell_values))
            else:
                normalized[component] = float(np.mean(cell_values))
    else:
        normalized = normalizer.transform(raw)

    reward = (
        float(weights["w1"]) * normalized["throughput"]
        - float(weights["w2"]) * normalized["energy"]
        - float(weights["w3"]) * normalized["rlf"]
        - float(weights["w4"]) * normalized["es_on_cost"]
        - float(weights["w2"]) * normalized["bs_on"]
    )
    return float(reward), normalized, dict(weights)


def _resolve_weights(
    weights: Mapping[str, float] | Sequence[float] | None,
    w1: float | None,
    w2: float | None,
    w3: float | None,
    w4: float | None,
) -> dict[str, float]:
    if weights is not None:
        if isinstance(weights, Mapping):
            resolved = {
                "w1": float(weights.get("w1", DEFAULT_REWARD_WEIGHTS["w1"])),
                "w2": float(weights.get("w2", DEFAULT_REWARD_WEIGHTS["w2"])),
                "w3": float(weights.get("w3", DEFAULT_REWARD_WEIGHTS["w3"])),
                "w4": float(weights.get("w4", DEFAULT_REWARD_WEIGHTS["w4"])),
            }
        else:
            values = list(weights)
            if len(values) != 4:
                raise ValueError("weights must be a mapping or a sequence of four values")
            resolved = dict(zip(("w1", "w2", "w3", "w4"), map(float, values)))
    else:
        resolved = dict(DEFAULT_REWARD_WEIGHTS)

    for key, value in (("w1", w1), ("w2", w2), ("w3", w3), ("w4", w4)):
        if value is not None:
            resolved[key] = float(value)

    if any(value < 0 for value in resolved.values()):
        raise ValueError("reward weights must be non-negative")
    return resolved


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EnergySavingEnv(NsOranEnv):
    """Paper-compliant energy-saving Gymnasium environment.

    Parameters
    ----------
    reward_weights:
        Mapping/sequence ``(w1, w2, w3, w4)``.  Individual ``w1..w4`` keyword
        arguments override it.
    normalization:
        ``"uniform"`` or ``"normal"`` (quantile normalisation of reward
        components), see :class:`QuantileNormalizer`.
    calibration_file:
        Optional ``.npz``/``.json`` file with reference samples per reward
        component.  When omitted a deterministic robust min-max calibration is
        used (see :data:`_DEFAULT_REWARD_RANGES`).
    p_tx:
        Transmission-power factor of the legacy energy proxy
        ``TB.TotNbrDl.1 * P_TX`` (used when ``energy_model="proxy"``).
    prb_total:
        Number of PRBs used by the PRB percentage (139 for 20 MHz NR).
    energy_model:
        ``"watts"`` (D1, default) uses the EARTH macro model
        ``P = N * (P_0 + delta_p * P_max * x)`` with ``P_sleep`` when OFF;
        ``"proxy"`` keeps the legacy transport-block proxy (ablation).
    power_model:
        Optional overrides for :data:`DEFAULT_POWER_MODEL`
        (``p0_w``, ``p_max_w``, ``delta_p``, ``p_sleep_w``, ``n_antennas``,
        ``prb_total``).
    es_cost_mode:
        ``"dense"`` (D2, default: cost while ON) or ``"impulse"`` (cost only on
        the OFF->ON transition, using the time spent OFF).
    per_cell_normalization:
        ``True`` (D3, default) normalises each component per cell and averages;
        ``False`` uses the legacy network-total normalisation.
    per_cell_aggregation:
        ``"mean"`` (default, keeps the [0, 1] scale) or ``"sum"``.
    state_variant:
        State layout: ``"paper85"`` (default, the paper's 85-dim state) or
        ``"compact"`` (the 70-dim E5 vector of ``docs/plano_tecnico/kpm_audit.md``:
        drops ``mac_pdu_64qam`` and ``prb_percentage`` per cell plus the global
        ``bs_on``).  ``"full"`` and ``"compact78"`` are also accepted.  The
        reward is computed from the raw KPM totals and is unaffected.
    constraint_mode:
        E1/CMDP restriction mode of ``docs/plano_tecnico/cmdp_design.md``
        (§6.1): ``"none"`` (default, the paper's Eq. (2), nothing changes),
        ``"lagrangian"`` (records the per-step hinge cost ``qos_cost`` in
        ``info["reward_components"]`` without changing the collected reward --
        the λ-penalty lives in offline training) or ``"barrier"`` (alias
        ``"gate"``: rewrites the reward as Eq. (5.1'), gating BOTH
        energy-saving channels on the moving average of the raw throughput).
    tau_sla:
        SLA threshold in raw throughput bytes per E2 indication (required for
        ``"barrier"``, optional channel for ``"lagrangian"``).
    epsilon:
        Maximum UEs in RLF per indication (optional ``"lagrangian"`` cost
        channel; unused by ``"barrier"``).
    gate_window:
        Moving-average window ``W`` of the barrier gate in indications
        (design §5.1: 10 ≈ 1 s of simulation).
    gate_temperature:
        Temperature of the soft gate ``sigmoid((mean − tau_sla) / T)``;
        ``None`` keeps the hard gate (design §8.2-B1 recommends the soft gate
        with ``T ≈ 0.05·μ_AO``).
    setup_sim:
        Set to ``False`` to build the object without configuring/building ns-3
        (used by unit tests).
    skip_build:
        Set to ``True`` to reuse an existing ns-3 build: the scenario
        executable is still resolved but the (not concurrency safe) CMake
        configuration step is skipped.  Parallel runs must build once and then
        pass ``skip_build=True``.
    """

    gnb_state_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "state": "INTEGER",
    }

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool,
        do_heuristic: bool = True,
        *,
        reward_weights: Mapping[str, float] | Sequence[float] | None = None,
        w1: float | None = None,
        w2: float | None = None,
        w3: float | None = None,
        w4: float | None = None,
        normalization: str = "uniform",
        calibration_file: str | None = None,
        state_calibration_file: str | None = None,
        p_tx: float = 1.0,
        prb_total: float = 139.0,
        rlf_sinr_threshold: float = -5.0,
        es_decay: float = 0.9,
        es_time_factor: float = 0.01,
        du_aggregation: str = "max",
        energy_model: str = "watts",
        power_model: Mapping[str, float] | None = None,
        es_cost_mode: str = "dense",
        per_cell_normalization: bool = True,
        per_cell_aggregation: str = "mean",
        state_variant: str = "paper85",
        constraint_mode: str = "none",
        tau_sla: float | None = None,
        epsilon: float | None = None,
        gate_window: int = 10,
        gate_temperature: float | None = None,
        step_timeout: float = 300.0,
        reset_timeout: float = 300.0,
        setup_sim: bool = True,
        skip_build: bool = False,
    ):
        super().__init__(
            ns3_path=ns3_path,
            scenario="scenario-three",
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=["timestamp", "cellId", "hoAllowed"],
            log_file="EsActions.txt",
            control_file="es_actions_for_ns3.csv",
            skip_setup=not setup_sim,
            skip_build=skip_build,
            reset_timeout=reset_timeout,
        )

        self.folder_name = "Simulation"
        self.ns3_simulation_time = float(self.scenario_configuration["simTime"]) * 1000.0
        self.indication_periodicity_ms = float(
            self.scenario_configuration.get("indicationPeriodicity", 0.1)
        ) * 1000.0

        # --- spaces (paper contract) -----------------------------------
        self.action_space = spaces.Discrete(2 ** N_CELLS)
        # ``state_variant`` defaults to the paper's 85-dim state; "compact"
        # selects the 70-dim E5 vector (docs/plano_tecnico/kpm_audit.md).
        # The kept indices (the contract) come from nsoran.drl.config; the
        # column names keep es_env's own ``_{cell}`` convention.
        # Unknown names raise ValueError listing the registered variants.
        self.state_variant = str(state_variant)
        self.state_keep_indices: tuple[int, ...] = state_variant_indices(
            self.state_variant
        )
        self.state_feature_names: tuple[str, ...] = tuple(
            STATE_FEATURE_NAMES[index] for index in self.state_keep_indices
        )
        self.obs_dim = len(self.state_keep_indices)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # --- energy / reward model (D1-D3) -----------------------------
        self.energy_model = str(energy_model).lower()
        if self.energy_model not in ENERGY_MODELS:
            raise ValueError(f"energy_model must be one of {ENERGY_MODELS}")
        self.prb_total = float(prb_total)
        power_overrides = dict(power_model or {})
        power_overrides.setdefault("prb_total", self.prb_total)
        self.power_model = _resolve_power_model(power_overrides)

        self.es_cost_mode = str(es_cost_mode).lower()
        if self.es_cost_mode not in ES_COST_MODES:
            raise ValueError(f"es_cost_mode must be one of {ES_COST_MODES}")
        self.per_cell_normalization = bool(per_cell_normalization)
        self.per_cell_aggregation = str(per_cell_aggregation).lower()
        if self.per_cell_aggregation not in ("mean", "sum"):
            raise ValueError("per_cell_aggregation must be 'mean' or 'sum'")

        energy_max = (
            max_cell_power_w(self.power_model)
            if self.energy_model == "watts"
            else None
        )

        # --- reward / normalisation ------------------------------------
        self.reward_weights = _resolve_weights(reward_weights, w1, w2, w3, w4)
        self.normalization = str(normalization).lower()
        self.calibration_file = calibration_file
        self.reward_normalizer = RewardNormalizer(
            self.normalization,
            calibration_file,
            energy_max=energy_max,
            per_cell=self.per_cell_normalization,
        )

        if state_calibration_file:
            self.state_normalizer = MinMaxNormalizer.load(state_calibration_file)
        else:
            self.state_normalizer = MinMaxNormalizer(
                default_state_ranges(energy_max=energy_max),
                feature_names=STATE_FEATURE_NAMES,
            )

        # --- CMDP / E1 (docs/plano_tecnico/cmdp_design.md §6.1) ----------
        # Default constraint_mode="none": a recompensa é exatamente a Eq. (2)
        # do paper e nenhum estado adicional entra em jogo.  A validação dos
        # limiares fica no CmdpConfig (ValueError na construção).
        self.cmdp_config = CmdpConfig(
            constraint_mode=constraint_mode,
            tau_sla=tau_sla,
            epsilon=epsilon,
            gate_window=gate_window,
            gate_temperature=gate_temperature,
        )
        #: Janela móvel do throughput cru (bytes/indicação) do episódio
        #: corrente, usada pela porta do modo "barrier" (Eq. 5.1′).  Esvaziada
        #: em reset().
        self._throughput_window: deque[float] = deque(
            maxlen=int(self.cmdp_config.gate_window)
        )

        # --- KPI extraction parameters ---------------------------------
        self.p_tx = float(p_tx)
        self.rlf_sinr_threshold = float(rlf_sinr_threshold)
        self.du_aggregation = str(du_aggregation)
        self.es_decay = float(es_decay)
        self.es_time_factor = float(es_time_factor)
        self.step_timeout = float(step_timeout)

        # --- compatibility attributes ----------------------------------
        self.columns_state = list(self.state_feature_names)
        self.columns_reward = [
            "SUM_QosFlow.PdcpPduVolumeDL_Filter",
            "SUM_TB.TotNbrDl.1",
            "SUM_RLF_VALUE",
            "SUM_ES_ON_COST",
            "ZERO_COUNT",
        ]
        self.action_list = list(LEGACY_ACTION_LIST)
        self.cellList = list(CELL_IDS)
        self.observations = pd.DataFrame()

        # --- runtime state ---------------------------------------------
        self.cells_states: dict[int, int] = {cell: 1 for cell in CELL_IDS}
        self.cell_on_since: dict[int, float | None] = {cell: None for cell in CELL_IDS}
        self.cell_off_since: dict[int, float | None] = {cell: None for cell in CELL_IDS}
        self._previous_states: dict[int, int] = {cell: 1 for cell in CELL_IDS}
        self.cell_timestamp_state_dict = {cell: float("inf") for cell in CELL_IDS}
        self.Cf = 1.0
        self.lambdaf = 1.0 - self.es_decay
        self.time_factor = self.es_time_factor
        self.heur = do_heuristic
        self.num_steps = 0
        self.previous_inverted_action = "0000000"
        self.last_action_bits: list[tuple[int, int]] | None = None
        self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._last_raw: dict[str, Any] | None = None
        self._last_cell_df = pd.DataFrame()
        self._last_info: dict[str, Any] = {}
        self._last_bsstate_unix = 0
        self.last_reward_components: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium lifecycle
    # ------------------------------------------------------------------
    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Close any previous simulation and start a fresh one.

        ``seed`` is mapped to ns-3's ``RngRun`` so independent runs can be
        reproduced.  The configured ``RngRun`` (555 in the paper) is used when
        ``seed`` is omitted.
        """
        if seed is None and options and options.get("rng_run") is not None:
            seed = int(options["rng_run"])
        if seed is not None:
            self.scenario_configuration["RngRun"] = int(seed)

        self.reset_es_tracking()
        self.cells_states = {cell: 1 for cell in CELL_IDS}
        self.cell_timestamp_state_dict = {cell: float("inf") for cell in CELL_IDS}
        self.num_steps = 0
        self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._last_raw = None
        self._last_cell_df = pd.DataFrame()
        self._last_bsstate_unix = 0
        self.previous_inverted_action = "0000000"
        self.last_action_bits = None
        self._throughput_window.clear()

        observation, _ = super().reset(seed=seed, options=options)
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.obs_dim,):
            observation = observation.reshape(-1)
        if seed is not None:
            self.action_space.seed(int(seed))
            self.observation_space.seed(int(seed))
        self._last_obs = observation
        return observation, self._get_info()

    @override
    def step(self, action: Any):
        """Apply one action and return the next paper-compliant observation.

        The wait for the next E2 indication is bounded by ``step_timeout`` so a
        missing semaphore release can never deadlock a training loop.
        """
        return self._advance(action, apply_action=True)

    def step_baseline(self):
        """Advance one E2 period *without* applying an external action.

        This is what the paper's baseline policies need: the ON/OFF decisions
        are taken inside ns-3 (the built-in ``EnergyHeuristic``), so the
        environment must only release the control semaphore to let the
        simulation continue.  Writing to the control file would override the
        heuristic and would evict the users of the cells it turned off.
        """
        return self._advance(None, apply_action=False)

    def _advance(self, action: Any, apply_action: bool):
        if not self.is_open:
            raise RuntimeError("The environment is not open; call reset() first")

        self.num_steps += 1
        if apply_action:
            self.last_action_bits = normalize_action(action)

        if self.is_simulation_over():
            # ``is_simulation_over()`` records the exit status of ns-3: a clean
            # exit sets terminated=True, an abort/assertion sets terminated=False
            # and truncated=True.  Reporting the real flags is what makes a
            # crashed simulation distinguishable from a complete one.
            return (
                self._last_obs,
                0.0,
                bool(getattr(self, "terminated", False)),
                bool(getattr(self, "truncated", False)),
                self._get_info(),
            )

        if apply_action:
            actions = self._compute_action(action)
            self.action_controller.create_control_action(self.last_timestamp, actions)
        self.controlSemaphore.release()

        deadline = time.monotonic() + self.step_timeout
        data_ready = self._wait_data_availability(
            timeout=self.step_timeout, deadline=deadline
        )

        if not data_ready:
            over = self.is_simulation_over()
            terminated = bool(getattr(self, "terminated", False))
            if over:
                terminated = True
                truncated = bool(getattr(self, "truncated", False))
            else:
                truncated = True
            self._last_info = self._build_info_dict()
            self._last_info["timeout"] = not over
            return self._last_obs, 0.0, terminated, truncated, self._get_info()

        try:
            self._fill_datalake()
        except Exception as error:  # pragma: no cover - defensive against ns-3 I/O
            # Keep the previous observation rather than killing a training run.
            print(f"Warning: could not fill datalake at step {self.num_steps}: {error}")

        observation = self._get_obs()
        if not apply_action:
            # The decision was taken inside ns-3 (the built-in heuristic): record
            # it in the log so that the run can be converted into an offline
            # dataset, but do not deliver it to ns-3.
            self.action_controller.log_action(
                self.last_timestamp,
                [[cell, int(self.cells_states.get(cell, 1))] for cell in CELL_IDS],
            )
        reward = self._compute_reward()
        terminated = bool(getattr(self, "terminated", False))
        truncated = bool(getattr(self, "truncated", False))
        return observation, reward, terminated, truncated, self._get_info()

    @override
    def close(self):
        """Terminate ns-3 and release semaphores.

        The base class implementation is not idempotent and leaks the selector
        and pipe file descriptors; this override is safe to call multiple
        times.
        """
        super().close()

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------
    @override
    def _compute_action(self, action: Any):
        """Convert an agent action to the ns-O-RAN control file rows.

        Integer actions follow the paper bit mapping (LSB = cell 2, 1 = ON).
        A sequence of seven ON/OFF values is accepted for backwards
        compatibility with ``examples/energy_saving.py``.
        """
        self.last_action_bits = normalize_action(action)
        self.previous_inverted_action = "".join(
            str(bit) for _, bit in reversed(self.last_action_bits)
        )
        return [[cell, int(bit)] for cell, bit in self.last_action_bits]

    # ------------------------------------------------------------------
    # Datalake / simulation helpers
    # ------------------------------------------------------------------
    @override
    def _init_datalake_usecase(self):
        grafana_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "time_grafana": "INTEGER",
            "step": "INTEGER",
            "throughput": "REAL",
            "en_cons": "REAL",
            "rlf": "REAL",
            "on_cost": "REAL",
            "reward": "REAL",
        }
        self.datalake._create_table("bsState", self.gnb_state_keys)
        self.datalake._create_table("grafana", grafana_keys)
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        """Insert new ``bsState.txt`` rows into the datalake.

        ``bsState`` uses UNIX milliseconds while the other KPM files use
        simulation-relative milliseconds, so ``self.last_timestamp`` is
        deliberately *not* updated here.  Only timestamps newer than the last
        inserted group are processed, which avoids duplicate rows when the
        whole file is re-read at every step.
        """
        rows_by_timestamp: dict[int, list[tuple[int, int]]] = {}
        for file_path in glob.glob(os.path.join(self.sim_path, "bsState.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as csvfile:
                    for row in csv.DictReader(csvfile, delimiter=" "):
                        try:
                            timestamp = int(row["UNIX"])
                            cell_id = int(row["Id"])
                            state = int(row["State"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if cell_id not in CELL_IDS:
                            continue
                        rows_by_timestamp.setdefault(timestamp, []).append((cell_id, state))
            except OSError:
                continue

        new_timestamps = sorted(
            timestamp for timestamp in rows_by_timestamp if timestamp > self._last_bsstate_unix
        )
        if not new_timestamps:
            return

        for timestamp in new_timestamps:
            for cell_id, state in rows_by_timestamp[timestamp]:
                try:
                    self.datalake.insert_data(
                        "bsState",
                        {
                            "timestamp": timestamp,
                            "ueImsiComplete": None,
                            "cellId": cell_id,
                            "state": state,
                        },
                    )
                except Exception:
                    continue
        self._last_bsstate_unix = max(new_timestamps)

    def _read_ue_kpms(self, timestamp: int) -> tuple[pd.DataFrame | None, int]:
        """Read the UE-level KPMs, falling back one E2 period and then to DU."""
        timestamps = [int(timestamp)]
        fallback = int(timestamp - self.indication_periodicity_ms)
        if fallback >= 0 and fallback not in timestamps:
            timestamps.append(fallback)

        for candidate in timestamps:
            for kpms in (UE_KPMS, DU_KPMS):
                try:
                    rows = self.datalake.read_kpms(candidate, kpms)
                except Exception:
                    rows = None
                if rows:
                    return pd.DataFrame(
                        rows, columns=["ueImsiComplete", *kpms]
                    ), candidate
        return None, int(timestamp)

    def _get_cell_states(self, timestamp: int) -> tuple[dict[int, int], dict[str, int]]:
        """Return the ON/OFF state of the seven cells and timestamp metadata.

        ``bsState`` timestamps are UNIX ms; the first observed timestamp is used
        as the epoch to convert them to simulation time.  A half-period
        tolerance absorbs the off-by-one E2 period mentioned in the integration
        notes.
        """
        default_states = {cell: 1 for cell in CELL_IDS}
        try:
            rows = self.datalake.read_table("bsState")
        except Exception:
            return default_states, {
                "bsstate_timestamp_unix": -1,
                "bsstate_timestamp_rel": -1,
                "bsstate_epoch_unix": -1,
            }

        groups: dict[int, dict[int, int]] = {}
        for row in rows:
            try:
                unix_timestamp = int(row[0])
                cell_id = int(row[2])
                state = int(row[3])
            except (IndexError, TypeError, ValueError):
                continue
            if cell_id not in CELL_IDS:
                continue
            groups.setdefault(unix_timestamp, {})[cell_id] = 1 if state else 0

        if not groups:
            return default_states, {
                "bsstate_timestamp_unix": -1,
                "bsstate_timestamp_rel": -1,
                "bsstate_epoch_unix": -1,
            }

        epoch = min(groups)
        tolerance = self.indication_periodicity_ms * 0.5

        # The DU/CU KPM timestamps are ``m_startTime + Simulator::Now()`` and
        # ``bsState`` uses the same start-time based UNIX-ms clock, so normally
        # both are directly comparable.  The relative-clock branch keeps the
        # environment usable with older logs where KPM timestamps start at 0.
        if int(timestamp) >= epoch:
            eligible = [
                unix_timestamp
                for unix_timestamp in groups
                if unix_timestamp <= float(timestamp) + tolerance
            ]
            chosen_unix = max(eligible) if eligible else epoch
        else:
            relative = {unix_timestamp - epoch: unix_timestamp for unix_timestamp in groups}
            eligible = [rel for rel in relative if rel <= float(timestamp) + tolerance]
            chosen_relative = max(eligible) if eligible else min(relative)
            chosen_unix = relative[chosen_relative]

        states = dict(default_states)
        states.update(groups[chosen_unix])
        metadata = {
            "bsstate_timestamp_unix": int(chosen_unix),
            "bsstate_timestamp_rel": int(chosen_unix - epoch),
            "bsstate_epoch_unix": int(epoch),
        }
        return states, metadata

    def _default_ues_per_cell(self) -> dict[int, int]:
        try:
            return {cell: int(self.scenario_configuration["ues"]) for cell in CELL_IDS}
        except (KeyError, TypeError, ValueError):
            return {cell: 0 for cell in CELL_IDS}

    def reset_es_tracking(self) -> None:
        """Reset the activation-cost state (also used by offline replay)."""
        self.cell_on_since = {cell: None for cell in CELL_IDS}
        self.cell_off_since = {cell: None for cell in CELL_IDS}
        self._previous_states = {cell: 1 for cell in CELL_IDS}

    def _update_es_on_cost(
        self, cell_df: pd.DataFrame, states: Mapping[int, int], timestamp: int
    ) -> pd.DataFrame:
        """Fill the per-cell activation cost of Eq. (2).

        ``dense`` (default) charges every ON step with
        ``0.9**(0.01*T_on_ms)``; ``impulse`` charges only the OFF->ON
        transition with ``0.9**(0.01*T_off_ms)``.  Both use the *observed*
        cell state, so the cost matches the state in the same observation.
        """
        period = float(self.indication_periodicity_ms)
        costs = []
        for cell in CELL_IDS:
            is_on = int(states.get(cell, 1)) == 1
            previous_on = int(self._previous_states.get(cell, 1)) == 1

            if self.es_cost_mode == "dense":
                if is_on:
                    if self.cell_on_since[cell] is None:
                        self.cell_on_since[cell] = float(timestamp)
                        time_on_ms = period
                    else:
                        time_on_ms = max(
                            float(timestamp) - float(self.cell_on_since[cell]),
                            period,
                        )
                    cost = self.es_decay ** (self.es_time_factor * time_on_ms)
                else:
                    self.cell_on_since[cell] = None
                    cost = 0.0
            else:  # impulse
                cost = 0.0
                if is_on and not previous_on:
                    if self.cell_off_since[cell] is None:
                        time_off_ms = period
                    else:
                        time_off_ms = max(
                            float(timestamp) - float(self.cell_off_since[cell]),
                            period,
                        )
                    cost = self.es_decay ** (self.es_time_factor * time_off_ms)
                if is_on:
                    self.cell_off_since[cell] = None
                elif previous_on or self.cell_off_since[cell] is None:
                    # First observation of this OFF period: start the clock.
                    self.cell_off_since[cell] = float(timestamp)

            costs.append(float(cost))

        cell_df["es_on_cost"] = costs
        self._previous_states = {
            cell: 1 if int(states.get(cell, 1)) == 1 else 0 for cell in CELL_IDS
        }
        return cell_df

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    @override
    def _get_obs(self) -> np.ndarray:
        timestamp = int(getattr(self, "last_timestamp", 0) or 0)
        ue_df, kpm_timestamp = self._read_ue_kpms(timestamp)
        states, bsstate_metadata = self._get_cell_states(timestamp)
        self.cells_states = states

        cell_df = cell_kpis_from_dataframe(
            ue_df,
            cell_ids=CELL_IDS,
            timestamp=kpm_timestamp,
            rlf_sinr_threshold=self.rlf_sinr_threshold,
            prb_total=self.prb_total,
            p_tx=self.p_tx,
            total_ues_per_cell=self._default_ues_per_cell(),
            aggregation=self.du_aggregation,
            energy_model=self.energy_model,
            power_model=self.power_model,
            cell_states=states,
        )
        cell_df = self._update_es_on_cost(cell_df, states, kpm_timestamp)
        cell_df["cell_state"] = [float(states.get(cell, 1)) for cell in CELL_IDS]

        raw_observation = build_observation(cell_df, states)
        normalized_observation = self.state_normalizer.transform(raw_observation)
        normalized_observation = np.asarray(
            normalized_observation, dtype=np.float32
        ).reshape(STATE_DIM)
        if self.obs_dim != STATE_DIM:
            # E5 compact vector: select the columns kept by the state variant
            # (the same indices the offline dataset converter applies, so a
            # live observation matches the corresponding dataset layout).
            normalized_observation = normalized_observation[
                list(self.state_keep_indices)
            ]

        self.observations = cell_df
        self._last_cell_df = cell_df
        self._last_obs = normalized_observation
        self._last_raw = self._build_raw_kpms(
            cell_df, raw_observation, states, kpm_timestamp, timestamp, bsstate_metadata
        )
        return normalized_observation

    def _build_raw_kpms(
        self,
        cell_df: pd.DataFrame,
        raw_observation: np.ndarray,
        states: Mapping[int, int],
        kpm_timestamp: int,
        last_timestamp: int,
        bsstate_metadata: Mapping[str, int],
    ) -> dict[str, Any]:
        # ``raw_observation`` is always the full 85-dim raw state, even when
        # ``state_variant="compact"`` shrinks the policy observation: the
        # reward and the KPM audit trail must stay variant-independent.
        cells: dict[str, dict[str, float]] = {}
        for cell in CELL_IDS:
            row = cell_df.loc[cell]
            cells[str(cell)] = {
                feature: float(row.get(feature, 0.0))
                for feature in STATE_FEATURES_PER_CELL
            }
            cells[str(cell)]["total_ues"] = float(row.get("total_ues", 0.0))

        totals = {
            "throughput": float(cell_df["throughput_dl"].sum()),
            "energy": float(cell_df["energy"].sum()),
            "rlf": float(cell_df["rlf_count"].sum()),
            "es_on_cost": float(cell_df["es_on_cost"].sum()),
            "bs_on": float(sum(1 for cell in CELL_IDS if int(states.get(cell, 1)) == 1)),
        }
        per_cell = {
            "throughput": [
                (cell, float(cell_df.loc[cell, "throughput_dl"])) for cell in CELL_IDS
            ],
            "energy": [
                (cell, float(cell_df.loc[cell, "energy"])) for cell in CELL_IDS
            ],
            "rlf": [
                (cell, float(cell_df.loc[cell, "rlf_count"])) for cell in CELL_IDS
            ],
            "es_on_cost": [
                (cell, float(cell_df.loc[cell, "es_on_cost"])) for cell in CELL_IDS
            ],
        }
        return {
            "timestamps": {
                "kpm": int(kpm_timestamp),
                "last_timestamp": int(last_timestamp),
                "simulation_ms": int(last_timestamp),
                **{key: int(value) for key, value in bsstate_metadata.items()},
            },
            "cells": cells,
            "totals": totals,
            "per_cell": per_cell,
            "raw_observation": raw_observation.astype(float).tolist(),
            "bs_on": int(totals["bs_on"]),
            "cell_states": {str(cell): int(states.get(cell, 1)) for cell in CELL_IDS},
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    @override
    def _compute_reward(self) -> float:
        raw = (self._last_raw or {}).get("totals")
        if raw is None:
            return 0.0

        reward, normalized, weights = compute_reward(
            raw,
            self.reward_weights,
            self.reward_normalizer,
            per_cell=(
                (self._last_raw or {}).get("per_cell")
                if self.per_cell_normalization
                else None
            ),
            aggregation=self.per_cell_aggregation,
        )
        self.last_reward_components = {
            "raw": dict(raw),
            "normalized": normalized,
            "weights": weights,
            "reward": reward,
        }

        # --- CMDP / E1 (docs/plano_tecnico/cmdp_design.md §6.1) ----------
        # Inativo em constraint_mode="none" (o dict acima fica exatamente
        # como antes).  "lagrangian": a restrição vive no treino offline —
        # aqui só registramos o custo hinge por passo (Eq. 3.2) para
        # diagnóstico, sem alterar a recompensa coletada.  "barrier":
        # reescreve a recompensa pela Eq. (5.1′), portando os DOIS canais de
        # economia (energia e bs_on, ambos ponderados por w2) pela média
        # móvel do throughput cru da janela do episódio.
        if self.cmdp_config.enabled:
            raw_throughput = float(raw["throughput"])
            self._throughput_window.append(raw_throughput)
            components = self.last_reward_components
            components["qos_cost"] = float(
                self.cmdp_config.qos_cost(raw_throughput, float(raw["rlf"]))
            )
            if self.cmdp_config.constraint_mode == "barrier":
                window_mean = sum(self._throughput_window) / len(
                    self._throughput_window
                )
                gate = float(self.cmdp_config.gate_factor(window_mean))
                if gate != 1.0:
                    # Referência testada desta fórmula:
                    # nsoran.drl.config.compute_gated_reward.
                    reward = (
                        float(weights["w1"]) * normalized["throughput"]
                        - float(weights["w3"]) * normalized["rlf"]
                        - float(weights["w4"]) * normalized["es_on_cost"]
                        - float(weights["w2"])
                        * (normalized["energy"] + normalized["bs_on"])
                        * gate
                    )
                    self.last_reward_components["reward"] = reward
                # "gate_open: bool" do design (§6.1); o valor exato da porta
                # (contínua no caso suave) fica em "gate_value".
                components["gate_open"] = bool(gate >= 0.5)
                components["gate_value"] = gate
                components["throughput_window_mean"] = float(window_mean)

        # Backwards-compatible Grafana table (best effort; never fail a step).
        try:
            db_row = {
                "timestamp": int(getattr(self, "last_timestamp", 0) or 0),
                "ueImsiComplete": None,
                "time_grafana": int(getattr(self, "last_timestamp", 0) or 0),
                "step": int(self.num_steps),
                "throughput": float(raw["throughput"]),
                "en_cons": float(raw["energy"]),
                "rlf": float(raw["rlf"]),
                "on_cost": float(raw["es_on_cost"]),
                "reward": reward,
            }
            self.datalake.insert_data("grafana", db_row)
        except Exception:
            pass
        return float(reward)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def _build_info_dict(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step": int(self.num_steps),
            "bs_on": int((self._last_raw or {}).get("bs_on", 0)),
            "cell_states": {
                str(cell): int(self.cells_states.get(cell, 1)) for cell in CELL_IDS
            },
            "action": [[cell, int(bit)] for cell, bit in (self.last_action_bits or [])],
            "reward_components": self.last_reward_components,
            "normalization": self.normalization,
            "reward_weights": dict(self.reward_weights),
            "env_config": {
                "energy_model": self.energy_model,
                "power_model": dict(self.power_model),
                "es_cost_mode": self.es_cost_mode,
                "per_cell_normalization": self.per_cell_normalization,
                "per_cell_aggregation": self.per_cell_aggregation,
                "state_variant": self.state_variant,
                "calibration_file": self.calibration_file,
                "p_tx": self.p_tx,
                "prb_total": self.prb_total,
                "cmdp": self.cmdp_config.as_dict(),
            },
        }
        if self._last_raw:
            info["timestamps"] = self._last_raw["timestamps"]
            info["raw_kpms"] = {
                "cells": self._last_raw["cells"],
                "totals": self._last_raw["totals"],
                "per_cell": self._last_raw.get("per_cell", {}),
            }
            info["raw_observation"] = self._last_raw["raw_observation"]
        return info

    @override
    def _get_info(self) -> dict[str, Any]:
        info = dict(super()._get_info() or {})
        info.update(self._build_info_dict())
        return info

    # ------------------------------------------------------------------
    # Backwards-compatible helpers
    # ------------------------------------------------------------------
    def bs_states_list(self) -> list[int]:
        """Return current cell states in cell order (1 = ON, 0 = OFF)."""
        return [int(self.cells_states.get(cell, 1)) for cell in CELL_IDS]

    def offline_training_preprocessing(
        self, df: pd.DataFrame, cell_states: Mapping[int, int] | None = None
    ) -> pd.DataFrame:
        """Legacy entry point: aggregate a UE-level frame into cell KPIs."""
        return cell_kpis_from_dataframe(
            df,
            cell_ids=CELL_IDS,
            rlf_sinr_threshold=self.rlf_sinr_threshold,
            prb_total=self.prb_total,
            p_tx=self.p_tx,
            aggregation=self.du_aggregation,
            energy_model=self.energy_model,
            power_model=self.power_model,
            cell_states=cell_states,
        )


class PaperEnergySavingEnv(EnergySavingEnv):
    """Convenience subclass using the PPO-1 row of Table I as defaults."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reward_weights", PAPER_REWARD_WEIGHTS["PPO-1"])
        kwargs.setdefault("normalization", PAPER_REWARD_WEIGHTS["PPO-1"]["normalization"])
        super().__init__(*args, **kwargs)


# The example imports ``EnergySavingEnv``; exporting the alias explicitly makes
# the paper-flavoured name discoverable without breaking existing imports.
__all__ = [
    "CELL_IDS",
    "DEFAULT_POWER_MODEL",
    "ENERGY_MODELS",
    "ES_COST_MODES",
    "EnergySavingEnv",
    "PaperEnergySavingEnv",
    "PAPER_REWARD_WEIGHTS",
    "QuantileNormalizer",
    "MinMaxNormalizer",
    "RewardNormalizer",
    "STATE_DIM",
    "STATE_FEATURE_NAMES",
    "STATE_FEATURES_PER_CELL",
    "action_to_cell_bits",
    "cell_bits_to_action",
    "cell_kpis_from_dataframe",
    "build_observation",
    "compute_reward",
    "default_state_ranges",
    "max_cell_power_w",
    "normalize_action",
]
