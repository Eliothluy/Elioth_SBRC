"""E1: explicit QoS constraint for the offline cell-sleep agents.

The paper's reward is a weighted sum, so a policy can buy energy with
throughput without any hard feasibility guarantee.  E1 turns the QoS target
into a constrained MDP (CMDP):

    maximise  E[r(s, a)]
    subject to E[c(s, a)] <= 0

with the per-transition cost

    c = max(0, tau_SLA - throughput) / throughput_scale
        + max(0, rlf - epsilon) / rlf_scale

The training objective is the Lagrangian ``r - lambda * c`` and ``lambda`` is
updated by dual ascent after every gradient step::

    lambda <- clip(lambda + lr * (mean(c) - tolerance), 0, lambda_max)

This is deliberately *not* a re-weighting of the paper's linear reward: the
QoS term is a separate cost channel and its multiplier is learned from the
constraint violation.  The raw KPM units are stored in the offline dataset's
``raw_components`` array, so the thresholds are expressed in the same units
as the ns-3 traces (throughput volume and RLF UE count), not in the
quantile-normalised reward space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # optional dependency group: drl
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.qos requires numpy. Install the DRL extras with "
        "`pip install -e '.[drl]'` or `pip install -r requirements-drl.txt`."
    ) from exc


@dataclass(frozen=True)
class QoSConstraint:
    """Absolute SLA thresholds in the units of the raw ns-3 KPMs.

    Parameters
    ----------
    throughput_min:
        Minimum mean throughput per E2 indication (same unit as
        ``QosFlow.PdcpPduVolumeDL_Filter``).  ``None`` disables the
        throughput part of the constraint.
    rlf_max:
        Maximum number of UEs in RLF (SINR below the configured threshold)
        per indication.  ``None`` disables the RLF part.
    throughput_scale / rlf_scale:
        Normalisers that put both violations on a comparable scale.  They
        default to ``max(throughput_min, 1)`` and ``max(rlf_max, 1)``.
    lambda_lr, lambda_init, lambda_max:
        Dual-ascent hyper-parameters.
    tolerance:
        Allowed mean cost.  ``0`` means the constraint must hold exactly.
    """

    throughput_min: Optional[float] = None
    rlf_max: Optional[float] = None
    throughput_scale: Optional[float] = None
    rlf_scale: Optional[float] = None
    lambda_lr: float = 0.01
    lambda_init: float = 0.0
    lambda_max: float = 100.0
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("throughput_min", "rlf_max"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0 when provided.")
        for name in ("throughput_scale", "rlf_scale"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0 when provided.")
        if self.lambda_lr < 0.0:
            raise ValueError("lambda_lr must be >= 0.")
        if self.lambda_init < 0.0:
            raise ValueError("lambda_init must be >= 0.")
        if self.lambda_max < 0.0:
            raise ValueError("lambda_max must be >= 0.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be >= 0.")

    @property
    def enabled(self) -> bool:
        return self.throughput_min is not None or self.rlf_max is not None

    def normalisers(self) -> Tuple[float, float]:
        throughput_scale = (
            float(self.throughput_scale)
            if self.throughput_scale is not None
            else max(float(self.throughput_min or 0.0), 1.0)
        )
        rlf_scale = (
            float(self.rlf_scale)
            if self.rlf_scale is not None
            else max(float(self.rlf_max or 0.0), 1.0)
        )
        return throughput_scale, rlf_scale

    def cost(self, raw_components: Any) -> np.ndarray:
        """Return the non-negative per-transition QoS cost.

        ``raw_components`` must have shape ``[N, 5]`` in the dataset order
        ``(throughput, energy, rlf, es_on_cost, bs_on)``.
        """

        values = np.asarray(raw_components, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError(
                "raw_components must have shape [N, >=3] with columns "
                "(throughput, energy, rlf, ...), got "
                f"{values.shape}."
            )
        throughput = values[:, 0]
        rlf = values[:, 2]
        throughput_scale, rlf_scale = self.normalisers()
        cost = np.zeros(values.shape[0], dtype=np.float64)
        if self.throughput_min is not None:
            cost += np.maximum(
                0.0, (float(self.throughput_min) - throughput) / throughput_scale
            )
        if self.rlf_max is not None:
            cost += np.maximum(
                0.0, (rlf - float(self.rlf_max)) / rlf_scale
            )
        return cost

    def is_feasible(self, raw_components: Any) -> np.ndarray:
        """Boolean per-transition feasibility under the two thresholds."""

        values = np.asarray(raw_components, dtype=np.float64)
        feasible = np.ones(values.shape[0], dtype=bool)
        if self.throughput_min is not None:
            feasible &= values[:, 0] >= float(self.throughput_min)
        if self.rlf_max is not None:
            feasible &= values[:, 2] <= float(self.rlf_max)
        return feasible

    def summary(self, raw_components: Any) -> Dict[str, Any]:
        """Diagnostics used in training summaries and the Pareto report."""

        values = np.asarray(raw_components, dtype=np.float64)
        cost = self.cost(values)
        feasible = self.is_feasible(values)
        return {
            "enabled": self.enabled,
            "throughput_min": self.throughput_min,
            "rlf_max": self.rlf_max,
            "throughput_mean": float(values[:, 0].mean()) if values.size else None,
            "rlf_mean": float(values[:, 2].mean()) if values.size else None,
            "rlf_max_observed": float(values[:, 2].max()) if values.size else None,
            "mean_cost": float(cost.mean()) if cost.size else None,
            "violation_rate": (
                float((~feasible).mean()) if feasible.size else None
            ),
            "feasible_fraction": (
                float(feasible.mean()) if feasible.size else None
            ),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "throughput_min": self.throughput_min,
            "rlf_max": self.rlf_max,
            "throughput_scale": self.throughput_scale,
            "rlf_scale": self.rlf_scale,
            "lambda_lr": self.lambda_lr,
            "lambda_init": self.lambda_init,
            "lambda_max": self.lambda_max,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QoSConstraint":
        known = {
            "throughput_min",
            "rlf_max",
            "throughput_scale",
            "rlf_scale",
            "lambda_lr",
            "lambda_init",
            "lambda_max",
            "tolerance",
        }
        return cls(**{key: value for key, value in values.items() if key in known})


class LagrangeMultiplier:
    """Dual variable of the E1 constraint with bounded dual ascent."""

    def __init__(self, constraint: QoSConstraint) -> None:
        self.constraint = constraint
        self.value = float(constraint.lambda_init)
        self.history: list[float] = [self.value]

    def update(self, mean_cost: float) -> float:
        mean_cost = float(mean_cost)
        if not math.isfinite(mean_cost):
            raise ValueError("mean_cost must be finite.")
        updated = self.value + float(self.constraint.lambda_lr) * (
            mean_cost - float(self.constraint.tolerance)
        )
        self.value = float(
            min(float(self.constraint.lambda_max), max(0.0, updated))
        )
        self.history.append(self.value)
        return self.value

    def penalty(self, cost: Any) -> Any:
        return float(self.value) * cost

    def state_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "history": list(self.history),
            "constraint": self.constraint.as_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return
        value = state.get("value")
        if value is not None:
            self.value = float(value)
        history = state.get("history")
        if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            self.history = [float(item) for item in history]


def infer_throughput_reference(
    raw_components: Any,
    episode_ids: Any,
    run_policies: Optional[Sequence[str]] = None,
    *,
    policy: str = "always_on",
    fallback_quantile: float = 0.9,
) -> Tuple[float, str]:
    """Infer the throughput reference used by ``throughput_fraction``.

    When the dataset records the policy of every episode (the campaign
    manifest does), the reference is the mean raw throughput of the
    ``always_on`` episodes.  Otherwise the function falls back to a high
    quantile of the whole dataset and says so in the returned source string.
    """

    values = np.asarray(raw_components, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("raw_components must have shape [N, 5].")
    episodes = np.asarray(episode_ids).reshape(-1)
    if episodes.size != values.shape[0]:
        raise ValueError("episode_ids and raw_components have different lengths.")

    if run_policies:
        policies = list(run_policies)
        mask = np.zeros(values.shape[0], dtype=bool)
        for episode in np.unique(episodes):
            index = int(episode)
            if 0 <= index < len(policies) and policies[index] == policy:
                mask |= episodes == episode
        if mask.any():
            return float(values[mask, 0].mean()), f"mean of {policy} episodes"

    quantile = float(np.quantile(values[:, 0], float(fallback_quantile)))
    return quantile, (
        f"q{fallback_quantile:g} of all episodes "
        "(no always_on policy labels in dataset metadata)"
    )


def resolve_constraint(
    constraint: Optional[QoSConstraint] = None,
    **values: Any,
) -> Optional[QoSConstraint]:
    """Build a constraint from a dataclass/mapping plus keyword overrides."""

    if constraint is None:
        merged: Dict[str, Any] = {}
    elif isinstance(constraint, QoSConstraint):
        merged = constraint.as_dict()
    elif isinstance(constraint, Mapping):
        merged = dict(constraint)
    else:
        raise TypeError(
            "constraint must be QoSConstraint, mapping or None, got "
            f"{type(constraint).__name__}."
        )
    merged.update({key: value for key, value in values.items() if value is not None})
    resolved = QoSConstraint.from_mapping(merged)
    return resolved if resolved.enabled else None


__all__ = [
    "QoSConstraint",
    "LagrangeMultiplier",
    "infer_throughput_reference",
    "resolve_constraint",
]
