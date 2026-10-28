"""Paper configuration for the DRL agents (Frente C).

This module is intentionally dependency-free (standard library only) so that
``import nsoran.drl`` and the CLI ``--help`` work even when numpy, torch or
Stable Baselines3 are not installed.

The numbers below come from Table I of the paper (see ``Tradução.md``,
Section IV): reward weights ``w1..w4`` and the per-component quantile
normalization.  The paper states that the PPO entropy coefficient was tuned
in ``[0.001, 0.003]`` but does not publish the exact value per agent; the
values in :data:`PAPER_AGENTS` are a deterministic, documented choice inside
that range (more energy-saving-oriented agents get slightly more entropy).

The module also hosts the E1/CMDP constraint layer of
``docs/plano_tecnico/cmdp_design.md``: an optional QoS restriction
(``constraint_mode="none" | "lagrangian" | "barrier"``, default ``"none"``
keeping the paper's Eq. (2) untouched) with the SLA thresholds ``tau_sla``/
``epsilon`` and the dual multiplier support in :func:`compute_reward`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared problem constants (mirror PAPER_IMPLEMENTATION.md / es_env.py)
# ---------------------------------------------------------------------------

STATE_DIM = 85
N_CELLS = 7
N_ACTIONS = 128
CELL_IDS = (2, 3, 4, 5, 6, 7, 8)
KPMS_PER_CELL = 12
GLOBAL_BS_ON_INDEX = STATE_DIM - 1  # index 84

#: Order of the 12 per-cell KPMs in the 85-dim state vector.
CELL_KPM_NAMES = (
    "throughput_dl",
    "energy",
    "ee_ratio",
    "rlf_count",
    "rlf_percentage",
    "prb_used",
    "prb_percentage",
    "tb_64qam",
    "tb_64qam_ratio",
    "mac_pdu_64qam",
    "es_on_cost",
    "cell_state",
)

#: Convenience index map for per-cell KPMs.
CELL_KPM_INDEX: Dict[str, int] = {
    name: index for index, name in enumerate(CELL_KPM_NAMES)
}

#: Human-readable names of the 85 state features, in the exact order emitted
#: by ``environments.es_env.build_observation`` (cell-major, then ``bs_on``).
STATE_FEATURE_NAMES = tuple(
    f"{name}_cell_{cell}" for cell in CELL_IDS for name in CELL_KPM_NAMES
) + ("bs_on",)

#: Legacy E5 ablation (78 dims): ``mac_pdu_64qam`` is assigned the same ns-3
#: counter as ``tb_64qam``, so removing it from every cell drops the one
#: bit-identical duplicate column.  Kept for compatibility; the current
#: compact vector of the audit is ``compact`` (70 dims, below).
E5_DROPPED_KPMS = ("mac_pdu_64qam",)
E5_KEEP_INDICES = tuple(
    index
    for index, feature in enumerate(STATE_FEATURE_NAMES)
    if feature
    not in {
        f"{kpi}_cell_{cell}"
        for kpi in E5_DROPPED_KPMS
        for cell in CELL_IDS
    }
)
STATE_DIM_COMPACT = len(E5_KEEP_INDICES)

#: E5 compact vector of ``docs/plano_tecnico/kpm_audit.md`` (§3, "Tabela de
#: cortes"): drops exactly the 15 columns whose redundancy is exact --
#: 7 x ``mac_pdu_64qam`` (bit-identical copy of ``tb_64qam``), 7 x
#: ``prb_percentage`` (affine ``prb_used / 139 * 100`` with proportional
#: min-max ranges, hence identical after normalisation) and the global
#: ``bs_on`` (exactly the sum of the 7 ``cell_state`` columns).  85 -> 70
#: with full linear rank (audit §2).  The reward (Eq. 2) is computed from
#: the raw KPM totals and is untouched by this cut.
E5_COMPACT_DROPPED_KPMS = ("mac_pdu_64qam", "prb_percentage")
E5_COMPACT_DROPPED_GLOBAL = ("bs_on",)
_E5_COMPACT_DROPPED_NAMES = (
    {
        f"{kpi}_cell_{cell}"
        for kpi in E5_COMPACT_DROPPED_KPMS
        for cell in CELL_IDS
    }
    | set(E5_COMPACT_DROPPED_GLOBAL)
)
E5_COMPACT_KEEP_INDICES = tuple(
    index
    for index, feature in enumerate(STATE_FEATURE_NAMES)
    if feature not in _E5_COMPACT_DROPPED_NAMES
)
STATE_DIM_COMPACT70 = len(E5_COMPACT_KEEP_INDICES)

#: Supported state layouts.  ``full`` / ``paper85`` are the paper's
#: 85-dim state (the default everywhere); ``compact`` is the E5 compact
#: vector of ``kpm_audit.md`` (70 dims); ``compact78`` is the earlier
#: 78-dim ablation, kept for compatibility.
STATE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {
        "obs_dim": STATE_DIM,
        "indices": tuple(range(STATE_DIM)),
        "dropped_kpms": (),
        "dropped_global": (),
    },
    # Explicit alias of ``full`` so the flag value used by the
    # environment/CLI ("paper85") names the paper contract directly.
    "paper85": {
        "obs_dim": STATE_DIM,
        "indices": tuple(range(STATE_DIM)),
        "dropped_kpms": (),
        "dropped_global": (),
    },
    "compact": {
        "obs_dim": STATE_DIM_COMPACT70,
        "indices": E5_COMPACT_KEEP_INDICES,
        "dropped_kpms": E5_COMPACT_DROPPED_KPMS,
        "dropped_global": E5_COMPACT_DROPPED_GLOBAL,
    },
    "compact78": {
        "obs_dim": STATE_DIM_COMPACT,
        "indices": E5_KEEP_INDICES,
        "dropped_kpms": E5_DROPPED_KPMS,
        "dropped_global": (),
    },
}


def state_variant_indices(variant: str = "full") -> tuple[int, ...]:
    """Return the indices of ``STATE_FEATURE_NAMES`` kept by ``variant``."""

    try:
        return tuple(STATE_VARIANTS[str(variant)]["indices"])
    except KeyError as exc:
        raise ValueError(
            f"Unknown state variant {variant!r}; expected one of "
            f"{', '.join(STATE_VARIANTS)}."
        ) from exc


def state_feature_names_for_variant(variant: str = "full") -> tuple[str, ...]:
    """Return the feature names of a state variant, in observation order."""

    return tuple(STATE_FEATURE_NAMES[index] for index in state_variant_indices(variant))


def state_feature_names_for_dim(obs_dim: int) -> tuple[str, ...]:
    """Return the known feature names for an observation width.

    The widths used by this project are 85 (paper state), 70 (E5 compact
    state of ``kpm_audit.md``) and 78 (legacy E5 ablation).  Unknown widths
    raise instead of guessing.
    """

    obs_dim = int(obs_dim)
    for variant in STATE_VARIANTS.values():
        if int(variant["obs_dim"]) == obs_dim:
            return tuple(STATE_FEATURE_NAMES[index] for index in variant["indices"])
    raise ValueError(
        f"No state-feature layout is registered for obs_dim={obs_dim}; "
        f"known widths: {[int(v['obs_dim']) for v in STATE_VARIANTS.values()]}."
    )


def select_state_features(observation: Any, variant: str = "full") -> Any:
    """Select the features of ``variant`` from a full 85-dim observation.

    Supports NumPy arrays (any leading dimensions) and plain Python
    sequences.  It is intentionally dependency-free so it can be used by the
    dataset converter without importing NumPy at module load time.
    """

    indices = list(state_variant_indices(variant))
    if hasattr(observation, "shape"):
        return observation[..., indices]
    return [observation[index] for index in indices]

#: Indices of the five reward components (optional ``reward_components``
#: array in the offline dataset, shape ``[N, 5]``).
REWARD_COMPONENT_NAMES = (
    "throughput_norm",
    "energy_norm",
    "rlf_norm",
    "es_on_cost_norm",
    "bs_on_norm",
)

#: Default optimisation budget / early stopping, from Section IV.
DEFAULT_TOTAL_TIMESTEPS = 1_200_000
DEFAULT_EPISODE_LENGTH = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_EARLY_STOP_PATIENCE = 4
DEFAULT_MIN_TIMESTEPS = 10_000
DEFAULT_EVAL_FREQ = 10_000

NORMALIZATIONS = ("normal", "uniform")


# ---------------------------------------------------------------------------
# Agent configurations
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Reward / algorithm configuration for one agent of Table I."""

    name: str
    algo: str  # "ppo" or "rem_dqn"
    w1: float
    w2: float
    w3: float
    w4: float
    normalization: str
    ent_coef: Optional[float] = None
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algo not in ("ppo", "rem_dqn"):
            raise ValueError(
                f"Unsupported algorithm {self.algo!r}; expected 'ppo' or 'rem_dqn'."
            )
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization {self.normalization!r}; "
                f"expected one of {NORMALIZATIONS}."
            )
        for weight_name in ("w1", "w2", "w3", "w4"):
            weight = getattr(self, weight_name)
            if weight < 0:
                raise ValueError(f"{self.name}: {weight_name} must be >= 0.")
        total = self.w1 + self.w2 + self.w3 + self.w4
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{self.name}: reward weights must sum to 1, got {total!r}."
            )
        if self.ent_coef is not None and not 0.001 <= self.ent_coef <= 0.003:
            raise ValueError(
                f"{self.name}: ent_coef must be in [0.001, 0.003] per the paper, "
                f"got {self.ent_coef!r}."
            )

    @property
    def weights(self) -> Dict[str, float]:
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3, "w4": self.w4}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


PAPER_AGENTS: Dict[str, AgentConfig] = {
    "PPO-1": AgentConfig(
        name="PPO-1",
        algo="ppo",
        w1=0.51,
        w2=0.19,
        w3=0.20,
        w4=0.10,
        normalization="normal",
        ent_coef=0.001,
        description="Throughput-first balanced PPO agent (Table I).",
    ),
    "PPO-2": AgentConfig(
        name="PPO-2",
        algo="ppo",
        w1=0.40,
        w2=0.40,
        w3=0.10,
        w4=0.10,
        normalization="uniform",
        ent_coef=0.002,
        description="Energy-saving-focused PPO agent (Table I).",
    ),
    "PPO-3": AgentConfig(
        name="PPO-3",
        algo="ppo",
        w1=0.20,
        w2=0.40,
        w3=0.20,
        w4=0.20,
        normalization="uniform",
        ent_coef=0.003,
        description="Aggressive energy-saving PPO agent (Table I).",
    ),
    "PPO-4": AgentConfig(
        name="PPO-4",
        algo="ppo",
        w1=0.40,
        w2=0.32,
        w3=0.18,
        w4=0.10,
        normalization="uniform",
        ent_coef=0.002,
        description="High energy saving with moderate RLF penalty (Table I).",
    ),
    "PPO-5": AgentConfig(
        name="PPO-5",
        algo="ppo",
        w1=0.45,
        w2=0.20,
        w3=0.25,
        w4=0.10,
        normalization="normal",
        ent_coef=0.0015,
        description="Coverage-oriented PPO agent (Table I).",
    ),
    "DQN": AgentConfig(
        name="DQN",
        algo="rem_dqn",
        w1=0.40,
        w2=0.40,
        w3=0.10,
        w4=0.10,
        normalization="uniform",
        ent_coef=None,
        description="REM-DQN + CQL offline agent (Table I).",
    ),
}

AGENT_NAMES = tuple(PAPER_AGENTS)

_AGENT_LOOKUP = {name.upper(): name for name in PAPER_AGENTS}


def get_agent_config(name: str) -> AgentConfig:
    """Return the configuration for ``name`` (case-insensitive)."""

    try:
        canonical = _AGENT_LOOKUP[str(name).upper()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown agent {name!r}. Available agents: {', '.join(AGENT_NAMES)}."
        ) from exc
    return PAPER_AGENTS[canonical]


def resolve_agent_config(config: Any) -> AgentConfig:
    """Accept an agent name, an :class:`AgentConfig` or a mapping."""

    if isinstance(config, AgentConfig):
        return config
    if isinstance(config, str):
        return get_agent_config(config)
    if isinstance(config, dict):
        name = config.get("name") or config.get("agent")
        if name:
            base = get_agent_config(name)
            merged = base.as_dict()
            merged.update(config)
            return AgentConfig(**merged)
        return AgentConfig(**config)
    raise TypeError(
        "agent config must be a name, AgentConfig or mapping, "
        f"got {type(config).__name__}."
    )


# ---------------------------------------------------------------------------
# Reward and action helpers
# ---------------------------------------------------------------------------


def compute_reward(
    throughput_norm: Any,
    energy_norm: Any,
    rlf_norm: Any,
    es_on_cost_norm: Any,
    bs_on_norm: Any,
    config: Any,
    *,
    dual_lambda: float = 0.0,
    qos_cost: float = 0.0,
    gate_open: float = 1.0,
) -> Any:
    """Eq. (2) of the paper, with the optional E1/CMDP layer on top.

    ``r = w1*throughput - w2*energy - w3*rlf - w4*es_on_cost - w2*bs_on``

    With the defaults (``dual_lambda=0.0``, ``gate_open=1.0``) the result is
    exactly the paper's Eq. (2).  Two optional channels:

    * ``dual_lambda`` / ``qos_cost`` -- Lagrangian relaxation (Eq. 4.1 of
      ``docs/plano_tecnico/cmdp_design.md``): subtracts ``dual_lambda *
      qos_cost``, where the per-indication hinge cost comes from
      :meth:`CmdpConfig.qos_cost` (or ``QoSConstraint.cost`` on the offline
      dataset).  The dual multiplier is advanced by
      :meth:`CmdpConfig.update_dual`.
    * ``gate_open`` -- barrier/gate formulation (Eq. 5.1'): factor ``g_t`` in
      ``[0, 1]`` that gates BOTH energy-saving channels (``energy`` and
      ``bs_on``, both weighted by ``w2``); ``1.0`` keeps Eq. (2) intact.  Use
      :meth:`CmdpConfig.gate_factor` to compute it from the raw moving-average
      throughput, or :func:`compute_gated_reward` directly.

    Inputs may be Python floats or numpy arrays.  NumPy is intentionally not
    imported here so this module remains dependency-free.  ``gate_open`` must
    be a scalar; the defaults below make the non-CMDP path bit-identical to
    the original Eq. (2).
    """

    agent = resolve_agent_config(config)
    if gate_open != 1.0:
        # Eq. (5.1'): os dois canais de economia (gamma_hat e B_hat) saem da
        # recompensa quando a porta fecha; throughput/RLF/custo ficam.
        reward = (
            agent.w1 * throughput_norm
            - agent.w3 * rlf_norm
            - agent.w4 * es_on_cost_norm
            - agent.w2 * (energy_norm + bs_on_norm) * gate_open
        )
    else:
        reward = (
            agent.w1 * throughput_norm
            - agent.w2 * energy_norm
            - agent.w3 * rlf_norm
            - agent.w4 * es_on_cost_norm
            - agent.w2 * bs_on_norm
        )
    if dual_lambda != 0.0:
        reward = reward - dual_lambda * qos_cost
    return reward


# ---------------------------------------------------------------------------
# E1 / CMDP: camada de restrição de QoS (docs/plano_tecnico/cmdp_design.md)
# ---------------------------------------------------------------------------

#: Modos de restrição do E1.  O default ``"none"`` preserva exatamente a
#: Eq. (2) do paper — a formulação CMDP só age quando ativada em explicito.
CONSTRAINT_MODES = ("none", "lagrangian", "barrier")

#: O design chama a formulação (b) de "gate" (§5/§6.1); o nome é aceito como
#: apelido de ``"barrier"``.
CONSTRAINT_MODE_ALIASES: Dict[str, str] = {"gate": "barrier"}


def normalize_constraint_mode(mode: Any) -> str:
    """Validate and normalise the constraint mode (``"gate"`` == ``"barrier"``)."""

    value = CONSTRAINT_MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
    if value not in CONSTRAINT_MODES:
        raise ValueError(
            f"constraint_mode inválido: {mode!r}; esperado um de "
            f"{', '.join(CONSTRAINT_MODES)} ('gate' é aceito como apelido de "
            "'barrier')."
        )
    return value


@dataclass
class CmdpConfig:
    """Parâmetros da formulação CMDP do E1 (design técnico, §3-§5).

    O custo por indicação é o hinge da Eq. (3.2), em unidades cruas do ns-3::

        c = [tau_sla - throughput]_+ / S_rho + [rlf - epsilon]_+ / S_varsigma

    Modos (``constraint_mode``):

    * ``"none"`` (default): sem restrição; a recompensa é a Eq. (2).
    * ``"lagrangian"``: r̃ = r − λ·c (Eq. 4.1) com ascensão dual λ ←
      clip(λ + η_λ·(c̄ − δ), 0, λ_max) (Eq. 4.2, :meth:`update_dual`).
      Exige ``tau_sla`` e/ou ``epsilon``.
    * ``"barrier"`` (alias ``"gate"``): Eq. (5.1′) — os dois canais de
      economia (energia e bs_on, ambos ponderados por w2) são portados por
      ``g_t`` calculado da média móvel do throughput cru
      (:meth:`gate_factor`).  Exige ``tau_sla``.

    Notas do design (rev. 1, §8): ``lambda_max`` default 10 (§8.1-A4, antes
    100) e ``tolerance`` (δ) default 0.05 (§2.1/§4.1).  A porta suave é a
    variante padrão de (b): passe ``gate_temperature`` ≈ 0.05·μ_AO
    (≈ 3.834 bytes para μ_AO = 76.686 dos dados reais de
    ``baselines_d1d3.csv``); ``None`` mantém a porta dura (ablação).
    """

    constraint_mode: str = "none"
    #: τ_SLA: throughput mínimo por indicação (bytes, ``QosFlow.PdcpPduVolumeDL_Filter``).
    tau_sla: Optional[float] = None
    #: ε: máximo de UEs em RLF (SINR < limiar) por indicação.
    epsilon: Optional[float] = None
    #: Escalas S_rho / S_varsigma do hinge (default: max(limiar, 1)).
    throughput_scale: Optional[float] = None
    rlf_scale: Optional[float] = None
    #: Multiplicador dual λ corrente (Eq. 4.1).
    dual_lambda: float = 0.0
    #: η_λ do passo dual (Eq. 4.2); design §4.1 recomenda 0.01.
    lambda_lr: float = 0.01
    #: Teto da projeção dual (design §8.1-A4: 10).
    lambda_max: float = 10.0
    #: δ: custo médio admissível (design §2.1: 0.05).
    tolerance: float = 0.05
    #: W: janela móvel da porta em indicações (design §5.1: 10 ≈ 1 s).
    gate_window: int = 10
    #: Temperatura T da porta suave; None = porta dura (ablação, §8.2-B1).
    gate_temperature: Optional[float] = None

    _KNOWN_KEYS = frozenset(
        {
            "constraint_mode",
            "tau_sla",
            "epsilon",
            "throughput_scale",
            "rlf_scale",
            "dual_lambda",
            "lambda_lr",
            "lambda_max",
            "tolerance",
            "gate_window",
            "gate_temperature",
        }
    )

    def __post_init__(self) -> None:
        self.constraint_mode = normalize_constraint_mode(self.constraint_mode)
        self.gate_window = int(self.gate_window)
        if self.constraint_mode == "barrier" and self.tau_sla is None:
            raise ValueError(
                "constraint_mode='barrier' exige tau_sla (a porta da Eq. 5.1′ "
                "é condicionada ao throughput)."
            )
        if (
            self.constraint_mode == "lagrangian"
            and self.tau_sla is None
            and self.epsilon is None
        ):
            raise ValueError(
                "constraint_mode='lagrangian' exige tau_sla e/ou epsilon "
                "(unidades cruas do ns-3)."
            )
        for name in ("tau_sla", "epsilon"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} deve ser >= 0 quando fornecido.")
        for name in ("throughput_scale", "rlf_scale"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} deve ser > 0 quando fornecido.")
        if self.lambda_lr < 0.0:
            raise ValueError("lambda_lr deve ser >= 0.")
        if self.lambda_max < 0.0:
            raise ValueError("lambda_max deve ser >= 0.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance deve ser >= 0.")
        if not 0.0 <= self.dual_lambda <= self.lambda_max:
            raise ValueError(
                f"dual_lambda deve estar em [0, lambda_max={self.lambda_max}], "
                f"recebeu {self.dual_lambda!r}."
            )
        if self.gate_window < 1:
            raise ValueError("gate_window deve ser >= 1 indicação.")
        if self.gate_temperature is not None and float(self.gate_temperature) <= 0.0:
            raise ValueError("gate_temperature deve ser > 0 quando fornecido.")

    @property
    def enabled(self) -> bool:
        """``True`` quando a restrição CMDP está ativa (modo != ``"none"``)."""

        return self.constraint_mode != "none"

    def normalisers(self) -> Tuple[float, float]:
        """Escalas ``S_rho``/``S_varsigma`` do hinge, com o mesmo fallback do
        ``QoSConstraint`` de ``nsoran.drl.qos`` (``max(limiar, 1)``)."""

        throughput_scale = (
            float(self.throughput_scale)
            if self.throughput_scale is not None
            else max(float(self.tau_sla or 0.0), 1.0)
        )
        rlf_scale = (
            float(self.rlf_scale)
            if self.rlf_scale is not None
            else max(float(self.epsilon or 0.0), 1.0)
        )
        return throughput_scale, rlf_scale

    def qos_cost(self, raw_throughput: Any, raw_rlf: Any = 0.0) -> Any:
        """Custo hinge por indicação, Eq. (3.2) do design.

        Aceita escalares ou sequências/arrays (aritmética duck-typed, sem
        NumPy); a versão veturada em lote do dataset continua sendo
        ``QoSConstraint.cost`` em ``nsoran.drl.qos``.  Sem limiares
        configurados devolve 0.0.
        """

        throughput_scale, rlf_scale = self.normalisers()
        cost: Any = 0.0
        if self.tau_sla is not None:
            deficit = (float(self.tau_sla) - raw_throughput) / throughput_scale
            cost = cost + (deficit > 0.0) * deficit
        if self.epsilon is not None:
            excess = (raw_rlf - float(self.epsilon)) / rlf_scale
            cost = cost + (excess > 0.0) * excess
        return cost

    def gate_factor(self, throughput_window_mean: float) -> float:
        """Valor da porta ``g_t`` da Eq. (5.1′) para a média móvel do throughput cru.

        Porta dura (``gate_temperature=None``, ablação): 1.0 se a média >=
        ``tau_sla``, 0.0 caso contrário.  Porta suave (variante padrão de (b),
        §8.2-B1): ``sigmoid((média − tau_sla) / T)``, com ``g_t(τ) = 0.5``.
        """

        if self.tau_sla is None:
            raise ValueError(
                "gate_factor exige tau_sla (use constraint_mode='barrier')."
            )
        offset = float(throughput_window_mean) - float(self.tau_sla)
        if self.gate_temperature is None:
            return 1.0 if offset >= 0.0 else 0.0
        x = offset / float(self.gate_temperature)
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def update_dual(self, mean_cost: float) -> float:
        """Passo de ascensão dual com projeção, Eq. (4.2)::

            λ ← clip(λ + lambda_lr·(c̄ − tolerance), 0, lambda_max)

        Retorna o novo λ.  (No treino offline a média do lote é constante do
        comportamento — design §8.1-A1/A2/A3: medir c̄ do dataset antes e
        preferir grade fixa de λ / crítico de custo ao ciclo 1:1.)
        """

        mean_cost = float(mean_cost)
        if not math.isfinite(mean_cost):
            raise ValueError("mean_cost deve ser finito.")
        updated = self.dual_lambda + float(self.lambda_lr) * (
            mean_cost - float(self.tolerance)
        )
        self.dual_lambda = float(min(float(self.lambda_max), max(0.0, updated)))
        return self.dual_lambda

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraint_mode": self.constraint_mode,
            "tau_sla": self.tau_sla,
            "epsilon": self.epsilon,
            "throughput_scale": self.throughput_scale,
            "rlf_scale": self.rlf_scale,
            "dual_lambda": self.dual_lambda,
            "lambda_lr": self.lambda_lr,
            "lambda_max": self.lambda_max,
            "tolerance": self.tolerance,
            "gate_window": self.gate_window,
            "gate_temperature": self.gate_temperature,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CmdpConfig":
        """Build a config from a mapping, ignoring unknown keys."""

        return cls(
            **{key: value for key, value in values.items() if key in cls._KNOWN_KEYS}
        )


def compute_gated_reward(
    throughput_norm: Any,
    energy_norm: Any,
    rlf_norm: Any,
    es_on_cost_norm: Any,
    bs_on_norm: Any,
    config: Any,
    gate_open: float,
) -> Any:
    """Eq. (5.1′) — recompensa do modo barreira (design §5/§8.2-B3).

    ``r = w1*ρ̂ - w3*ς̂ - w4*δ̂ - w2*(γ̂ + B̂)*g_t`` com a porta ``g_t`` sobre os
    DOIS canais de economia.  ``gate_open`` vem de
    :meth:`CmdpConfig.gate_factor` (1.0 ⇒ igual à Eq. (2)).
    """

    return compute_reward(
        throughput_norm,
        energy_norm,
        rlf_norm,
        es_on_cost_norm,
        bs_on_norm,
        config,
        gate_open=gate_open,
    )


def action_to_cells(action: int) -> Dict[int, int]:
    """Decode a ``0..127`` action into ``{cell_id: 1=ON, 0=OFF}``.

    Bit 0 (LSB) maps to cell 2 and bit 6 (MSB) maps to cell 8, matching the
    contract in ``PAPER_IMPLEMENTATION.md`` (action 127 = all cells ON).
    """

    action = int(action)
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action must be in [0, {N_ACTIONS - 1}], got {action}.")
    return {cell: (action >> bit) & 1 for bit, cell in enumerate(CELL_IDS)}


def cells_to_action(states: Dict[int, int]) -> int:
    """Encode ``{cell_id: state}`` into a ``0..127`` action (missing = OFF)."""

    action = 0
    for bit, cell in enumerate(CELL_IDS):
        value = int(states.get(cell, 0))
        if value not in (0, 1):
            raise ValueError(f"cell {cell} state must be 0 or 1, got {value}.")
        action |= value << bit
    return action


def action_to_bitstring(action: int) -> str:
    """Return the 7-bit string in cell order 2..8 (left to right)."""

    states = action_to_cells(action)
    return "".join(str(states[cell]) for cell in CELL_IDS)


__all__ = [
    "STATE_DIM",
    "N_CELLS",
    "N_ACTIONS",
    "CELL_IDS",
    "KPMS_PER_CELL",
    "GLOBAL_BS_ON_INDEX",
    "CELL_KPM_NAMES",
    "CELL_KPM_INDEX",
    "REWARD_COMPONENT_NAMES",
    "DEFAULT_TOTAL_TIMESTEPS",
    "DEFAULT_EPISODE_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EARLY_STOP_PATIENCE",
    "DEFAULT_MIN_TIMESTEPS",
    "DEFAULT_EVAL_FREQ",
    "NORMALIZATIONS",
    "AgentConfig",
    "PAPER_AGENTS",
    "AGENT_NAMES",
    "get_agent_config",
    "resolve_agent_config",
    "compute_reward",
    "compute_gated_reward",
    "CONSTRAINT_MODES",
    "CONSTRAINT_MODE_ALIASES",
    "normalize_constraint_mode",
    "CmdpConfig",
    "action_to_cells",
    "cells_to_action",
    "action_to_bitstring",
]
