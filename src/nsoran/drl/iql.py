"""Discrete Implicit Q-Learning (IQL) for offline cell sleep control.

Decision D4 of the approved plan: the five PPO rows of Table I are trained
with a genuine offline RL algorithm instead of replaying PPO on a static
dataset (online PPO on a frozen replay buffer degenerates into reward-weighted
behaviour cloning).  The DQN row keeps the paper's REM-DQN + CQL agent.

The implementation follows Kostrikov, Nair & Levine, *Offline Reinforcement
Learning with Implicit Q-Learning* (NeurIPS 2021), specialised to the finite
action space of this problem:

* two Q-networks ``Q1``/``Q2`` trained with the usual Bellman error against a
  slow target value network;
* a value network ``V`` trained with the expectile loss
  ``|tau - 1[Q_min - V < 0]| * (Q_min - V)^2``, which approximates the
  ``tau``-quantile of the Q distribution without querying unseen actions;
* a policy network trained with advantage-weighted regression (AWR),
  ``exp(beta * (Q_min - V)) * log pi(a|s)``, clipped at ``max_weight``.

Training is fully offline: mini-batches come from an
:class:`~nsoran.drl.offline.OfflineDataset` and no ns-3 process is involved.
Importing this module requires PyTorch; ``import nsoran.drl`` does not import
it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:  # optional dependency group: drl
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.iql requires PyTorch and numpy. Install the DRL extras "
        "with `pip install -e '.[drl]'` or `pip install -r requirements-drl.txt`."
    ) from exc

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EVAL_FREQ,
    DEFAULT_MIN_TIMESTEPS,
    DEFAULT_TOTAL_TIMESTEPS,
    N_ACTIONS,
    STATE_DIM,
)
from .offline import OfflineDataset, split_dataset
from .qos import LagrangeMultiplier, QoSConstraint
from .rem_dqn import OfflineReplayBuffer, _resolve_device, _torch_load


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IQLConfig:
    """Hyper-parameters of the discrete IQL agent.

    ``tau`` is the expectile of the value loss (``0.5`` is the mean, ``>0.5``
    is optimistic).  ``beta`` is the inverse temperature of the AWR policy
    loss and ``max_weight`` clips its importance weight.  ``target_update`` is
    the soft-update rate of the target value network.  ``batch_size`` is kept
    here so checkpoints are self-contained.
    """

    obs_dim: int = STATE_DIM
    n_actions: int = N_ACTIONS
    hidden_dims: Tuple[int, ...] = (256, 256)

    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.7
    beta: float = 1.0
    max_weight: float = 100.0
    target_update: float = 0.005
    grad_clip: float = 10.0

    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"
    seed: Optional[int] = None

    # E1: explicit QoS constraint (see ``nsoran.drl.qos``).  ``None`` on both
    # thresholds disables the constrained objective and reproduces the
    # unconstrained IQL baseline.
    qos_throughput_min: Optional[float] = None
    qos_rlf_max: Optional[float] = None
    qos_throughput_scale: Optional[float] = None
    qos_rlf_scale: Optional[float] = None
    qos_lambda_lr: float = 0.01
    qos_lambda_init: float = 0.0
    qos_lambda_max: float = 100.0
    qos_tolerance: float = 0.0

    # State layout metadata used by the evaluator to slice live observations.
    state_variant: str = "full"

    def __post_init__(self) -> None:
        self.obs_dim = int(self.obs_dim)
        self.n_actions = int(self.n_actions)
        self.hidden_dims = tuple(int(d) for d in self.hidden_dims)
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes.")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be in (0, 1).")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.max_weight <= 0.0:
            raise ValueError("max_weight must be positive.")
        if not 0.0 < self.target_update <= 1.0:
            raise ValueError("target_update must be in (0, 1].")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in [0, 1].")
        # E1 validation is delegated to QoSConstraint so the training CLI and
        # the evaluator reject the same invalid thresholds.
        self.qos_constraint()

    def qos_constraint(self) -> Optional[QoSConstraint]:
        """Return the E1 constraint, or ``None`` when QoS is disabled."""

        constraint = QoSConstraint(
            throughput_min=self.qos_throughput_min,
            rlf_max=self.qos_rlf_max,
            throughput_scale=self.qos_throughput_scale,
            rlf_scale=self.qos_rlf_scale,
            lambda_lr=self.qos_lambda_lr,
            lambda_init=self.qos_lambda_init,
            lambda_max=self.qos_lambda_max,
            tolerance=self.qos_tolerance,
        )
        return constraint if constraint.enabled else None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "IQLConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(previous, int(width)))
        layers.append(nn.ReLU())
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class IQLAgent:
    """Discrete Implicit Q-Learning agent (offline training only)."""

    def __init__(
        self,
        config: Optional[Union[IQLConfig, Mapping[str, Any]]] = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            resolved = IQLConfig(**overrides)
        elif isinstance(config, IQLConfig):
            resolved = replace(config, **overrides) if overrides else config
        elif isinstance(config, Mapping):
            values = dict(config)
            values.update(overrides)
            resolved = IQLConfig.from_dict(values)
        else:
            raise TypeError(
                "config must be IQLConfig, mapping or None, got "
                f"{type(config).__name__}."
            )
        self.config = resolved
        self.device = _resolve_device(resolved.device)
        if resolved.seed is not None:
            np.random.seed(int(resolved.seed))
            torch.manual_seed(int(resolved.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(resolved.seed))

        self.q1 = _mlp(resolved.obs_dim, resolved.hidden_dims, resolved.n_actions).to(
            self.device
        )
        self.q2 = _mlp(resolved.obs_dim, resolved.hidden_dims, resolved.n_actions).to(
            self.device
        )
        self.value = _mlp(resolved.obs_dim, resolved.hidden_dims, 1).to(self.device)
        self.policy = _mlp(
            resolved.obs_dim, resolved.hidden_dims, resolved.n_actions
        ).to(self.device)
        self.value_target = _mlp(
            resolved.obs_dim, resolved.hidden_dims, 1
        ).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())
        self.value_target.eval()
        for parameter in self.value_target.parameters():
            parameter.requires_grad_(False)

        parameters = (
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.value.parameters())
            + list(self.policy.parameters())
        )
        self.optimizer = torch.optim.Adam(parameters, lr=resolved.lr)
        self.global_step = 0
        self._rng = np.random.default_rng(resolved.seed)

        # E1: the Lagrangian multiplier is a learned dual variable, not a
        # fixed reward weight.  It is restored from the checkpoint below.
        self.qos_constraint = resolved.qos_constraint()
        self.lagrange: Optional[LagrangeMultiplier] = (
            LagrangeMultiplier(self.qos_constraint)
            if self.qos_constraint is not None
            else None
        )

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def q_values(
        self, observations: Union[np.ndarray, torch.Tensor], *, target: bool = False
    ) -> torch.Tensor:
        """Return ``min(Q1, Q2)`` for every action (target networks optional)."""

        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        q1 = self.q1(tensor)
        q2 = self.q2(tensor)
        return torch.minimum(q1, q2)

    @torch.no_grad()
    def value_of(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return self.value(tensor).reshape(-1)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        *,
        step: Optional[int] = None,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[int, np.ndarray]:
        """Argmax policy action(s).

        IQL has no exploration schedule: ``step`` is accepted for API symmetry
        with :class:`~nsoran.drl.rem_dqn.REMDQNAgent`, and ``deterministic`` is
        ignored because the policy is deterministic in both modes.
        """

        del step, deterministic
        single = np.asarray(observation).ndim == 1
        observations = np.atleast_2d(np.asarray(observation, dtype=np.float32))
        if observations.shape[1] != self.config.obs_dim:
            raise ValueError(
                f"observation must have {self.config.obs_dim} features, got "
                f"{observations.shape[1]}."
            )
        if rng is None:
            actions = self.policy(
                torch.as_tensor(observations, device=self.device)
            ).argmax(dim=1)
        else:
            # Kept for interface compatibility; the policy is deterministic.
            actions = self.policy(
                torch.as_tensor(observations, device=self.device)
            ).argmax(dim=1)
        values = actions.cpu().numpy().astype(np.int64)
        return int(values[0]) if single else values

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> Tuple[Any, None]:
        """Stable-Baselines-style prediction used by ``evaluate.py``.

        Returns ``(action, None)`` so the evaluation loop can share the PPO
        branch.  Batched observations return a batch of actions.
        """

        del state, episode_start
        single = np.asarray(observation).ndim == 1
        action = self.act(observation, deterministic=deterministic)
        if single:
            return int(action), None
        return np.asarray(action, dtype=np.int64), None

    # -- losses --------------------------------------------------------------

    def _to_tensors(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        required = (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Batch is missing keys: {', '.join(missing)}.")
        observations = torch.as_tensor(
            np.asarray(batch["observations"], dtype=np.float32),
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.asarray(batch["next_observations"], dtype=np.float32),
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(batch["actions"], dtype=np.int64), device=self.device
        ).reshape(-1, 1)
        rewards = torch.as_tensor(
            np.asarray(batch["rewards"], dtype=np.float32), device=self.device
        ).reshape(-1)
        terminals = torch.as_tensor(
            np.asarray(batch["terminals"], dtype=np.float32), device=self.device
        ).reshape(-1)
        tensors = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminals": terminals,
        }
        if self.qos_constraint is not None:
            raw = batch.get("raw_components")
            if raw is None:
                raise KeyError(
                    "E1 QoS training requires batch['raw_components']; convert "
                    "the dataset with examples/collect_real_dataset.py so the "
                    "raw KPM components are stored."
                )
            cost = torch.as_tensor(
                self.qos_constraint.cost(raw),
                dtype=torch.float32,
                device=self.device,
            )
            # The Lagrangian penalty is the only difference between the E1
            # objective and the unconstrained IQL objective.
            tensors["rewards"] = rewards - self.lagrange.penalty(cost)
            tensors["qos_cost"] = cost
        return tensors

    def compute_losses(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute the Q, value and policy losses of one mini-batch.

        Public so the expectile/AWR losses can be unit-tested without a full
        training loop.
        """

        tensors = self._to_tensors(batch)
        observations = tensors["observations"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        next_observations = tensors["next_observations"]
        terminals = tensors["terminals"]

        with torch.no_grad():
            q_next = torch.minimum(
                self.q1(next_observations), self.q2(next_observations)
            )
            v_next = self.value_target(next_observations).reshape(-1)
            bootstrap = rewards + self.config.gamma * (1.0 - terminals) * v_next

        q1_all = self.q1(observations)
        q2_all = self.q2(observations)
        q1 = q1_all.gather(1, actions).reshape(-1)
        q2 = q2_all.gather(1, actions).reshape(-1)

        q_loss = 0.5 * (F.mse_loss(q1, bootstrap) + F.mse_loss(q2, bootstrap))

        # Expectile regression of V on min(Q1, Q2).
        q_min = torch.minimum(q1, q2)
        value = self.value(observations).reshape(-1)
        difference = q_min.detach() - value
        expectile_weight = torch.where(
            difference > 0.0,
            torch.full_like(difference, self.config.tau),
            torch.full_like(difference, 1.0 - self.config.tau),
        )
        value_loss = (expectile_weight * difference.pow(2)).mean()

        # Advantage-weighted regression on the dataset actions.
        advantage = (q_min.detach() - value.detach()).clamp(-50.0, 50.0)
        weights = torch.exp(self.config.beta * advantage).clamp(
            max=self.config.max_weight
        )
        log_probabilities = F.log_softmax(self.policy(observations), dim=1)
        log_probability = log_probabilities.gather(1, actions).reshape(-1)
        policy_loss = -(weights * log_probability).mean()

        total_loss = q_loss + value_loss + policy_loss
        metrics = {
            "q_loss": q_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "total_loss": total_loss,
            "q_min_mean": q_min.mean().detach(),
            "value_mean": value.mean().detach(),
            "advantage_mean": advantage.mean().detach(),
            "weight_mean": weights.mean().detach(),
        }
        if "qos_cost" in tensors:
            metrics["qos_cost_mean"] = tensors["qos_cost"].mean().detach()
            metrics["lagrange_multiplier"] = torch.as_tensor(
                float(self.lagrange.value), device=self.device
            )
        return metrics

    @torch.no_grad()
    def validation_loss(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        """Evaluate the losses without gradient updates."""

        was_training = self.q1.training
        self.q1.eval()
        self.q2.eval()
        self.value.eval()
        self.policy.eval()
        losses = self.compute_losses(batch)
        self.q1.train(was_training)
        self.q2.train(was_training)
        self.value.train(was_training)
        self.policy.train(was_training)
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    # -- optimisation --------------------------------------------------------

    def update(
        self, batch: Mapping[str, Any], step: Optional[int] = None
    ) -> Dict[str, float]:
        """One gradient update on the shared Q/V/policy objective."""

        del step
        self.q1.train()
        self.q2.train()
        self.value.train()
        self.policy.train()
        losses = self.compute_losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            parameters = (
                list(self.q1.parameters())
                + list(self.q2.parameters())
                + list(self.value.parameters())
                + list(self.policy.parameters())
            )
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
        self.optimizer.step()
        self.global_step += 1
        self.soft_update_target(self.config.target_update)

        metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
        # Dual ascent: increase lambda while the batch violates the SLA and
        # decrease it (bounded at zero) once the cost is below tolerance.
        if self.lagrange is not None and "qos_cost_mean" in metrics:
            self.lagrange.update(metrics["qos_cost_mean"])
            metrics["lagrange_multiplier"] = float(self.lagrange.value)
        metrics["global_step"] = float(self.global_step)
        return metrics

    @torch.no_grad()
    def update_target(self) -> None:
        self.value_target.load_state_dict(self.value.state_dict())

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        for target_parameter, online_parameter in zip(
            self.value_target.parameters(), self.value.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    # -- persistence ---------------------------------------------------------

    def save(
        self,
        path: Union[str, os.PathLike],
        *,
        include_optimizer: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "format_version": 1,
            "algo": "iql",
            "config": self.config.as_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "value_target_state_dict": self.value_target.state_dict(),
            "global_step": self.global_step,
        }
        if self.lagrange is not None:
            payload["lagrange_multiplier"] = self.lagrange.state_dict()
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "IQLAgent":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"IQL checkpoint not found: {path}")
        payload = _torch_load(path, map_location or "cpu")
        if not isinstance(payload, Mapping) or "policy_state_dict" not in payload:
            raise ValueError(f"{path} is not an IQL checkpoint.")
        config_values = payload.get("config", {})
        config = IQLConfig.from_dict(config_values)
        agent = cls(config)
        agent.q1.load_state_dict(payload["q1_state_dict"])
        agent.q2.load_state_dict(payload["q2_state_dict"])
        agent.value.load_state_dict(payload["value_state_dict"])
        agent.policy.load_state_dict(payload["policy_state_dict"])
        if "value_target_state_dict" in payload:
            agent.value_target.load_state_dict(payload["value_target_state_dict"])
        else:
            agent.update_target()
        if "optimizer_state_dict" in payload:
            try:
                agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
                for optimizer_state in agent.optimizer.state.values():
                    for key, value in optimizer_state.items():
                        if isinstance(value, torch.Tensor):
                            optimizer_state[key] = value.to(agent.device)
            except (ValueError, KeyError):
                # Inference only needs the network weights.
                pass
        agent.global_step = int(payload.get("global_step", 0))
        if agent.lagrange is not None and "lagrange_multiplier" in payload:
            agent.lagrange.load_state_dict(payload["lagrange_multiplier"])
        return agent


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def train_iql(
    dataset: OfflineDataset,
    *,
    agent_config: Optional[Any] = None,
    iql_config: Optional[Union[IQLConfig, Mapping[str, Any]]] = None,
    output_dir: Union[str, os.PathLike] = "runs/iql",
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    batch_size: Optional[int] = None,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    min_timesteps: int = DEFAULT_MIN_TIMESTEPS,
    val_fraction: float = 0.1,
    seed: int = 0,
    log_freq: int = 1000,
    checkpoint_freq: Optional[int] = None,
    verbose: int = 1,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """Train discrete IQL offline with validation-loss early stopping.

    ``total_timesteps`` is interpreted as the number of gradient updates
    (offline training does not consume environment steps).  ``agent_config``
    is the Table I reward configuration of the wrapped PPO agent; it is
    recorded in the summary for traceability but does not affect training.
    """

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive.")
    if patience <= 0:
        raise ValueError("patience must be positive.")
    if min_timesteps < 0:
        raise ValueError("min_timesteps must be >= 0.")

    dataset.validate()
    config_values: Dict[str, Any] = {}
    if iql_config is not None:
        if isinstance(iql_config, IQLConfig):
            config_values = iql_config.as_dict()
        elif isinstance(iql_config, Mapping):
            config_values = dict(iql_config)
        else:
            raise TypeError("iql_config must be IQLConfig, mapping or None.")
    config_values.update(config_overrides)
    if batch_size is not None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        config_values["batch_size"] = int(batch_size)
    config_values.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    config_values.setdefault("seed", seed)
    config = IQLConfig.from_dict(config_values)
    if config.qos_constraint() is not None and dataset.raw_components is None:
        raise ValueError(
            "E1 QoS training requires dataset.raw_components; convert the "
            "dataset with examples/collect_real_dataset.py (it now stores "
            "both normalised and raw reward components)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = split_dataset(
        dataset, val_fraction=val_fraction, seed=seed
    )
    train_buffer = OfflineReplayBuffer.from_dataset(train_dataset, seed=seed)
    val_buffer = OfflineReplayBuffer.from_dataset(val_dataset, seed=seed + 1)
    validation_batch = val_buffer.sample(
        min(len(val_buffer), max(config.batch_size, 512))
    )

    agent = IQLAgent(config)
    best_validation = float("inf")
    evaluations_without_improvement = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / "best_model.pt"
    final_path = output_dir / "final_model.pt"
    started = time.time()
    stopped_early = False

    if verbose:
        print(
            f"[IQL] training on {len(train_dataset)} transitions "
            f"({train_dataset.num_episodes} episodes), validation on "
            f"{len(val_dataset)} transitions ({val_dataset.num_episodes} episodes)"
        )
        print(f"[IQL] device={agent.device} config={config.as_dict()}")

    for step in range(1, int(total_timesteps) + 1):
        batch = train_buffer.sample(config.batch_size)
        metrics = agent.update(batch, step=step)

        should_evaluate = (
            step >= min_timesteps
            and (step % eval_freq == 0 or step == int(total_timesteps))
        )
        if should_evaluate:
            validation = agent.validation_loss(validation_batch)
            improved = validation["total_loss"] < best_validation - 1e-9
            if improved:
                best_validation = validation["total_loss"]
                evaluations_without_improvement = 0
                agent.save(best_path)
            else:
                evaluations_without_improvement += 1

            record = {
                "step": step,
                "train_q_loss": metrics["q_loss"],
                "train_value_loss": metrics["value_loss"],
                "train_policy_loss": metrics["policy_loss"],
                "train_total_loss": metrics["total_loss"],
                "validation_q_loss": validation["q_loss"],
                "validation_value_loss": validation["value_loss"],
                "validation_policy_loss": validation["policy_loss"],
                "validation_total_loss": validation["total_loss"],
                "best_validation_total_loss": best_validation,
                "evaluations_without_improvement": (
                    evaluations_without_improvement
                ),
            }
            history.append(record)
            if verbose:
                print(
                    f"[IQL] step {step}: val_total={validation['total_loss']:.6f} "
                    f"best={best_validation:.6f} "
                    f"no_improve={evaluations_without_improvement}/{patience}"
                )
            if evaluations_without_improvement >= patience:
                stopped_early = True
                if verbose:
                    print(
                        f"[IQL] early stopping at step {step} after "
                        f"{patience} evaluations without improvement."
                    )
                break
        elif verbose and (step == 1 or step % log_freq == 0):
            print(
                f"[IQL] step {step}: q={metrics['q_loss']:.5f} "
                f"v={metrics['value_loss']:.5f} "
                f"pi={metrics['policy_loss']:.5f}"
            )

        if checkpoint_freq and step % checkpoint_freq == 0:
            agent.save(output_dir / f"checkpoint_step_{step}.pt")

    agent.save(final_path)
    summary = {
        "algo": "iql",
        "iql_config": config.as_dict(),
        "requested_steps": int(total_timesteps),
        "completed_steps": int(agent.global_step),
        "batch_size": int(config.batch_size),
        "early_stopped": stopped_early,
        "evaluations_without_improvement": evaluations_without_improvement,
        "patience": int(patience),
        "min_timesteps": int(min_timesteps),
        "eval_freq": int(eval_freq),
        "best_validation_total_loss": (
            None if not history else float(best_validation)
        ),
        "history": history,
        "best_model_path": str(best_path) if best_path.exists() else None,
        "final_model_path": str(final_path),
        "train_transitions": len(train_dataset),
        "validation_transitions": len(val_dataset),
        "train_episodes": train_dataset.num_episodes,
        "validation_episodes": val_dataset.num_episodes,
        "qos": (
            None
            if config.qos_constraint() is None
            else {
                "constraint": config.qos_constraint().as_dict(),
                "dataset": config.qos_constraint().summary(
                    dataset.raw_components
                ),
                "lambda_final": float(agent.lagrange.value),
                "lambda_history": list(agent.lagrange.history),
            }
        ),
        "duration_seconds": round(time.time() - started, 3),
    }
    if agent_config is not None:
        if hasattr(agent_config, "as_dict"):
            summary["paper_agent"] = agent_config.as_dict()
        elif isinstance(agent_config, Mapping):
            summary["paper_agent"] = dict(agent_config)
    _write_json(output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "IQLConfig",
    "IQLAgent",
    "train_iql",
]
