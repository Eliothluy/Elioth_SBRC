"""REM-DQN with Conservative Q-Learning (CQL) for offline cell sleep control.

Reference: Sections IV of the paper.  The implementation follows the paper's
hybrid network (1-D CNN + dense ``[512, 512, 256]``), a Random Ensemble Mixture
(REM) of ``K`` Q-heads and a conservative CQL penalty.  Training is fully
offline: mini-batches are sampled from an :class:`OfflineDataset`.

Importing this module requires PyTorch; ``import nsoran.drl`` does not import it.
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
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.rem_dqn requires PyTorch and numpy. Install the DRL extras "
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
from .networks import REMQNetwork
from .offline import OfflineDataset, split_dataset
from .qos import LagrangeMultiplier, QoSConstraint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class REMDQNConfig:
    """Hyper-parameters for REM-DQN + CQL."""

    obs_dim: int = STATE_DIM
    n_actions: int = N_ACTIONS
    hidden_dims: Tuple[int, ...] = (512, 512, 256)
    conv_channels: Tuple[int, ...] = (32, 64)
    k_heads: int = 3

    cql_alpha: float = 0.5
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = DEFAULT_BATCH_SIZE
    target_update_freq: int = 1000
    tau: Optional[float] = None  # None -> hard target updates
    grad_clip: float = 10.0
    double_dqn: bool = True

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    device: str = "auto"
    seed: Optional[int] = None

    # E1: explicit QoS constraint, shared with the IQL implementation.
    qos_throughput_min: Optional[float] = None
    qos_rlf_max: Optional[float] = None
    qos_throughput_scale: Optional[float] = None
    qos_rlf_scale: Optional[float] = None
    qos_lambda_lr: float = 0.01
    qos_lambda_init: float = 0.0
    qos_lambda_max: float = 100.0
    qos_tolerance: float = 0.0
    state_variant: str = "full"

    def __post_init__(self) -> None:
        self.obs_dim = int(self.obs_dim)
        self.n_actions = int(self.n_actions)
        self.hidden_dims = tuple(int(d) for d in self.hidden_dims)
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        self.k_heads = int(self.k_heads)
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if self.k_heads <= 0:
            raise ValueError("k_heads must be positive.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty.")
        if self.cql_alpha < 0:
            raise ValueError("cql_alpha must be >= 0.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive.")
        if self.tau is not None and not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1] when provided.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1."
            )
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive.")
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
    def from_dict(cls, values: Mapping[str, Any]) -> "REMDQNConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {device!r} was requested but CUDA is not available."
        )
    return torch.device(device)


def _torch_load(path: Union[str, os.PathLike], map_location: Any) -> Any:
    # ``weights_only=False`` is required for optimizer/target payloads and is
    # safe here because checkpoints are produced by this package.  Older torch
    # versions do not accept the keyword.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class OfflineReplayBuffer:
    """Uniform mini-batch sampler over an offline dataset."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        raw_components: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.next_observations = np.asarray(
            next_observations, dtype=np.float32
        )
        self.terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
        self.raw_components = (
            None
            if raw_components is None
            else np.asarray(raw_components, dtype=np.float32)
        )
        lengths = {
            len(self.observations),
            len(self.actions),
            len(self.rewards),
            len(self.next_observations),
            len(self.terminals),
        }
        if self.raw_components is not None:
            lengths.add(len(self.raw_components))
        if len(lengths) != 1:
            raise ValueError("Replay buffer arrays must all have the same length.")
        if len(self) == 0:
            raise ValueError("Replay buffer must not be empty.")
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_dataset(
        cls, dataset: OfflineDataset, seed: Optional[int] = None
    ) -> "OfflineReplayBuffer":
        return cls(
            observations=dataset.observations,
            actions=dataset.actions,
            rewards=dataset.rewards,
            next_observations=dataset.next_observations,
            terminals=dataset.terminals,
            raw_components=dataset.raw_components,
            seed=seed,
        )

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = self._rng.integers(0, len(self), size=int(batch_size))
        batch = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "terminals": self.terminals[indices],
        }
        if self.raw_components is not None:
            batch["raw_components"] = self.raw_components[indices]
        return batch


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class REMDQNAgent:
    """Random Ensemble Mixture DQN with a CQL conservative penalty."""

    def __init__(
        self,
        config: Optional[Union[REMDQNConfig, Mapping[str, Any]]] = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            resolved = REMDQNConfig(**overrides)
        elif isinstance(config, REMDQNConfig):
            resolved = replace(config, **overrides) if overrides else config
        elif isinstance(config, Mapping):
            values = dict(config)
            values.update(overrides)
            resolved = REMDQNConfig.from_dict(values)
        else:
            raise TypeError(
                "config must be REMDQNConfig, mapping or None, got "
                f"{type(config).__name__}."
            )
        self.config = resolved
        self.device = _resolve_device(resolved.device)
        if resolved.seed is not None:
            np.random.seed(int(resolved.seed))
            torch.manual_seed(int(resolved.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(resolved.seed))

        self.online = REMQNetwork(
            obs_dim=resolved.obs_dim,
            n_actions=resolved.n_actions,
            hidden_dims=resolved.hidden_dims,
            k_heads=resolved.k_heads,
            channels=resolved.conv_channels,
        ).to(self.device)
        self.target = REMQNetwork(
            obs_dim=resolved.obs_dim,
            n_actions=resolved.n_actions,
            hidden_dims=resolved.hidden_dims,
            k_heads=resolved.k_heads,
            channels=resolved.conv_channels,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=resolved.lr
        )
        self.global_step = 0
        self._rng = np.random.default_rng(resolved.seed)

        # E1: explicit QoS constraint (see ``nsoran.drl.qos``).  The multiplier
        # is a learned dual variable restored from the checkpoint.
        self.qos_constraint = resolved.qos_constraint()
        self.lagrange: Optional[LagrangeMultiplier] = (
            LagrangeMultiplier(self.qos_constraint)
            if self.qos_constraint is not None
            else None
        )

    # -- inference -----------------------------------------------------------

    def epsilon(self, step: Optional[int] = None) -> float:
        """Linearly decayed exploration rate (included for the paper schedule)."""

        step = self.global_step if step is None else int(step)
        fraction = min(1.0, max(0.0, step / self.config.epsilon_decay_steps))
        return (
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start)
        )

    @torch.no_grad()
    def q_values(
        self, observations: Union[np.ndarray, torch.Tensor], target: bool = False
    ) -> torch.Tensor:
        network = self.target if target else self.online
        network.eval()
        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return network(tensor)

    def act(
        self,
        observation: np.ndarray,
        *,
        step: Optional[int] = None,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[int, np.ndarray]:
        """Epsilon-greedy action(s) using the ensemble-mean Q-values."""

        single = np.asarray(observation).ndim == 1
        observations = np.atleast_2d(
            np.asarray(observation, dtype=np.float32)
        )
        if observations.shape[1] != self.config.obs_dim:
            raise ValueError(
                f"observation must have {self.config.obs_dim} features, got "
                f"{observations.shape[1]}."
            )
        rng = self._rng if rng is None else rng
        was_training = self.online.training
        n = observations.shape[0]
        epsilon = 0.0 if deterministic else self.epsilon(step)
        actions = rng.integers(0, self.config.n_actions, size=n).astype(np.int64)
        if epsilon < 1.0:
            q = self.q_values(observations).mean(dim=1)
            greedy = q.argmax(dim=1).cpu().numpy().astype(np.int64)
            exploit_mask = rng.random(n) >= epsilon
            actions[exploit_mask] = greedy[exploit_mask]
        self.online.train(was_training)
        return int(actions[0]) if single else actions

    # -- losses --------------------------------------------------------------

    def _sample_alpha(
        self, batch_size: int, deterministic: bool = False
    ) -> torch.Tensor:
        if deterministic:
            return torch.full(
                (int(batch_size), self.config.k_heads),
                1.0 / self.config.k_heads,
                device=self.device,
                dtype=torch.float32,
            )
        concentration = torch.ones(
            self.config.k_heads, device=self.device, dtype=torch.float32
        )
        return torch.distributions.Dirichlet(concentration).sample(
            (int(batch_size),)
        )

    def _to_tensors(
        self, batch: Mapping[str, Any]
    ) -> Dict[str, torch.Tensor]:
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
            tensors["rewards"] = rewards - self.lagrange.penalty(cost)
            tensors["qos_cost"] = cost
        return tensors

    def compute_losses(
        self, batch: Mapping[str, Any], deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute TD, CQL and total losses for one mini-batch.

        Public so the conservative loss can be unit-tested without a full
        training loop.  ``deterministic=True`` fixes the REM mixture weights to
        the uniform average (used for stable validation metrics).
        """

        tensors = self._to_tensors(batch)
        observations = tensors["observations"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        next_observations = tensors["next_observations"]
        terminals = tensors["terminals"]
        batch_size = observations.shape[0]

        with torch.no_grad():
            next_online = self.online(next_observations)  # [B, K, A]
            alpha_next = self._sample_alpha(batch_size, deterministic)  # [B, K]
            next_mixture = (alpha_next.unsqueeze(-1) * next_online).sum(dim=1)
            if self.config.double_dqn:
                next_actions = next_mixture.argmax(dim=1, keepdim=True)
            else:
                next_target = self.target(next_observations)
                next_target_mixture = (
                    alpha_next.unsqueeze(-1) * next_target
                ).sum(dim=1)
                next_actions = next_target_mixture.argmax(dim=1, keepdim=True)

            target_all = self.target(next_observations)  # [B, K, A]
            target_actions = next_actions.unsqueeze(1).expand(
                -1, self.config.k_heads, 1
            )
            target_q = target_all.gather(2, target_actions).squeeze(2)
            target_mixture = (alpha_next * target_q).sum(dim=1)
            bootstrap = rewards + self.config.gamma * (1.0 - terminals) * target_mixture

        current_all = self.online(observations)  # [B, K, A]
        action_indices = actions.unsqueeze(1).expand(
            -1, self.config.k_heads, 1
        )
        current_q = current_all.gather(2, action_indices).squeeze(2)  # [B, K]
        alpha = self._sample_alpha(batch_size, deterministic)
        current_mixture = (alpha * current_q).sum(dim=1)

        td_loss = F.smooth_l1_loss(current_mixture, bootstrap)
        # CQL(H): push down log-sum-exp over all actions while keeping the
        # dataset action value; averaged over REM heads.
        cql_loss = (
            torch.logsumexp(current_all, dim=2) - current_q
        ).mean()
        total_loss = td_loss + self.config.cql_alpha * cql_loss

        metrics = {
            "td_loss": td_loss,
            "cql_loss": cql_loss,
            "total_loss": total_loss,
            "q_mean": current_q.mean(),
        }
        if "qos_cost" in tensors:
            metrics["qos_cost_mean"] = tensors["qos_cost"].mean().detach()
            metrics["lagrange_multiplier"] = torch.as_tensor(
                float(self.lagrange.value), device=self.device
            )
        return metrics

    @torch.no_grad()
    def validation_loss(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        """Evaluate TD/CQL losses without gradient updates."""

        was_training = self.online.training
        self.online.eval()
        losses = self.compute_losses(batch, deterministic=True)
        self.online.train(was_training)
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    # -- optimisation --------------------------------------------------------

    def update(
        self, batch: Mapping[str, Any], step: Optional[int] = None
    ) -> Dict[str, float]:
        """One gradient update.  ``step`` is accepted for logging API symmetry."""

        self.online.train()
        losses = self.compute_losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.config.grad_clip
            )
        self.optimizer.step()
        self.global_step += 1

        if self.config.tau is not None:
            self.soft_update_target(self.config.tau)
        elif self.global_step % self.config.target_update_freq == 0:
            self.update_target()

        metrics = {
            key: float(value.detach().cpu()) for key, value in losses.items()
        }
        if self.lagrange is not None and "qos_cost_mean" in metrics:
            self.lagrange.update(metrics["qos_cost_mean"])
            metrics["lagrange_multiplier"] = float(self.lagrange.value)
        metrics["epsilon"] = self.epsilon(
            self.global_step if step is None else step
        )
        metrics["global_step"] = float(self.global_step)
        return metrics

    @torch.no_grad()
    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        for target_parameter, online_parameter in zip(
            self.target.parameters(), self.online.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(
                online_parameter, alpha=tau
            )

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
            "algo": "rem_dqn",
            "config": self.config.as_dict(),
            "state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "global_step": self.global_step,
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lagrange is not None:
            payload["lagrange_multiplier"] = self.lagrange.state_dict()
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
    ) -> "REMDQNAgent":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"REM-DQN checkpoint not found: {path}")
        payload = _torch_load(path, map_location or "cpu")
        if not isinstance(payload, Mapping) or "state_dict" not in payload:
            raise ValueError(f"{path} is not a REM-DQN checkpoint.")
        config_values = payload.get("config", {})
        config = REMDQNConfig.from_dict(config_values)
        agent = cls(config)
        agent.online.load_state_dict(payload["state_dict"])
        if "target_state_dict" in payload:
            agent.target.load_state_dict(payload["target_state_dict"])
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
                # Optimizer state is optional and device-specific; inference
                # only needs the network weights.
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


def train_rem_dqn(
    dataset: OfflineDataset,
    *,
    agent_config: Optional[Union[REMDQNConfig, Mapping[str, Any]]] = None,
    output_dir: Union[str, os.PathLike] = "runs/rem_dqn",
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
    """Train REM-DQN offline with validation-loss early stopping.

    ``total_timesteps`` is interpreted as the number of gradient updates
    (offline training does not consume environment steps).  Validation uses a
    held-out set of episodes when possible.
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
    if agent_config is not None:
        if isinstance(agent_config, REMDQNConfig):
            config_values = agent_config.as_dict()
        elif isinstance(agent_config, Mapping):
            config_values = dict(agent_config)
        else:
            raise TypeError("agent_config must be REMDQNConfig, mapping or None.")
    config_values.update(config_overrides)
    if batch_size is not None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        config_values["batch_size"] = int(batch_size)
    config_values.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    config_values.setdefault("seed", seed)
    config = REMDQNConfig.from_dict(config_values)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = split_dataset(
        dataset, val_fraction=val_fraction, seed=seed
    )
    train_buffer = OfflineReplayBuffer.from_dataset(train_dataset, seed=seed)
    val_buffer = OfflineReplayBuffer.from_dataset(
        val_dataset, seed=seed + 1
    )
    validation_batch = val_buffer.sample(
        min(len(val_buffer), max(config.batch_size, 512))
    )

    agent = REMDQNAgent(config)
    best_validation = float("inf")
    evaluations_without_improvement = 0
    history: List[Dict[str, Any]] = []
    best_path = output_dir / "best_model.pt"
    final_path = output_dir / "final_model.pt"
    started = time.time()
    stopped_early = False

    if verbose:
        print(
            f"[REM-DQN] training on {len(train_dataset)} transitions "
            f"({train_dataset.num_episodes} episodes), validation on "
            f"{len(val_dataset)} transitions ({val_dataset.num_episodes} episodes)"
        )
        print(f"[REM-DQN] device={agent.device} config={config.as_dict()}")

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
                "train_td_loss": metrics["td_loss"],
                "train_cql_loss": metrics["cql_loss"],
                "train_total_loss": metrics["total_loss"],
                "validation_td_loss": validation["td_loss"],
                "validation_cql_loss": validation["cql_loss"],
                "validation_total_loss": validation["total_loss"],
                "best_validation_total_loss": best_validation,
                "evaluations_without_improvement": (
                    evaluations_without_improvement
                ),
                "epsilon": metrics["epsilon"],
            }
            history.append(record)
            if verbose:
                print(
                    f"[REM-DQN] step {step}: val_total={validation['total_loss']:.6f} "
                    f"best={best_validation:.6f} "
                    f"no_improve={evaluations_without_improvement}/{patience}"
                )
            if evaluations_without_improvement >= patience:
                stopped_early = True
                if verbose:
                    print(
                        f"[REM-DQN] early stopping at step {step} after "
                        f"{patience} evaluations without improvement."
                    )
                break
        elif verbose and (step == 1 or step % log_freq == 0):
            print(
                f"[REM-DQN] step {step}: td={metrics['td_loss']:.5f} "
                f"cql={metrics['cql_loss']:.5f} eps={metrics['epsilon']:.3f}"
            )

        if checkpoint_freq and step % checkpoint_freq == 0:
            agent.save(output_dir / f"checkpoint_step_{step}.pt")

    agent.save(final_path)
    summary = {
        "algo": "rem_dqn",
        "agent_config": config.as_dict(),
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
        "duration_seconds": round(time.time() - started, 3),
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "REMDQNConfig",
    "OfflineReplayBuffer",
    "REMDQNAgent",
    "train_rem_dqn",
]
