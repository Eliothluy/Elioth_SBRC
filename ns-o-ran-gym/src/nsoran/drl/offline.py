"""Offline dataset format, validation and Gymnasium replay environment.

Dataset contract (``.npz``)
---------------------------
Required arrays, all with the same first dimension ``N``::

    observations       float32 [N, D]    D=85 (paper), 70 (E5 compact) or
                                         78 (legacy E5 ablation)
    actions            int     [N]       values in [0, 127]
    rewards            float32 [N]
    next_observations  float32 [N, D]
    terminals          int/bool[N]       0/1
    episode_ids        int     [N]
    timestamps         numeric [N]       milliseconds (kept as provided)

Optional arrays::

    reward_components  float32 [N, 5]    throughput, energy, rlf,
                                          es_on_cost, bs_on (all normalized)
    raw_components     float32 [N, 5]    same order, raw KPM units (optional;
                                          required by the E1 QoS constraint)
    metadata           JSON string       arbitrary metadata

``reward_components`` enables re-scoring a dataset with any of the six
Table I configurations (``--apply-reward-config`` in the training CLI).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:  # optional dependency group: drl
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.offline requires numpy. Install the DRL extras with "
        "`pip install -e '.[drl]'` or `pip install -r requirements-drl.txt`."
    ) from exc

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.offline requires gymnasium (a base dependency of nsoran). "
        "Install the package with `pip install -e .`."
    ) from exc

from .config import (
    N_ACTIONS,
    STATE_DIM,
    compute_reward,
    resolve_agent_config,
)

REQUIRED_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "episode_ids",
    "timestamps",
)


class DatasetValidationError(ValueError):
    """Raised when an offline dataset does not satisfy the contract."""


@dataclass
class OfflineDataset:
    """In-memory offline transition dataset."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    episode_ids: np.ndarray
    timestamps: np.ndarray
    reward_components: Optional[np.ndarray] = None
    raw_components: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce to the canonical dtypes without silently dropping data.
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64).reshape(-1)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.terminals = np.asarray(self.terminals).reshape(-1)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64).reshape(-1)
        timestamps = np.asarray(self.timestamps).reshape(-1)
        if np.issubdtype(timestamps.dtype, np.number):
            if timestamps.size and np.all(np.isfinite(timestamps)):
                # Keep integer millisecond timestamps compact when possible.
                if np.all(timestamps == np.floor(timestamps)):
                    timestamps = timestamps.astype(np.int64)
        self.timestamps = timestamps
        if self.reward_components is not None:
            self.reward_components = np.asarray(
                self.reward_components, dtype=np.float32
            )
            if self.reward_components.ndim == 1:  # tolerate [N*5]? reject clearly
                raise DatasetValidationError(
                    "reward_components must have shape [N, 5]."
                )
        if self.raw_components is not None:
            self.raw_components = np.asarray(self.raw_components, dtype=np.float32)
            if self.raw_components.ndim == 1:
                raise DatasetValidationError(
                    "raw_components must have shape [N, 5]."
                )

    def __len__(self) -> int:
        return int(self.observations.shape[0]) if self.observations.ndim else 0

    @property
    def num_transitions(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return int(np.unique(self.episode_ids).size)

    def validate(self, obs_dim: Optional[int] = None) -> "OfflineDataset":
        """Validate shapes, dtypes, ranges and finiteness.

        ``obs_dim`` defaults to the width actually stored in the dataset, so
        the paper state (85), the E5 compact state (70) and the legacy E5
        ablation (78) all validate without callers having to hard-code a
        width.

        Returns ``self`` to allow chaining.  Raises
        :class:`DatasetValidationError` with every detected problem.
        """

        if obs_dim is None:
            obs_dim = (
                int(self.observations.shape[1])
                if self.observations.ndim == 2
                else STATE_DIM
            )
        errors: List[str] = []
        arrays = {
            name: getattr(self, name)
            for name in REQUIRED_ARRAYS
        }

        for name, array in arrays.items():
            if array.ndim == 0:
                errors.append(f"{name} must be an array, got a scalar.")
            elif array.shape[0] != len(self):
                errors.append(
                    f"{name} has length {array.shape[0]}, expected {len(self)}."
                )

        if self.observations.ndim != 2:
            errors.append(
                f"observations must be 2-D [N, {obs_dim}], "
                f"got shape {self.observations.shape}."
            )
        elif self.observations.shape[1] != obs_dim:
            errors.append(
                f"observations must have {obs_dim} columns, "
                f"got {self.observations.shape[1]}."
            )

        if self.next_observations.ndim != 2:
            errors.append(
                f"next_observations must be 2-D [N, {obs_dim}], "
                f"got shape {self.next_observations.shape}."
            )
        elif self.next_observations.shape[1] != obs_dim:
            errors.append(
                f"next_observations must have {obs_dim} columns, "
                f"got {self.next_observations.shape[1]}."
            )

        for name in ("observations", "next_observations"):
            array = getattr(self, name)
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                errors.append(f"{name} contains NaN or infinite values.")

        if len(self) == 0:
            errors.append("dataset is empty (N == 0).")

        if self.actions.ndim != 1:
            errors.append(f"actions must be 1-D, got shape {self.actions.shape}.")
        elif self.actions.size and (
            self.actions.min() < 0 or self.actions.max() >= N_ACTIONS
        ):
            errors.append(
                f"actions must be in [0, {N_ACTIONS - 1}], got range "
                f"[{self.actions.min()}, {self.actions.max()}]."
            )
        elif not np.issubdtype(self.actions.dtype, np.integer):
            errors.append(f"actions must be integers, got {self.actions.dtype}.")

        if self.rewards.ndim != 1:
            errors.append(f"rewards must be 1-D, got shape {self.rewards.shape}.")
        elif not np.all(np.isfinite(self.rewards)):
            errors.append("rewards contains NaN or infinite values.")

        if self.terminals.ndim != 1:
            errors.append(f"terminals must be 1-D, got shape {self.terminals.shape}.")
        elif self.terminals.size and not np.all(np.isin(self.terminals, (0, 1, False, True))):
            errors.append("terminals must contain only 0/1 (or booleans).")

        if self.episode_ids.ndim != 1:
            errors.append(
                f"episode_ids must be 1-D, got shape {self.episode_ids.shape}."
            )
        elif not np.issubdtype(self.episode_ids.dtype, np.integer):
            errors.append(
                f"episode_ids must be integers, got {self.episode_ids.dtype}."
            )

        if self.timestamps.ndim != 1:
            errors.append(
                f"timestamps must be 1-D, got shape {self.timestamps.shape}."
            )
        elif not np.issubdtype(self.timestamps.dtype, np.number):
            errors.append(
                f"timestamps must be numeric, got {self.timestamps.dtype}."
            )
        elif not np.all(np.isfinite(self.timestamps)):
            errors.append("timestamps contains NaN or infinite values.")

        if self.reward_components is not None:
            if self.reward_components.shape != (len(self), 5):
                errors.append(
                    "reward_components must have shape "
                    f"[{len(self)}, 5], got {self.reward_components.shape}."
                )
            elif not np.all(np.isfinite(self.reward_components)):
                errors.append("reward_components contains NaN or infinite values.")

        if self.raw_components is not None:
            if self.raw_components.shape != (len(self), 5):
                errors.append(
                    "raw_components must have shape "
                    f"[{len(self)}, 5], got {self.raw_components.shape}."
                )
            elif not np.all(np.isfinite(self.raw_components)):
                errors.append("raw_components contains NaN or infinite values.")

        if errors:
            raise DatasetValidationError(
                "Invalid offline dataset:\n  - " + "\n  - ".join(errors)
            )
        return self

    def subset(self, indices: Sequence[int]) -> "OfflineDataset":
        """Return a copy containing only ``indices`` (order preserved)."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError("subset indices are out of range.")
        return OfflineDataset(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminals=self.terminals[indices],
            episode_ids=self.episode_ids[indices],
            timestamps=self.timestamps[indices],
            reward_components=(
                None
                if self.reward_components is None
                else self.reward_components[indices]
            ),
            raw_components=(
                None
                if self.raw_components is None
                else self.raw_components[indices]
            ),
            metadata=dict(self.metadata),
        )

    def with_rewards_from_components(self, agent: Any) -> "OfflineDataset":
        """Recompute rewards from ``reward_components`` using Table I weights.

        Raises ``ValueError`` when the optional component array is absent, so
        callers never silently train on the wrong reward function.
        """

        if self.reward_components is None:
            raise ValueError(
                "This dataset has no 'reward_components' array; cannot apply the "
                "Table I reward configuration. Use the stored rewards instead."
            )
        cfg = resolve_agent_config(agent)
        components = self.reward_components
        rewards = compute_reward(
            components[:, 0],
            components[:, 1],
            components[:, 2],
            components[:, 3],
            components[:, 4],
            cfg,
        )
        return replace(
            self,
            rewards=np.asarray(rewards, dtype=np.float32),
            metadata={
                **self.metadata,
                "reward_agent": cfg.name,
                "reward_weights": cfg.weights,
                "reward_normalization": cfg.normalization,
            },
        )


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------


def load_offline_dataset(
    path: Union[str, os.PathLike],
    obs_dim: Optional[int] = None,
    validate: bool = True,
) -> OfflineDataset:
    """Load and (by default) validate an ``.npz`` offline dataset.

    ``obs_dim`` defaults to the width stored in the file, which keeps the
    85-dim paper datasets and the 78-dim E5 datasets interchangeable.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Offline dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
        if missing:
            raise DatasetValidationError(
                f"{path} is missing required array(s): {', '.join(missing)}. "
                f"Found: {', '.join(sorted(data.files))}."
            )
        arrays = {name: data[name] for name in REQUIRED_ARRAYS}
        reward_components = (
            data["reward_components"]
            if "reward_components" in data.files
            else None
        )
        raw_components = (
            data["raw_components"] if "raw_components" in data.files else None
        )
        metadata: Dict[str, Any] = {}
        if "metadata" in data.files:
            raw_metadata = data["metadata"]
            try:
                if isinstance(raw_metadata, np.ndarray):
                    raw_metadata = raw_metadata.item()
                if isinstance(raw_metadata, bytes):
                    raw_metadata = raw_metadata.decode("utf-8")
                metadata = json.loads(str(raw_metadata))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(
                    f"Could not decode the 'metadata' JSON string in {path}: {exc}"
                ) from exc
            if not isinstance(metadata, dict):
                raise DatasetValidationError(
                    f"'metadata' in {path} must decode to a JSON object."
                )

    dataset = OfflineDataset(
        **arrays,
        reward_components=reward_components,
        raw_components=raw_components,
        metadata=metadata,
    )
    if validate:
        dataset.validate(obs_dim=obs_dim)
    return dataset


def save_offline_dataset(
    path: Union[str, os.PathLike],
    dataset: OfflineDataset,
    compressed: bool = True,
) -> Path:
    """Persist an :class:`OfflineDataset` to ``.npz``."""

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "observations": dataset.observations,
        "actions": dataset.actions,
        "rewards": dataset.rewards,
        "next_observations": dataset.next_observations,
        "terminals": dataset.terminals,
        "episode_ids": dataset.episode_ids,
        "timestamps": dataset.timestamps,
        "metadata": np.asarray(json.dumps(dataset.metadata, default=str)),
    }
    if dataset.reward_components is not None:
        payload["reward_components"] = dataset.reward_components
    if dataset.raw_components is not None:
        payload["raw_components"] = dataset.raw_components
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Splitting and synthetic data (useful for smoke tests / examples)
# ---------------------------------------------------------------------------


def split_dataset(
    dataset: OfflineDataset,
    val_fraction: float = 0.1,
    seed: int = 0,
    min_val_episodes: int = 1,
) -> Tuple[OfflineDataset, OfflineDataset]:
    """Split by episode to avoid temporal leakage between train and validation.

    If the dataset contains a single episode, a transition-level split is used
    as a fallback (with a warning in the returned metadata).
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    rng = np.random.default_rng(seed)
    episode_ids = np.unique(dataset.episode_ids)

    if episode_ids.size >= 2:
        shuffled = rng.permutation(episode_ids)
        n_val = int(round(episode_ids.size * val_fraction))
        n_val = max(min_val_episodes, n_val)
        n_val = min(episode_ids.size - 1, n_val)
        val_ids = shuffled[:n_val]
        val_mask = np.isin(dataset.episode_ids, val_ids)
        train_indices = np.flatnonzero(~val_mask)
        val_indices = np.flatnonzero(val_mask)
        train = dataset.subset(train_indices)
        val = dataset.subset(val_indices)
        train.metadata = {**train.metadata, "split": "train", "split_seed": seed}
        val.metadata = {
            **val.metadata,
            "split": "validation",
            "split_seed": seed,
            "validation_episode_ids": [int(v) for v in val_ids],
        }
        return train, val

    # Single-episode fallback: split transitions, keep ends where possible.
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(
            "Dataset is too small to split into train/validation transitions."
        )
    permutation = rng.permutation(len(dataset))
    val_indices = np.sort(permutation[:n_val])
    train_indices = np.sort(permutation[n_val:])
    train = dataset.subset(train_indices)
    val = dataset.subset(val_indices)
    train.metadata = {
        **train.metadata,
        "split": "train",
        "split_seed": seed,
        "single_episode_fallback": True,
    }
    val.metadata = {
        **val.metadata,
        "split": "validation",
        "split_seed": seed,
        "single_episode_fallback": True,
    }
    return train, val


def synthetic_dataset(
    num_transitions: int = 500,
    episode_length: int = 100,
    obs_dim: int = STATE_DIM,
    seed: int = 0,
    include_reward_components: bool = True,
) -> OfflineDataset:
    """Create a small valid dataset for smoke tests and CLI demos.

    This is **not** a substitute for data collected from ns-3.
    """

    if num_transitions < 1 or episode_length < 1:
        raise ValueError("num_transitions and episode_length must be positive.")
    rng = np.random.default_rng(seed)
    n = int(num_transitions)

    episode_ids = np.arange(n, dtype=np.int64) // int(episode_length)
    positions = np.arange(n, dtype=np.int64) % int(episode_length)
    timestamps = positions * 100  # 100 ms E2 periodicity per episode
    observations = rng.random((n, obs_dim), dtype=np.float32)
    next_observations = np.roll(observations, -1, axis=0).copy()
    # Keep the final next-observation of each episode equal to its last obs.
    for episode in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == episode)
        next_observations[idx[-1]] = observations[idx[-1]]
    actions = rng.integers(0, N_ACTIONS, size=n, dtype=np.int64)
    rewards = rng.normal(0.0, 0.1, size=n).astype(np.float32)
    terminals = (positions == int(episode_length) - 1).astype(np.int64)

    reward_components = None
    if include_reward_components:
        components = rng.random((n, 5), dtype=np.float32)
        # Make components sum to a plausible reward scale.
        reward_components = components
        rewards = (
            0.51 * components[:, 0]
            - 0.19 * components[:, 1]
            - 0.20 * components[:, 2]
            - 0.10 * components[:, 3]
            - 0.19 * components[:, 4]
        ).astype(np.float32)

    return OfflineDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        episode_ids=episode_ids,
        timestamps=timestamps,
        reward_components=reward_components,
        metadata={"source": "synthetic_dataset", "seed": seed},
    )


# ---------------------------------------------------------------------------
# Gymnasium replay environment
# ---------------------------------------------------------------------------


class OfflineReplayEnv(gym.Env):
    """Replay an offline dataset as a Gymnasium environment.

    The environment is used to train PPO with Stable Baselines3 without
    running ns-3.  Each call to :meth:`reset` starts one recorded episode and
    :meth:`step` returns the recorded transition.  The agent's action is
    **ignored** for the transition (the dataset is fixed) but is reported in
    ``info`` together with the logged action and a mismatch flag, which is
    useful for diagnostics.

    Modes
    -----
    ``deterministic``
        Episodes are replayed in dataset order.
    ``shuffled``
        Episode order is reshuffled after every full pass (epoch).  The
        temporal order *inside* each episode is always preserved.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        dataset: Union[OfflineDataset, str, os.PathLike],
        mode: str = "deterministic",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        obs_dim: Optional[int] = None,
    ) -> None:
        if isinstance(dataset, (str, os.PathLike)):
            dataset = load_offline_dataset(dataset, obs_dim=obs_dim)
        if not isinstance(dataset, OfflineDataset):
            raise TypeError(
                "dataset must be an OfflineDataset or a path to an .npz file."
            )
        if obs_dim is None:
            obs_dim = int(dataset.observations.shape[1])
        dataset.validate(obs_dim=obs_dim)
        if mode not in ("deterministic", "shuffled"):
            raise ValueError(
                f"mode must be 'deterministic' or 'shuffled', got {mode!r}."
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}.")

        self.dataset = dataset
        self.mode = mode
        self.render_mode = render_mode
        self.obs_dim = obs_dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._episodes: List[Tuple[int, np.ndarray]] = self._group_episodes()
        if not self._episodes:
            raise ValueError("Cannot replay an empty dataset.")
        self._episode_order = np.arange(len(self._episodes), dtype=np.int64)
        self._order_pos = 0
        self._epoch = 0
        if self.mode == "shuffled":
            self._episode_order = self._rng.permutation(len(self._episodes))

        self._active = False
        self._current_episode_position = -1
        self._current_indices: Optional[np.ndarray] = None
        self._step_position = 0
        self._last_observation: Optional[np.ndarray] = None

    # -- episode bookkeeping -------------------------------------------------

    def _group_episodes(self) -> List[Tuple[int, np.ndarray]]:
        groups: Dict[int, List[int]] = {}
        for index, episode_id in enumerate(self.dataset.episode_ids):
            groups.setdefault(int(episode_id), []).append(index)
        episodes: List[Tuple[int, np.ndarray]] = []
        for episode_id, indices in groups.items():
            ordered = np.asarray(indices, dtype=np.int64)
            # Replay each episode chronologically even if the dataset rows were
            # stored in a different order.  Stable sort preserves ties.
            timestamps = np.asarray(
                self.dataset.timestamps[ordered], dtype=np.float64
            )
            if np.all(np.isfinite(timestamps)):
                ordered = ordered[np.argsort(timestamps, kind="stable")]
            episodes.append((episode_id, ordered))
        return episodes

    def _shuffle_episode_order(self) -> None:
        self._episode_order = self._rng.permutation(len(self._episodes))
        self._order_pos = 0

    def _next_episode(self) -> Tuple[int, np.ndarray]:
        if self._order_pos >= len(self._episode_order):
            self._epoch += 1
            if self.mode == "shuffled":
                self._shuffle_episode_order()
            else:
                self._order_pos = 0
        episode_position = int(self._episode_order[self._order_pos])
        self._order_pos += 1
        episode_id, indices = self._episodes[episode_position]
        self._current_episode_position = episode_position
        self._current_indices = indices
        self._step_position = 0
        self._active = True
        return episode_id, indices

    # -- Gym API -------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._epoch = 0
            if self.mode == "shuffled":
                self._shuffle_episode_order()
            else:
                self._order_pos = 0

        requested_episode = None if options is None else options.get("episode_index")
        if requested_episode is not None:
            episode_position = int(requested_episode)
            if not 0 <= episode_position < len(self._episodes):
                raise IndexError(
                    f"episode_index must be in [0, {len(self._episodes) - 1}]."
                )
            episode_id, indices = self._episodes[episode_position]
            self._current_episode_position = episode_position
            self._current_indices = indices
            self._step_position = 0
            self._active = True
        else:
            episode_id, indices = self._next_episode()

        first_index = int(indices[0])
        observation = np.array(
            self.dataset.observations[first_index], dtype=np.float32, copy=True
        )
        self._last_observation = observation
        info = {
            "dataset_index": first_index,
            "episode_id": int(episode_id),
            "episode_index": int(self._current_episode_position),
            "episode_length": int(indices.size),
            "episode_step": 0,
            "epoch": int(self._epoch),
            "mode": self.mode,
        }
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self._active or self._current_indices is None:
            raise RuntimeError(
                "OfflineReplayEnv.step() called before reset() or after the "
                "episode ended; call reset() first."
            )
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be in [0, {N_ACTIONS - 1}], got {action}."
            )
        if self._step_position >= self._current_indices.size:
            raise RuntimeError("Replay cursor exceeded the current episode.")

        dataset_index = int(self._current_indices[self._step_position])
        recorded_action = int(self.dataset.actions[dataset_index])
        reward = float(self.dataset.rewards[dataset_index])
        terminated = bool(self.dataset.terminals[dataset_index])
        at_episode_end = self._step_position == self._current_indices.size - 1
        truncated = bool(at_episode_end and not terminated)

        observation = np.array(
            self.dataset.next_observations[dataset_index],
            dtype=np.float32,
            copy=True,
        )
        self._last_observation = observation
        self._step_position += 1
        if terminated or truncated:
            self._active = False

        info = {
            "dataset_index": dataset_index,
            "episode_id": int(self.dataset.episode_ids[dataset_index]),
            "episode_index": int(self._current_episode_position),
            "episode_step": self._step_position,
            "timestamp": (
                int(self.dataset.timestamps[dataset_index])
                if np.issubdtype(self.dataset.timestamps.dtype, np.integer)
                else float(self.dataset.timestamps[dataset_index])
            ),
            "recorded_action": recorded_action,
            "action_mismatch": action != recorded_action,
            "truncated_at_dataset_boundary": truncated,
            "mode": self.mode,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self._current_indices is None:
            return None
        text = (
            f"OfflineReplayEnv(mode={self.mode}, episode="
            f"{self._current_episode_position}, step={self._step_position}/"
            f"{self._current_indices.size})"
        )
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def close(self) -> None:
        self._active = False
        self._current_indices = None


__all__ = [
    "REQUIRED_ARRAYS",
    "DatasetValidationError",
    "OfflineDataset",
    "load_offline_dataset",
    "save_offline_dataset",
    "split_dataset",
    "synthetic_dataset",
    "OfflineReplayEnv",
]
