"""PPO training (Stable Baselines3) on the offline replay environment.

Paper hyper-parameters (Section IV): Stable Baselines3 defaults except
``batch_size=256`` and an entropy coefficient in ``[0.001, 0.003]``; maximum
1,200,000 timesteps, 100-step episodes, early stopping after four evaluations
without improvement following a minimum of 10,000 timesteps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:  # optional dependency group: drl
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.ppo requires Stable Baselines3 (and torch). Install the DRL "
        "extras with `pip install -e '.[drl]'` or "
        "`pip install -r requirements-drl.txt`."
    ) from exc

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EVAL_FREQ,
    DEFAULT_MIN_TIMESTEPS,
    DEFAULT_TOTAL_TIMESTEPS,
    resolve_agent_config,
)
from .offline import OfflineDataset, OfflineReplayEnv, split_dataset

__all__ = [
    "EarlyStopCallback",
    "build_ppo_model",
    "train_ppo",
    "load_ppo",
    "predict_action",
]


class EarlyStopCallback(BaseCallback):
    """Periodic evaluation + early stopping for PPO.

    Stops training when ``patience`` consecutive evaluations fail to improve
    the mean evaluation return, provided at least ``min_timesteps`` environment
    steps have been collected.  The best model is saved when
    ``best_model_save_path`` is provided.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        *,
        eval_freq: int = DEFAULT_EVAL_FREQ,
        n_eval_episodes: int = 5,
        patience: int = DEFAULT_EARLY_STOP_PATIENCE,
        min_timesteps: int = DEFAULT_MIN_TIMESTEPS,
        min_delta: float = 1e-6,
        best_model_save_path: Optional[Union[str, os.PathLike]] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive.")
        if n_eval_episodes <= 0:
            raise ValueError("n_eval_episodes must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if min_timesteps < 0:
            raise ValueError("min_timesteps must be >= 0.")
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.patience = int(patience)
        self.min_timesteps = int(min_timesteps)
        self.min_delta = float(min_delta)
        self.best_model_save_path = (
            None if best_model_save_path is None else Path(best_model_save_path)
        )
        self.deterministic = deterministic

        self.best_mean_reward = float("-inf")
        self.evaluations_without_improvement = 0
        self.history: List[Dict[str, Any]] = []
        self.early_stopped = False
        self._last_eval_timestep: Optional[int] = None

    def _on_step(self) -> bool:
        if self.num_timesteps < self.min_timesteps:
            return True
        if (
            self._last_eval_timestep is not None
            and self.num_timesteps - self._last_eval_timestep < self.eval_freq
        ):
            return True

        self._last_eval_timestep = int(self.num_timesteps)
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            warn=False,
        )
        mean_reward = float(mean_reward)
        std_reward = float(std_reward)
        improved = mean_reward > self.best_mean_reward + self.min_delta
        if improved:
            self.best_mean_reward = mean_reward
            self.evaluations_without_improvement = 0
            if self.best_model_save_path is not None:
                self.best_model_save_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                self.model.save(self.best_model_save_path)
        else:
            self.evaluations_without_improvement += 1

        self.history.append(
            {
                "timesteps": int(self.num_timesteps),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "best_mean_reward": self.best_mean_reward,
                "evaluations_without_improvement": (
                    self.evaluations_without_improvement
                ),
            }
        )
        if self.verbose:
            print(
                f"[PPO] timesteps={self.num_timesteps} "
                f"eval_mean={mean_reward:.4f} +/- {std_reward:.4f} "
                f"best={self.best_mean_reward:.4f} "
                f"no_improve={self.evaluations_without_improvement}/{self.patience}"
            )

        if self.evaluations_without_improvement >= self.patience:
            self.early_stopped = True
            if self.verbose:
                print(
                    f"[PPO] early stopping at {self.num_timesteps} timesteps "
                    f"after {self.patience} evaluations without improvement."
                )
            return False
        return True


def build_ppo_model(
    env: gym.Env,
    agent_config: Any = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ent_coef: Optional[float] = None,
    seed: Optional[int] = None,
    tensorboard_log: Optional[Union[str, os.PathLike]] = None,
    device: str = "auto",
    verbose: int = 0,
    **ppo_kwargs: Any,
) -> PPO:
    """Create a PPO model with the paper's hyper-parameters.

    Only ``batch_size`` and ``ent_coef`` deviate from SB3 defaults, as stated
    in the paper.  ``ent_coef`` is constrained to ``[0.001, 0.003]``.
    """

    config = None if agent_config is None else resolve_agent_config(agent_config)
    if ent_coef is None:
        ent_coef = (
            config.ent_coef
            if config is not None and config.ent_coef is not None
            else 0.001
        )
    if not 0.001 <= float(ent_coef) <= 0.003:
        raise ValueError(
            "ent_coef must be in [0.001, 0.003] per the paper, "
            f"got {ent_coef!r}."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Standard SB3 PPO defaults, made explicit for reproducibility.
    defaults: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_epochs": 10,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    defaults.update(ppo_kwargs)
    defaults.update(
        {
            "policy": "MlpPolicy",
            "env": env,
            "batch_size": int(batch_size),
            "ent_coef": float(ent_coef),
            "seed": seed,
            "device": device,
            "verbose": verbose,
        }
    )
    if tensorboard_log is not None:
        defaults["tensorboard_log"] = str(tensorboard_log)
    return PPO(**defaults)


def train_ppo(
    dataset: OfflineDataset,
    *,
    agent_config: Any = None,
    output_dir: Union[str, os.PathLike] = "runs/ppo",
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ent_coef: Optional[float] = None,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    n_eval_episodes: int = 5,
    patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    min_timesteps: int = DEFAULT_MIN_TIMESTEPS,
    val_fraction: float = 0.1,
    seed: int = 0,
    tensorboard_log: Optional[Union[str, os.PathLike]] = None,
    device: str = "auto",
    verbose: int = 1,
    progress_bar: bool = False,
    **ppo_kwargs: Any,
) -> Dict[str, Any]:
    """Train PPO on an offline dataset and write checkpoints/summary."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    dataset.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = split_dataset(
        dataset, val_fraction=val_fraction, seed=seed
    )
    train_env = OfflineReplayEnv(
        train_dataset, mode="shuffled", seed=seed, render_mode=None
    )
    eval_env = OfflineReplayEnv(
        val_dataset, mode="deterministic", seed=seed + 1, render_mode=None
    )

    config = None if agent_config is None else resolve_agent_config(agent_config)
    best_path = output_dir / "best_model.zip"
    final_path = output_dir / "final_model.zip"
    callback = EarlyStopCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        patience=patience,
        min_timesteps=min_timesteps,
        best_model_save_path=best_path,
        verbose=verbose,
    )

    started = time.time()
    try:
        model = build_ppo_model(
            train_env,
            config,
            batch_size=batch_size,
            ent_coef=ent_coef,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=max(0, verbose - 1),
            **ppo_kwargs,
        )
        if verbose:
            print(
                f"[PPO] training on {len(train_dataset)} transitions "
                f"({train_dataset.num_episodes} episodes), validating on "
                f"{len(val_dataset)} transitions ({val_dataset.num_episodes} episodes)"
            )
            print(
                f"[PPO] batch_size={batch_size} ent_coef={model.ent_coef} "
                f"total_timesteps={total_timesteps}"
            )
        model.learn(
            total_timesteps=int(total_timesteps),
            callback=callback,
            progress_bar=progress_bar,
        )
        model.save(final_path)
    finally:
        train_env.close()
        eval_env.close()

    summary = {
        "algo": "ppo",
        "agent": None if config is None else config.name,
        "agent_config": None if config is None else config.as_dict(),
        "requested_timesteps": int(total_timesteps),
        "completed_timesteps": int(model.num_timesteps),
        "batch_size": int(batch_size),
        "ent_coef": float(model.ent_coef),
        "early_stopped": bool(callback.early_stopped),
        "evaluations_without_improvement": (
            callback.evaluations_without_improvement
        ),
        "patience": int(patience),
        "min_timesteps": int(min_timesteps),
        "eval_freq": int(eval_freq),
        "n_eval_episodes": int(n_eval_episodes),
        "best_mean_reward": (
            callback.best_mean_reward if callback.history else None
        ),
        "history": callback.history,
        "best_model_path": str(best_path) if best_path.exists() else None,
        "final_model_path": str(final_path),
        "train_transitions": len(train_dataset),
        "validation_transitions": len(val_dataset),
        "train_episodes": train_dataset.num_episodes,
        "validation_episodes": val_dataset.num_episodes,
        "duration_seconds": round(time.time() - started, 3),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return summary


def load_ppo(
    path: Union[str, os.PathLike],
    env: Optional[gym.Env] = None,
    *,
    device: str = "auto",
) -> PPO:
    """Load a saved PPO model (``.zip``)."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PPO model not found: {path}")
    return PPO.load(path, env=env, device=device)


def predict_action(
    model: PPO,
    observation: Any,
    *,
    deterministic: bool = True,
) -> int:
    """Run one deterministic PPO prediction and return an ``int`` action."""

    action, _ = model.predict(observation, deterministic=deterministic)
    array = getattr(action, "numpy", lambda: action)()
    return int(array)
