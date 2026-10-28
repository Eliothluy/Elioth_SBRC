"""Neural network architectures for the paper's DRL agents.

The DQN agent uses the hybrid architecture of the paper: a 1-D convolutional
encoder over the 85-dim KPM vector followed by three dense layers of sizes
``[512, 512, 256]`` and a Random Ensemble Mixture (REM) head bank.

Importing this module requires PyTorch; ``import nsoran.drl`` itself does not
import it (see ``nsoran/drl/__init__.py``).
"""

from __future__ import annotations

from typing import Iterable, Sequence

try:  # optional dependency group: drl
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "nsoran.drl.networks requires PyTorch. Install the DRL extras with "
        "`pip install -e '.[drl]'` or `pip install -r requirements-drl.txt`."
    ) from exc

from .config import N_ACTIONS, STATE_DIM


class CNN1DEncoder(nn.Module):
    """Small 1-D CNN applied to the flattened 85-dim KPM vector.

    ``obs`` is interpreted as a length-85 signal with a single input channel.
    Two strided convolutions reduce it to a compact feature vector; the exact
    flattened size is computed at construction time and exposed as
    :attr:`feature_dim`.
    """

    def __init__(
        self,
        obs_dim: int = STATE_DIM,
        channels: Sequence[int] = (32, 64),
        kernel_sizes: Sequence[int] = (5, 3),
        strides: Sequence[int] = (2, 2),
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        self.obs_dim = int(obs_dim)
        channels = tuple(int(c) for c in channels)
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        strides = tuple(int(s) for s in strides)
        if not (
            len(channels) == len(kernel_sizes) == len(strides)
        ):
            raise ValueError(
                "channels, kernel_sizes and strides must have the same length."
            )
        if any(c <= 0 for c in channels):
            raise ValueError("channels must be positive.")
        if any(k <= 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must be positive.")
        if any(s <= 0 for s in strides):
            raise ValueError("strides must be positive.")

        layers = []
        in_channels = 1
        for out_channels, kernel_size, stride in zip(
            channels, kernel_sizes, strides
        ):
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.obs_dim)
            self.feature_dim = int(self.conv(dummy).reshape(1, -1).shape[1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(
                "observations must have shape [batch, obs_dim], got "
                f"{tuple(observations.shape)}."
            )
        if observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"observations must have {self.obs_dim} features, got "
                f"{observations.shape[1]}."
            )
        signal = observations.unsqueeze(1)  # [B, 1, obs_dim]
        return self.conv(signal).reshape(observations.shape[0], -1)


class REMQNetwork(nn.Module):
    """1-D CNN + dense trunk + ``k_heads`` Q-value heads (REM).

    Forward returns a tensor of shape ``[batch, k_heads, n_actions]``.
    """

    def __init__(
        self,
        obs_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dims: Iterable[int] = (512, 512, 256),
        k_heads: int = 3,
        channels: Sequence[int] = (32, 64),
        kernel_sizes: Sequence[int] = (5, 3),
        strides: Sequence[int] = (2, 2),
    ) -> None:
        super().__init__()
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if k_heads <= 0:
            raise ValueError("k_heads must be positive.")
        hidden_dims = tuple(int(d) for d in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        if any(d <= 0 for d in hidden_dims):
            raise ValueError("hidden_dims values must be positive.")

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.k_heads = int(k_heads)
        self.hidden_dims = hidden_dims

        self.encoder = CNN1DEncoder(
            obs_dim=self.obs_dim,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
        )
        trunk = []
        in_features = self.encoder.feature_dim
        for hidden in hidden_dims:
            trunk.extend(
                [nn.Linear(in_features, hidden), nn.ReLU(inplace=True)]
            )
            in_features = hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleList(
            [nn.Linear(in_features, self.n_actions) for _ in range(self.k_heads)]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape ``[batch, k_heads, n_actions]``."""

        features = self.trunk(self.encoder(observations))
        return torch.stack(
            [head(features) for head in self.heads], dim=1
        )

    def mean_q(self, observations: torch.Tensor) -> torch.Tensor:
        """Mean Q-values over the REM heads, shape ``[batch, n_actions]``."""

        return self.forward(observations).mean(dim=1)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["CNN1DEncoder", "REMQNetwork"]
