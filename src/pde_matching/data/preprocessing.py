from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torchtyping import TensorType


@dataclass
class MeanStdAccumulator:
    """Accumulate the feature statistics of a dataset in a single pass."""

    total: int = 0
    s1: np.ndarray = 0.0
    s2: np.ndarray = 0.0

    @staticmethod
    def sum(accs: list["MeanStdAccumulator"]):
        return MeanStdAccumulator(
            total=sum(acc.total for acc in accs),
            s1=sum(acc.s1 for acc in accs),
            s2=sum(acc.s2 for acc in accs),
        )

    def add(self, u: np.ndarray):
        """Add more data to the stats."""

        u = u.astype(np.float64).reshape((-1, u.shape[-1]))
        self.total += u.shape[0]
        self.s1 += u.sum(axis=0)
        self.s2 += (u**2).sum(axis=0)

    def mean_and_std(self):
        mean = (self.s1 / self.total).astype(np.float32)
        std = (np.sqrt(self.s2 / self.total - mean**2)).astype(np.float32)
        std[std == 0] = 1e-6
        return mean, std


StandardizerTensor = Union[
    TensorType["feature"],
    TensorType["node", "feature"],
    TensorType["time", "node", "feature"],
    TensorType["batch", "time", "node", "feature"],
]


@dataclass
class Standardizer:
    """Node feature standardization

    The standardization means and standard deviation can optionally be node-, time-, and
    batch-specific.
    """

    mean: StandardizerTensor
    std: StandardizerTensor

    def __post_init__(self):
        assert self.mean.shape == self.std.shape
        assert 1 <= self.mean.ndim <= 4

    @property
    def shape(self):
        return self.mean.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to(self, device, *args, **kwargs):
        return self.__class__(
            self.mean.to(device, *args, **kwargs), self.std.to(device, *args, **kwargs)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={list(self.mean.shape)}, std={list(self.std.shape)}"
            f")"
        )


@dataclass
class DummyStandardizer(Standardizer):
    """Dummy standardizer that does nothing."""

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device, *args, **kwargs):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@dataclass
class TrajectoryStandardizer(Standardizer):
    trajectory_mean: StandardizerTensor
    trajectory_std: StandardizerTensor

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.trajectory_std.unsqueeze(1) + self.trajectory_mean.unsqueeze(1)

    def to(self, device, *args, **kwargs):
        return self.__class__(
            self.mean.to(device, *args, **kwargs),
            self.std.to(device, *args, **kwargs),
            self.trajectory_mean.to(device, *args, **kwargs),
            self.trajectory_std.to(device, *args, **kwargs),
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={list(self.mean.shape)}, std={list(self.std.shape)}, "
            f"trajectory_mean={list(self.trajectory_mean.shape)}, trajectory_std={list(self.trajectory_std.shape)}"
            f")"
        )
