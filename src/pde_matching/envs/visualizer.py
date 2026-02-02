from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data


class Visualizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def visualize(
        self,
        u: torch.Tensor,
        u_hat: torch.Tensor,
        u_dot_hat: torch.Tensor,
        t: int,
        data_batch: Data,
        properties: torch.Tensor,
        umin: torch.Tensor,
        umax: torch.Tensor,
        **kwargs,
    ):
        pass
