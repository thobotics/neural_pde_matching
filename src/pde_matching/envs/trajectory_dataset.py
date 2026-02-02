import copy
import os
from abc import abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from pde_matching.utils.common_utils import noise_like
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset


def load_trajectory(filename: str, idx: int, stage: str):
    """
    Load the dataset trajectory from a file. Give it its id as metadata

    Args:
        filename (str): The path to the file containing the dataset.
    """
    data = torch.load(filename)  # loads the data for one full trajectory

    # We now append the trajectory idx and the timestep to each graph as metadata
    for step_idx, step_graph in enumerate(data):
        step_graph.trajectory_idx = f"{stage}_{idx}"
        step_graph.step_idx = step_idx
    return data


class TrajectoryDatasetMixin(Dataset):
    def __init__(
        self,
        window_size: int,
        window_shift: int,
        ignore_n_initial_steps: int = 0,
        **kwargs,
    ):
        self.window_shift = window_shift
        self.window_size = window_size
        self.ignore_n_initial_steps = ignore_n_initial_steps

        super().__init__(**kwargs)
        self._generate_windows()

        self.perm_idx = [
            np.arange(0, self.n_sequences),
            [
                np.arange(0, self.n_windows_per_sequence)
                for _ in range(self.n_sequences)
            ],
        ]
        
        self._should_load_all_in_ram = self._check_load_all_in_ram()
        self._ram_cache = {} if self._should_load_all_in_ram else None
        
        if self._should_load_all_in_ram:
            self._load_all_trajectories()

    def _check_load_all_in_ram(self) -> bool:
        effective_trajectory_length = self.trajectory_length - self.ignore_n_initial_steps
        return self.window_size + self.window_shift >= effective_trajectory_length
    
    def _load_all_trajectories(self):
        for seq_idx in range(self.n_sequences):
            filename = os.path.join(self.processed_dir, self.stage, f"{seq_idx}.pt")
            self._ram_cache[seq_idx] = load_trajectory(filename, seq_idx, stage=self.stage)

    @property
    def interesting_indices(self) -> Tensor:
        return None

    @property
    def trajectory_length(self) -> int:
        raise NotImplementedError

    @property
    def n_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get(self, idx: int) -> BaseData:
        raise NotImplementedError

    def len(self) -> int:
        return self.n_sequences * self.n_windows_per_sequence

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union["Dataset", Tuple["Dataset", Tensor]]:
        self.perm_idx = [
            np.random.permutation(self.n_sequences),
            [
                np.random.permutation(self.n_windows_per_sequence)
                for _ in range(self.n_sequences)
            ],
        ]
        return self

    def get_interesting_batch(self, batch_size: int) -> Tensor:
        trajectory_list = []
        if self.interesting_indices is None:
            indices = np.random.choice(
                self.n_sequences * self.n_windows_per_sequence, batch_size, replace=False
            )
        else:
            indices = self.interesting_indices[:batch_size]

        for i in indices:
            data = self.get(i)
            if self.transform is not None:
                data = self.transform(data)
            trajectory_list.append(data)

        data_batch = []
        for t in range(len(trajectory_list[0])):
            data_list = [trajectory_list[i][t] for i in range(batch_size)]
            data_batch.append(Batch.from_data_list(data_list))

        return data_batch

    def _generate_windows(self):
        seq_len = self.window_size
        seq_idxs = range(self.n_sequences)

        self.windows = {}
        for seq in seq_idxs:
            self.windows[seq] = []
            for start in range(
                self.ignore_n_initial_steps,
                self.trajectory_length - seq_len,
                self.window_shift,
            ):
                self.windows[seq].append((start, start + seq_len, 1))

        self.n_windows_per_sequence = len(self.windows[0])

    def _get_indices(self, index):
        seq_idx = index // self.n_windows_per_sequence
        window_idx = index % self.n_windows_per_sequence
        return seq_idx, window_idx

    def _get_permuted_indicies(self, seq_idx, window_idx):
        perm_seq_idx = self.perm_idx[0][seq_idx]
        perm_window_idx = self.perm_idx[1][seq_idx][window_idx]
        return perm_seq_idx, perm_window_idx

    def _add_noise_to_data(self, data_list):
        for i in range(len(data_list)):
            u_noise = noise_like(data_list[i].u, self.training_noise_std)
            u_dot_noise = noise_like(data_list[i].u_dot, self.training_noise_std)
            u_dot_dot_noise = noise_like(
                data_list[i].u_dot_dot, self.training_noise_std
            )

            # Noise is added to the position here before the transform
            # Note that the pos is always before the other physical quantities
            data_list[i].pos = data_list[i].pos + u_noise[:, :3]
            data_list[i].u_noise = u_noise
            data_list[i].u_dot_noise = u_dot_noise
            data_list[i].u_dot_dot_noise = u_dot_dot_noise

    def get(self, idx: int) -> BaseData:
        seq_idx, window_idx = self._get_indices(idx)
        perm_seq_idx, perm_window_idx = self._get_permuted_indicies(seq_idx, window_idx)

        if self._should_load_all_in_ram:
            cache_data = self._ram_cache[perm_seq_idx]
        else:
            if seq_idx != self.cache["idx"]:
                filename = os.path.join(
                    self.processed_dir, self.stage, f"{perm_seq_idx}.pt"
                )
                data = load_trajectory(filename, perm_seq_idx, stage=self.stage)
                self.cache["idx"] = seq_idx
                self.cache["trajectory"] = data

            cache_data = self.cache["trajectory"]

        sliced_idx = slice(*self.windows[perm_seq_idx][perm_window_idx])

        data_list = copy.deepcopy(cache_data[sliced_idx])

        if self.stage == "train" and self.training_noise:
            self._add_noise_to_data(data_list)

        return data_list
