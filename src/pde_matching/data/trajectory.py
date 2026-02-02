import copy
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch_geometric.data.data import BaseData, Data

from .preprocessing import Standardizer
from .state import State, StateKey


@dataclass
class DataTrajectory:
    data_list: List[Data]
    state_list: List[State]
    input_keys: List[StateKey]
    output_keys: List[StateKey]
    standardizer: Dict[StateKey, Standardizer]

    @classmethod
    def from_batch_list(
        cls,
        batch_list: List[BaseData],
        input_keys: List[StateKey],
        output_keys: List[StateKey],
        standardizer: Dict[StateKey, Standardizer],
    ):
        data_list = []
        state_list = []

        for data in batch_list:
            data_copy = data.clone()

            state_dict = {
                s: getattr(data, s) for s in StateKey if hasattr(data_copy, s)
            }

            state = State(
                **state_dict,
            )

            for k in state_dict:
                delattr(data_copy, k)

            data_list.append(data_copy)
            state_list.append(state)

        return cls(
            data_list=data_list,
            state_list=state_list,
            input_keys=input_keys,
            output_keys=output_keys,
            standardizer=standardizer,
        )

    def __getitem__(self, key):
        if isinstance(self.state_list[key], State):
            return self.__class__(
                data_list=[self.data_list[key]],
                state_list=[self.state_list[key]],
                input_keys=self.input_keys,
                output_keys=self.output_keys,
                standardizer=self.standardizer,
            )
        else:
            return self.__class__(
                data_list=self.data_list[key],
                state_list=self.state_list[key],
                input_keys=self.input_keys,
                output_keys=self.output_keys,
                standardizer=self.standardizer,
            )

    def __setitem__(self, key, value):
        self.data_list[key] = value.data_list
        self.state_list[key] = value.state_list

    def integrate(self, state: torch.Tensor, output: torch.Tensor, dt: float = 1.0) -> "torch.Tensor":
        if len(self.output_keys) == 1:
            if self.output_keys[0] == StateKey.STATE:
                return state + output * dt
            elif self.output_keys[0] == StateKey.STATE_DOT:
                state_dot = self.u_dot[0] + output * dt
                return state + state_dot * dt
        else:
            return state + output * dt

    def normalize(self, inplace=False):
        if inplace:
            mesh_states = self.state_list
        else:
            mesh_states = copy.deepcopy(self.state_list)

        for mesh_state in mesh_states:
            for key, standardizer in self.standardizer.items():
                attr = getattr(mesh_state, key)
                if attr is not None:
                    setattr(mesh_state, key, standardizer.do(attr))

                    if getattr(mesh_state, f"{key}_noise") is not None:
                        # Note that the standardized noise has to be shifted by mean/std
                        setattr(
                            mesh_state,
                            f"{key}_noise",
                            standardizer.do(getattr(mesh_state, f"{key}_noise"))
                            + standardizer.mean / standardizer.std,
                        )

        return self.__class__(
            data_list=self.data_list,
            state_list=mesh_states,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            standardizer=self.standardizer,
        )

    def unnormalize(self, output: torch.Tensor) -> torch.Tensor:
        output_dim = [
            standardizer.shape[-1]
            for key, standardizer in self.standardizer.items()
            if key in self.output_keys
        ]
        output_split = torch.split(output, output_dim, dim=-1)

        output_list = []
        for output, key in zip(output_split, self.output_keys):
            output_list.append(self.standardizer[key].undo(output))

        return torch.cat(output_list, dim=-1)

    def split(self, output: torch.Tensor) -> torch.Tensor:
        output_dim = [
            standardizer.shape[-1]
            for key, standardizer in self.standardizer.items()
            if key in self.output_keys
        ]
        output_split = torch.split(output, output_dim, dim=-1)
        return output_split

    def to(self, device, *args, **kwargs) -> torch.Tensor:
        mesh_states = copy.deepcopy(self.state_list)

        for mesh_state in mesh_states:
            for s in mesh_state.__dataclass_fields__:
                attr = getattr(mesh_state, s)
                if attr is not None:
                    setattr(mesh_state, s, attr.to(device, *args, **kwargs))

        meshs = [mesh.to(device, *args, **kwargs) for mesh in self.data_list]
        standardizer = {
            k: v.to(device, *args, **kwargs) for k, v in self.standardizer.items()
        }

        return self.__class__(
            data_list=meshs,
            state_list=mesh_states,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            standardizer=standardizer,
        )

    def cpu(self) -> torch.Tensor:
        mesh_states = copy.deepcopy(self.state_list)

        for mesh_state in mesh_states:
            for s in mesh_state.__dataclass_fields__:
                attr = getattr(mesh_state, s)
                if attr is not None:
                    setattr(mesh_state, s, attr.cpu())

        meshs = [mesh.cpu() for mesh in self.data_list]
        standardizer = {k: v.cpu() for k, v in self.standardizer.items()}

        return self.__class__(
            data_list=meshs,
            state_list=mesh_states,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            standardizer=standardizer,
        )

    def cuda(self, device=None) -> torch.Tensor:
        mesh_states = copy.deepcopy(self.state_list)

        for mesh_state in mesh_states:
            for s in mesh_state.__dataclass_fields__:
                attr = getattr(mesh_state, s)
                if attr is not None:
                    setattr(mesh_state, s, attr.cuda(device=device))

        meshs = [mesh.cuda(device=device) for mesh in self.data_list]
        standardizer = {k: v.cuda(device=device) for k, v in self.standardizer.items()}

        return self.__class__(
            data_list=meshs,
            state_list=mesh_states,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            standardizer=standardizer,
        )

    def clone(self):
        mesh_states = copy.deepcopy(self.state_list)

        for mesh_state in mesh_states:
            for s in mesh_state.__dataclass_fields__:
                attr = getattr(mesh_state, s)
                if attr is not None:
                    setattr(mesh_state, s, attr.clone())

        meshs = [mesh.clone() for mesh in self.data_list]

        return self.__class__(
            data_list=meshs,
            state_list=mesh_states,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            standardizer=self.standardizer,
        )

    def __len__(self):
        return len(self.state_list)

    @property
    def batch_size(self):
        return self.state_list[0].u.size(0)

    @property
    def device(self):
        return self.state_list[0].u.device

    @property
    def u_properties(self) -> torch.Tensor:
        return torch.stack([state.properties for state in self.state_list])

    @property
    def u(self) -> torch.Tensor:
        return torch.stack([state.u for state in self.state_list])

    @property
    def u_dot(self) -> torch.Tensor:
        return torch.stack([state.u_dot for state in self.state_list])

    @property
    def u_dot_dot(self) -> torch.Tensor:
        return torch.stack([state.u_dot_dot for state in self.state_list])

    @property
    def u_noise(self) -> torch.Tensor:
        if self.state_list[0].u_noise is None:
            return torch.zeros_like(self.u)
        else:
            return torch.stack([state.u_noise for state in self.state_list])

    @property
    def u_dot_noise(self) -> torch.Tensor:
        if self.state_list[0].u_dot_noise is None:
            return torch.zeros_like(self.u_dot)
        else:
            return torch.stack([state.u_dot_noise for state in self.state_list])

    @property
    def u_dot_dot_noise(self) -> torch.Tensor:
        if self.state_list[0].u_dot_dot_noise is None:
            return torch.zeros_like(self.u_dot_dot)
        else:
            return torch.stack([state.u_dot_dot_noise for state in self.state_list])

    @property
    def input_vector(self) -> torch.Tensor:
        outputs = []
        for state in self.state_list:
            outputs.append(
                torch.cat([getattr(state, s) for s in self.input_keys], dim=-1)
            )

        return torch.stack(outputs, dim=0)

    @property
    def input_vector_perturbed(self) -> torch.Tensor:
        outputs = []
        for state in self.state_list:
            output_state = []
            for s in self.input_keys:
                if (
                    hasattr(state, f"{s}_noise")
                    and getattr(state, f"{s}_noise") is not None
                ):
                    output_state.append(
                        getattr(state, s) + getattr(state, f"{s}_noise")
                    )
                else:
                    output_state.append(getattr(state, s))

            outputs.append(torch.cat(output_state, dim=-1))

        return torch.stack(outputs, dim=0)

    @property
    def u_perturbed(self) -> torch.Tensor:
        u_list = []
        for state in self.state_list:
            if state.u_noise is not None:
                u_list.append(state.u + state.u_noise)
            else:
                u_list.append(state.u)
        return torch.stack(u_list, dim=0)

    @property
    def u_dot_perturbed(self) -> torch.Tensor:
        u_dot_list = []
        for state in self.state_list:
            if state.u_dot_noise is not None:
                u_dot_list.append(state.u_dot + state.u_dot_noise)
            else:
                u_dot_list.append(state.u_dot)
        return torch.stack(u_dot_list, dim=0)

    @property
    def state_vector(self) -> torch.Tensor:
        if self.is_output_a_state_pair():
            state = torch.cat([self.u, self.u_dot], dim=-1)
        else:
            state = self.u

        return state

    @property
    def state_perturbed(self) -> torch.Tensor:
        if self.is_output_a_state_pair():
            state = torch.cat([self.u_perturbed, self.u_dot_perturbed], dim=-1)
        else:
            state = self.u_perturbed

        return state

    @property
    def state_noise(self) -> torch.Tensor:
        if self.is_output_a_state_pair():
            state = torch.cat([self.u_noise, self.u_dot_noise], dim=-1)
        else:
            state = self.u_noise

        return state

    def is_output_a_state_pair(self) -> bool:
        return all(
            [
                s in self.output_keys
                for s in [StateKey.STATE_DOT, StateKey.STATE_DOT_DOT]
            ]
        ) or all([s in self.output_keys for s in [StateKey.STATE, StateKey.STATE_DOT]])

    def get_input_dim(self) -> int:
        return sum([getattr(self.state_list[0], k).shape[-1] for k in self.input_keys])

    def get_output_dim(self) -> int:
        return sum([getattr(self.state_list[0], k).shape[-1] for k in self.output_keys])

    def get_edge_dim(self) -> int:
        return self.data_list[0].edge_attr.shape[-1]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_list={[len(self.data_list)]}, "
            f"state_list={[len(self.state_list)]}, "
            f"input_keys={[k.value for k in self.input_keys]}, "
            f"output_keys={[k.value for k in self.output_keys]}, "
            f"standardizer={[k.value for k in self.standardizer.keys()]} "
            f")"
        )
