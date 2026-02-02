import enum
from dataclasses import dataclass
from typing import Union

import torch


class StateKey(str, enum.Enum):
    STATE = "u"
    STATE_DOT = "u_dot"
    STATE_DOT_DOT = "u_dot_dot"
    PROPERTIES = "properties"
    POSITION_NOISE = "u_noise"
    VELOCITY_NOISE = "u_dot_noise"
    ACCELERATION_NOISE = "u_dot_dot_noise"


@dataclass
class State:
    """A mesh object is a mesh with a position and velocity."""

    u: Union[torch.Tensor, None] = None
    u_dot: Union[torch.Tensor, None] = None
    u_dot_dot: Union[torch.Tensor, None] = None
    properties: Union[torch.Tensor, None] = None
    u_noise: Union[torch.Tensor, None] = None
    u_dot_noise: Union[torch.Tensor, None] = None
    u_dot_dot_noise: Union[torch.Tensor, None] = None

    def __repr__(self) -> str:
        string = (
            f"{self.__class__.__name__}("
            f"u={list(self.u.shape)}, "
            f"u_dot={list(self.u_dot.shape)}, "
            f"u_dot_dot={list(self.u_dot_dot.shape)}, "
            f"properties={list(self.properties.shape)}"
        )
        if self.u_noise is not None:
            string += f", u_noise={list(self.u_noise.shape)}"
        if self.u_dot_noise is not None:
            string += f", u_dot_noise={list(self.u_dot_noise.shape)}"
        if self.u_dot_dot_noise is not None:
            string += f", u_dot_dot_noise={list(self.u_dot_dot_noise.shape)}"
        string += ")"
        return string
