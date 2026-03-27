from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    state_representation: str = None


class EnvironmentEnum(StrEnum):
    VMAS_ROVER = "rover"
    VMAS_SALP_NAVIGATE = "salp_navigate"
    VMAS_SALP_NAVIGATE_LIDAR = "salp_navigate_lidar"
    VMAS_SALP_PASSAGE = "salp_passage"
    VMAS_BALANCE = "balance"
    VMAS_BUZZ_WIRE = "buzz_wire"
    MPE_SPREAD = "mpe_spread"
    MPE_SIMPLE = "mpe_simple"
    BOX2D_SALP = "box2d_salp"
