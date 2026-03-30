import torch
from environments.salp_navigate.utils import (
    centre_and_rotate,
    batch_discrete_frechet_distance,
    menger_curvature,
)


def calculate_distance_reward(a_pos: torch.Tensor, t_pos: torch.Tensor):
    pos_err = t_pos - a_pos  # [B, N, 2]
    dists = torch.linalg.norm(pos_err, dim=-1)
    dist_rew = -dists.mean(dim=-1)
    return dist_rew


def calculate_frechet_reward(
    a_pos: torch.Tensor, t_pos: torch.Tensor, aligned: bool = False
) -> torch.Tensor:

    if aligned:
        a_pos, t_pos = centre_and_rotate(a_pos, t_pos)

    f_dist = batch_discrete_frechet_distance(a_pos, t_pos)
    f_rew = 1 / torch.exp(f_dist)

    return -f_dist, f_rew


def calculate_centroid_reward(
    a_centroid: torch.Tensor, t_centroid: torch.Tensor
) -> torch.Tensor:

    c_dist = torch.norm(a_centroid - t_centroid, dim=1)
    c_rew = 1 / torch.exp(c_dist)

    return -c_dist, c_rew


def calculate_curvature_reward(
    a_pos: torch.Tensor, t_pos: torch.Tensor, joint_length: float
) -> torch.Tensor:

    k = menger_curvature(a_pos, joint_length)
    k_star = menger_curvature(t_pos, joint_length)

    rew = -torch.sum(torch.abs(k - k_star), dim=-1)

    return rew
