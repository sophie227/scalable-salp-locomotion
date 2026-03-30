from dataclasses import dataclass
from environments.types import EnvironmentParams
import torch


@dataclass(frozen=True)
class GlobalObservation:
    # Passage positions
    passage_pos: torch.Tensor
    # Menger_curvature
    curvature: torch.Tensor

    # Internal angles
    a_chain_internal_angles: torch.Tensor
    a_chain_internal_angles_speed: torch.Tensor

    # Link angles
    a_chain_link_angles: torch.Tensor
    a_chain_link_angles_speed: torch.Tensor

    # Relative angles
    a_chain_relative_angles: torch.Tensor
    a_chain_relative_angles_speed: torch.Tensor

    # Raw Global
    t_chain_all_pos: torch.Tensor
    a_chain_all_pos: torch.Tensor
    a_chain_all_vel: torch.Tensor
    a_chain_all_ang_pos: torch.Tensor
    a_chain_all_ang_vel: torch.Tensor
    a_chain_all_forces: torch.Tensor

    # Condensed global
    t_chain_centroid_pos: torch.Tensor
    a_chain_centroid_pos: torch.Tensor
    a_chain_centroid_vel: torch.Tensor
    a_chain_centroid_ang_pos: torch.Tensor
    a_chain_centroid_ang_vel: torch.Tensor
    total_force: torch.Tensor
    total_moment: torch.Tensor
    frechet_dist: torch.Tensor


@dataclass
class SalpPassageEnvironmentParams(EnvironmentParams):
    rotating_salps: bool = False


class Chain:
    def __init__(self, path: list):
        self.path = path
        self.centroid = self.calculate_centroid()
        # self.orientation = self.calculate_orientation()

    def calculate_centroid(self):

        centroid = self.path.mean(dim=0)

        return centroid

    def calculate_orientation(self):
        """
        Applies the atan2 function to each point's (y, x) coordinates in a tensor of shape (1, n_points, 2).

        Parameters:
            path (torch.Tensor): Tensor of shape (1, n_points, 2), where each entry is (x, y).

        Returns:
            torch.Tensor: Tensor of shape (1, n_points) containing the angles in radians.
        """

        # Extract x and y coordinates
        x = self.path[..., 0] - self.path[:, 0, 0]  # Shape: (1, n_points)
        y = self.path[..., 1] - self.path[:, 0, 1]  # Shape: (1, n_points)

        # Compute the angle using atan2(y, x)
        angles = torch.atan2(y, x)  # Shape: (1, n_points)

        # return angles.mean(dim=1) % (2 * torch.pi)

        orientation = (angles[:, 1:].mean(dim=1) + torch.pi) % (2 * torch.pi) - torch.pi

        if orientation < 0:
            orientation += torch.pi

        return orientation
