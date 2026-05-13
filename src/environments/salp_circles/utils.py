from vmas.simulator.utils import Color
import random
import numpy as np
import torch
import math
from typing import List, Tuple

COLOR_MAP = {
    "GREEN": Color.GREEN,
    "RED": Color.RED,
    "BLUE": Color.BLUE,
    "BLACK": Color.BLACK,
    "LIGHT_GREEN": Color.LIGHT_GREEN,
    "GRAY": Color.GRAY,
    "WHITE": Color.WHITE,
    "PURPLE": (0.75, 0.25, 0.75),
    "ORANGE": (0.75, 0.75, 0.25),
    "MAGENTA": (0.9, 0.25, 0.5),
}

COLOR_LIST = [
    # Primary colors
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    # Secondary colors
    (1.0, 1.0, 0.0),  # Yellow
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 0.0, 1.0),  # Magenta
    # Tertiary colors
    (1.0, 0.5, 0.0),  # Orange
    (0.5, 1.0, 0.0),  # Chartreuse
    (0.0, 1.0, 0.5),  # Spring green
    (0.0, 0.5, 1.0),  # Azure
    (0.5, 0.0, 1.0),  # Violet
    (1.0, 0.0, 0.5),  # Rose
    # Mixed intensities
    (0.75, 0.25, 0.25),  # Pale red
    (0.25, 0.75, 0.25),  # Pale green
    (0.25, 0.25, 0.75),  # Pale blue
    (0.75, 0.75, 0.25),  # Pale yellow
    (0.25, 0.75, 0.75),  # Pale cyan
    (0.75, 0.25, 0.75),  # Pale magenta
    # Dark variants
    (0.5, 0.0, 0.0),  # Dark red
    (0.0, 0.5, 0.0),  # Dark green
    (0.0, 0.0, 0.5),  # Dark blue
    (0.5, 0.5, 0.0),  # Dark yellow
    (0.0, 0.5, 0.5),  # Dark cyan
    (0.5, 0.0, 0.5),  # Dark magenta
    # Earth tones
    (0.6, 0.3, 0.1),  # Brown
    (0.8, 0.7, 0.6),  # Tan
    (0.4, 0.3, 0.2),  # Dark brown
    (0.5, 0.4, 0.3),  # Taupe
    # Pastels
    (1.0, 0.8, 0.8),  # Pastel red
    (0.8, 1.0, 0.8),  # Pastel green
    (0.8, 0.8, 1.0),  # Pastel blue
    (1.0, 1.0, 0.8),  # Pastel yellow
    # Vibrant variants
    (1.0, 0.4, 0.4),  # Coral
    (0.4, 1.0, 0.4),  # Light green
    (0.4, 0.4, 1.0),  # Periwinkle
    (1.0, 0.84, 0.0),  # Gold
    (0.8, 0.4, 0.8),  # Orchid
    # More variations
    (0.9, 0.6, 0.2),  # Orange gold
    (0.2, 0.9, 0.6),  # Aquamarine
    (0.6, 0.2, 0.9),  # Purple
    # In-betweens
    (0.7, 0.0, 0.0),  # Brick red
    (0.0, 0.7, 0.0),  # Forest green
    (0.0, 0.0, 0.7),  # Navy blue
    # Grayscales
    (0.9, 0.9, 0.9),  # Very light gray
    (0.75, 0.75, 0.75),  # Light gray
    (0.5, 0.5, 0.5),  # Gray
    (0.25, 0.25, 0.25),  # Dark gray
    # Additional colors
    (0.55, 0.71, 0.0),  # Olive green
    (0.18, 0.31, 0.31),  # Dark slate
    (0.82, 0.41, 0.12),  # Sienna
    (0.58, 0.0, 0.83),  # Purple
    (0.0, 0.5, 0.25),  # Teal green
    (0.5, 0.25, 0.0),  # Brown
    (0.96, 0.87, 0.7),  # Wheat
    (0.25, 0.88, 0.82),  # Turquoise
    (0.93, 0.51, 0.93),  # Violet
    (0.99, 0.0, 0.0),  # Crimson
    (0.67, 0.43, 0.16),  # Raw sienna
    (1.0, 0.39, 0.28),  # Tomato
    (0.29, 0.0, 0.51),  # Indigo
    (0.42, 0.56, 0.14),  # Olive drab
    (0.44, 0.5, 0.56),  # Slate
    (0.96, 0.96, 0.86),  # Light yellow
    (0.4, 0.8, 0.67),  # Medium aquamarine
    (0.8, 0.52, 0.25),  # Peru
    (0.39, 0.58, 0.93),  # Cornflower blue
]


def sample_filtered_normal(mean, std_dev, threshold):
    while True:
        # Sample a single value from the normal distribution
        value = random.normalvariate(mu=mean, sigma=std_dev)
        # Check if the value is outside the threshold range
        if abs(value) > threshold:
            return value


def generate_target_points(
    x: float, y: float, n_points: int, theta_range: list, d_max: float
):
    """
    Generate n_points points starting from (x, y), where each point is positioned
    at a fixed distance (d_max) from the previous point at a random angle within theta_range.

    Parameters:
        x (float): Starting x-coordinate.
        y (float): Starting y-coordinate.
        n_points (int): Total number of points to generate.
        theta_range (tuple): Angle range in degrees (min_angle, max_angle).
        d_max (float): Fixed distance between consecutive points.

    Returns:
        list: List of tuples containing the generated (x, y) coordinates.
    """
    points = [torch.tensor((x, y))]  # Initialize with the starting point

    for _ in range(n_points - 1):
        # Generate a random angle within the theta_range
        theta = np.radians(np.random.uniform(theta_range[0], theta_range[1]))

        # Calculate the new point
        x_new = points[-1][0] + d_max * np.cos(theta)
        y_new = points[-1][1] + d_max * np.sin(theta)

        # Append the new point to the list
        points.append(torch.tensor((x_new, y_new)))

    return points


def rotate_points(points, angle_rad):
    """
    Rotate a list of (x, y) tensors around the first point.

    Args
    ----
    points      : list[Tensor]  length N, each shape (2,)
    angle_rad   : float         rotation angle in **radians**

    Returns
    -------
    rotated     : Tensor shape (N, 2)  (same device / dtype as input)
    """
    xy = torch.stack(points)  # (N, 2)
    pivot = xy[0]  # shape (2,)

    # 1) translate so pivot → origin
    rel = xy - pivot  # (N, 2)

    # 2) rotation matrix
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = torch.tensor([[c, -s], [s, c]], dtype=xy.dtype, device=xy.device)  # (2, 2)

    # 3) rotate & translate back
    rotated = rel @ R.T + pivot  # (N, 2)
    return rotated


def batch_discrete_frechet_distance(batch_P, batch_Q):
    """
    Compute the discrete Fréchet distance between two batched tensors of points.

    Parameters:
        batch_P (torch.Tensor): Tensor of shape [B, N, 2] representing B sets of N points (x, y).
        batch_Q (torch.Tensor): Tensor of shape [B, M, 2] representing B sets of M points (x, y).

    Returns:
        torch.Tensor: Tensor of shape [B] containing the discrete Fréchet distance for each batch.
    """
    B, N, _ = batch_P.shape
    _, M, _ = batch_Q.shape

    # Initialize a large distance matrix for each batch
    ca = torch.full((B, N, M), -1.0, device=batch_P.device)

    def recursive_frechet(ca, P, Q, i, j, b):
        if ca[b, i, j] > -1:  # Use cached value
            return ca[b, i, j]

        # Compute Euclidean distance between P[i] and Q[j] for the current batch
        dist = torch.norm(P[b, i] - Q[b, j], p=2)

        if i == 0 and j == 0:  # Base case
            ca[b, i, j] = dist
        elif i == 0:  # First row
            ca[b, i, j] = torch.max(recursive_frechet(ca, P, Q, i, j - 1, b), dist)
        elif j == 0:  # First column
            ca[b, i, j] = torch.max(recursive_frechet(ca, P, Q, i - 1, j, b), dist)
        else:  # General case
            ca[b, i, j] = torch.max(
                torch.min(
                    torch.stack(
                        [
                            recursive_frechet(ca, P, Q, i - 1, j, b),
                            recursive_frechet(ca, P, Q, i - 1, j - 1, b),
                            recursive_frechet(ca, P, Q, i, j - 1, b),
                        ]
                    ),
                ),
                dist,
            )
        return ca[b, i, j]

    # Iterate over each batch and compute the Fréchet distance
    for b in range(B):
        recursive_frechet(ca, batch_P, batch_Q, N - 1, M - 1, b)

    return ca[:, -1, -1]  # Return the Fréchet distance for each batch


def angle_between_vectors(v1, v2):
    """
    Calculate the angle (in radians) between two vectors using PyTorch.

    Parameters:
        v1 (torch.Tensor): Tensor of shape [N, D] representing N vectors.
        v2 (torch.Tensor): Tensor of shape [N, D] representing N vectors.

    Returns:
        torch.Tensor: Tensor of shape [N] containing angles in radians.
    """
    # Compute dot product
    dot_product = torch.sum(v1 * v2, dim=1)

    # Compute magnitudes (L2 norms)
    norm_v1 = torch.norm(v1, p=2, dim=1)
    norm_v2 = torch.norm(v2, p=2, dim=1)

    # Compute cosine similarity
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # Avoid division by zero

    # Clamp values to avoid numerical errors in arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians
    angle = torch.acos(cos_theta)

    return angle


def is_within_any_range(number, ranges):
    """
    Check if a number is within any of the given ranges.

    Parameters:
        number (float or torch.Tensor): The number to check.
        ranges (list of tuples): List of (min, max) tuples representing the ranges.

    Returns:
        torch.Tensor (bool): True if the number is within any range, False otherwise.
    """
    # Convert ranges to a tensor of shape [N, 2]
    range_tensor = torch.tensor(ranges)  # Shape: [N, 2]

    # Extract min and max values
    range_min = range_tensor[:, 0]  # First column (min values)
    range_max = range_tensor[:, 1]  # Second column (max values)

    # Check if the number is inside any range
    inside_any_range = (number >= range_min) & (number <= range_max)

    # Return True if the number is in any range
    return torch.any(inside_any_range)


def closest_number(target, numbers):
    """
    Given a target number and a set of numbers, return the number closest to the target.

    Parameters:
        target (float or torch.Tensor): The target number.
        numbers (list of float or torch.Tensor): List of numbers to compare against.

    Returns:
        torch.Tensor: The closest number.
    """
    # Convert numbers to a PyTorch tensor
    numbers_tensor = torch.tensor(numbers)  # Shape: [N]

    # Compute absolute differences
    differences = torch.abs(numbers_tensor - target)  # Shape: [N]

    # Get the index of the minimum difference
    closest_index = torch.argmin(differences)

    # Retrieve the closest number
    closest_value = numbers_tensor[closest_index]

    return closest_value


def angular_velocity(R, V):
    """
    Computes the angular velocity given a distance vector and a velocity vector in 2D.

    Parameters:
        R (torch.Tensor): Tensor of shape [N, 2] representing the distance vectors.
        V (torch.Tensor): Tensor of shape [N, 2] representing the velocity vectors.

    Returns:
        torch.Tensor: Tensor of shape [N] representing the angular velocities.
    """
    # Compute 2D cross product: R_x * V_y - R_y * V_x
    cross_product = R[:, 0] * V[:, 1] - R[:, 1] * V[:, 0]

    # Compute squared norm of R (add small epsilon to avoid division by zero)
    r_norm_sq = (R**2).sum(dim=1) + 1e-8

    # Compute angular velocity
    omega = cross_product / r_norm_sq

    return omega


def generate_curve(
    x0: float,
    y0: float,
    n_points: int,
    radius: float,
    max_dist: float,
    clockwise: bool = False,
) -> List[Tuple[float, float]]:
    """
    Return `n_points` (x, y) samples lying on an arc of a circle with
    the given `radius`, starting from (x0, y0).  The chord (straight‑line)
    distance between successive points is ≤ `max_dist`.

    Parameters
    ----------
    x0, y0 : float
        Coordinates of the starting point (included in the output).
    n_points : int
        Total number of points to emit (must be ≥ 2 for a curve).
    radius : float
        Radius of curvature.  Positive; units are the same as the coords.
    max_dist : float
        Upper bound for the distance between every pair of neighbours.
    clockwise : bool, optional
        If True, walk along the arc clockwise; else counter‑clockwise.

    Returns
    -------
    List[Tuple[float, float]]
        Sequence of (x, y) coordinates on the arc.

    Notes
    -----
    ‑ The function automatically clips `max_dist` to `[0, 2 r]`.
    ‑ The centre of the circle is chosen so that (x0, y0) is the
      “south‑most” point if `clockwise=False` (centre above the start)
      or the “north‑most” point if `clockwise=True`.
    ‑ The angular step Δθ is the largest angle that still honours the
      chord‑length constraint:
          Δθ_max = 2 asin(max_dist / (2 r))
      and the algorithm uses that same Δθ for every segment.
    """
    if n_points < 1:
        raise ValueError("n_points must be ≥ 1")
    if n_points == 1:
        return [(x0, y0)]
    if radius <= 0:
        raise ValueError("radius must be positive")

    # Clip max_dist to a feasible range
    max_dist = max(0.0, min(max_dist, 2 * radius))

    # Largest permissible angular step
    max_delta_theta = 2.0 * math.asin(max_dist / (2.0 * radius))

    # Use that same Δθ for every segment
    delta_theta = max_delta_theta
    if clockwise:
        delta_theta = -delta_theta  # reverse the sign for CW motion

    # Choose the circle centre so the start point is at angle −π/2 (CW)
    # or +π/2 (CCW) relative to the centre.  This keeps the first tangent
    # horizontal, but you can re‑orient easily if you need to.
    cx, cy = x0, y0 + radius if not clockwise else y0 - radius
    theta0 = -math.pi / 2.0 if not clockwise else +math.pi / 2.0

    pts = []
    for k in range(n_points):
        theta_k = theta0 + k * delta_theta
        xk = cx + radius * math.cos(theta_k)
        yk = cy + radius * math.sin(theta_k)
        pts.append(torch.tensor((xk, yk)))

    return pts


def generate_bending_curve(
    x0: float,
    y0: float,
    n_points: int,
    radius: float,
    max_dist: float,
    n_bends: int = 2,
    start_heading: float = 0.0,
    start_left: bool = True,
) -> List[Tuple[float, float]]:
    """
    Create a poly‑arc composed of `n_bends + 1` circular segments whose
    signed curvature alternates, e.g.   left→right→left  for n_bends = 2.

    Parameters
    ----------
    x0, y0        : float
        Coordinates of the first point (always included).
    n_points      : int
        Total number of samples along the whole curve  (≥ 2).
    radius        : float
        Constant radius for every arc segment.
    max_dist      : float
        Upper bound on the straight‑line distance between neighbours.
    n_bends       : int, default 2
        Number of *direction reversals* in curvature.
    start_heading : float, default 0.0 (radians)
        Tangent direction at the very first point (0 ⇒ +x axis).
    start_left    : bool, default True
        If True the first bend turns *left* (CCW); otherwise right (CW).

    Returns
    -------
    List[(x, y)]
        `n_points` coordinates tracing the bending curve.
    """
    if n_points < 1:
        raise ValueError("n_points must be ≥ 1")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if max_dist <= 0:
        raise ValueError("max_dist must be positive")

    # Angular step that honours the chord‑length constraint
    max_dist = min(max_dist, 2 * radius)  # can’t exceed diameter
    dtheta = 2.0 * math.asin(max_dist / (2.0 * radius))  # always positive

    # Split the point budget as evenly as possible over all segments
    n_segments = n_bends + 1
    base = n_points // n_segments
    extra = n_points % n_segments  # first ‘extra’ segments get +1
    seg_lengths = [base + (1 if i < extra else 0) for i in range(n_segments)]
    seg_lengths[0] -= 1  # first point already accounted for

    # Initial state
    pts = [torch.tensor((x0, y0))]
    x, y = x0, y0
    heading = start_heading  # tangent angle φ
    sign = 1 if start_left else -1  # +1 ⇒ left/CCW, −1 ⇒ right/CW

    for seg_idx, steps in enumerate(seg_lengths):
        for _ in range(steps):
            # Circle centre lies ±radius along the normal to the heading
            cx = x - sign * radius * math.sin(heading)
            cy = y + sign * radius * math.cos(heading)

            # Vector from centre to current point
            rx, ry = x - cx, y - cy

            # Rotate that vector by ±dθ around the centre
            cosd, sind = math.cos(sign * dtheta), math.sin(sign * dtheta)
            rx_new = cosd * rx - sind * ry
            ry_new = sind * rx + cosd * ry
            x, y = cx + rx_new, cy + ry_new

            # New heading (tangent turns by the same signed dθ)
            heading += sign * dtheta
            pts.append(torch.tensor((x, y)))

        # Flip curvature direction for the next segment
        sign *= -1

    return pts


def calculate_moment(position, force):
    """
    Calculate the moment generated by a 2D force at a given position using PyTorch.

    Parameters:
        position (torch.Tensor): Tensor of shape [N, 2], where each row is (x_r, y_r).
        force (torch.Tensor): Tensor of shape [N, 2], where each row is (x_f, y_f).

    Returns:
        torch.Tensor: Tensor of shape [N] containing the moment for each pair of position and force.
    """
    # Ensure tensors are of the same shape
    assert (
        position.shape == force.shape
    ), "Position and force tensors must have the same shape."

    # Extract components
    x_r, y_r = position[:, 0], position[:, 1]
    x_f, y_f = force[:, 0], force[:, 1]

    # Compute the moment
    moment = x_r * y_f - y_r * x_f

    return moment


def wrap_to_pi(angle):
    """Map any angle array to (-π, π]."""
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def unwrap(p, discont=torch.pi, axis=-1):
    """
    Unwraps a phase array by changing absolute differences greater than discont to their 2*pi complement.

    Args:
        p (torch.Tensor): Input array of phase angles.
        discont (float or torch.Tensor): Discontinuity threshold.
        axis (int): Axis along which to unwrap.

    Returns:
        torch.Tensor: Unwrapped phase array.
    """
    p_diff = torch.diff(p, dim=axis)

    # Identify discontinuities
    mask = torch.abs(p_diff) > discont

    # Calculate cumulative correction factors
    angles = torch.cumsum(2 * torch.pi * torch.sign(p_diff) * mask, dim=axis)

    # Pad with a zero at the beginning to match original size
    pad_shape = list(p.shape)
    pad_shape[axis] = 1

    angles_padded = torch.cat(
        (torch.zeros(pad_shape, dtype=p.dtype, device=p.device), angles), dim=axis
    )

    # Apply corrections
    unwrapped_p = p + angles_padded

    return unwrapped_p


def internal_angles_xy(points: torch.Tensor) -> torch.Tensor:
    """
    Compute Δθ for a planar chain from **positions**.

    Args
    ----
    points : (*, N, 2) tensor
        (x, y) positions of N robots; * can be a batch dimension.

    Returns
    -------
    dtheta : (*, N‑2) tensor
        Relative heading at each internal joint.
    """
    # segment vectors v_i = p_{i+1} - p_i   shape: (*, N‑1, 2)
    v = points[:, 1:, :] - points[:, :-1, :]

    # headings φ_i = atan2(v_y, v_x)        shape: (*, N‑1)
    phi = torch.atan2(v[..., 1], v[..., 0])

    # Δθ_j = wrap(φ_j − φ_{j-1})            shape: (*, N‑2)
    internal_angles = wrap_to_pi(phi[..., 1:] - phi[..., :-1])
    return internal_angles, phi


def get_neighbor_angles(
    positions: torch.Tensor, idx: int, n_agents: int
) -> torch.Tensor:

    if idx == 0:
        v = torch.zeros_like(positions[:, 0, :])
        v = torch.stack([v, positions[:, idx + 1, :] - positions[:, idx, :]], dim=1)
    elif idx == (n_agents - 1):
        v = positions[:, idx - 1, :] - positions[:, idx, :]
        v = torch.stack([v, torch.zeros_like(v)], dim=1)
    else:
        v = (
            positions[:, idx - 1 : idx + 2 : 2, :] - positions[:, [idx], :]
        )  # brackets on idx preserve dimension after index

    # headings φ_i = atan2(v_y, v_x)        shape: (*, N‑1)
    angles = torch.atan2(v[..., 1], v[..., 0])

    return angles


def internal_angles_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """
    Alternative: compute Δθ directly from robot yaw readings.

    yaw : (*, N) tensor of body yaw angles in radians.
    Returns the same shape (*, N‑2) as above.
    """
    return wrap_to_pi(yaw[..., 1:] - yaw[..., :-1])


def bending_speed(
    dtheta: torch.Tensor, dtheta_prev: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    First‑order time derivative Δθ̇ (bending velocity).
    """
    return wrap_to_pi(dtheta - dtheta_prev) / dt


def menger_curvature(p: torch.Tensor, link_length: float) -> torch.Tensor:
    """
    p : (B, N, 2) positions in body frame
    return κ : (B, N-2) curvature of internal joints
    """
    p_prev, p_i, p_next = p[:, :-2], p[:, 1:-1], p[:, 2:]
    a = (p_prev - p_i).norm(dim=-1)
    b = (p_next - p_i).norm(dim=-1)
    c = (p_next - p_prev).norm(dim=-1)
    # twice triangle area (2-D determinant)
    area2 = torch.abs(
        (p_i - p_prev)[..., 0] * (p_next - p_i)[..., 1]
        - (p_i - p_prev)[..., 1] * (p_next - p_i)[..., 0]
    )
    # κ = 2*Area / (a b c)
    kappa = 2.0 * area2 / (a * b * c + 1e-8)
    return kappa * link_length


def centre_and_rotate(points, goal_points):
    p_cent = points.mean(dim=1, keepdim=True)
    g_cent = goal_points.mean(dim=1, keepdim=True)
    P = points - p_cent
    G = goal_points - g_cent
    # # 2-D Umeyama alignment (no scale)
    # H = P.transpose(2, 1) @ G
    # U, _, Vt = torch.linalg.svd(H)
    # R = Vt.transpose(2, 1) @ U.transpose(2, 1)
    # if torch.det(R) < 0:  # reflection fix
    #     Vt[-1] *= -1
    #     R = Vt.transpose(2, 1) @ U.transpose(2, 1)
    # P_aligned = (R @ P.transpose(2, 1)).transpose(2, 1)
    return P, G  # curves centred & best-rotated


def one_hot_encode(id: int, num_classes: int) -> torch.Tensor:
    """
    Returns a one-hot encoded tensor for a given ID.

    Args:
        id (int): The class index to encode.
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: A one-hot encoded 1D tensor of shape (num_classes,)
    """
    one_hot = torch.zeros(num_classes)
    one_hot[id] = 1.0
    return one_hot


def binary_encode(number: int, num_bits: int, device=None) -> torch.Tensor:
    """
    Encode an integer as a binary tensor of 0s and 1s with specified bit length.

    Args:
        number: Integer to encode
        num_bits: Number of bits to use for encoding
        device: PyTorch device for the tensor

    Returns:
        Tensor of 0s and 1s representing the binary encoding
    """
    if number >= 2**num_bits:
        raise ValueError(f"Number {number} cannot be encoded with {num_bits} bits")

    binary = []
    for i in range(num_bits - 1, -1, -1):
        bit = (number >> i) & 1
        binary.append(bit)

    return torch.tensor(binary, dtype=torch.float32, device=device)


def random_point_around_center(center_x, center_y, radius):
    """
    Generates a random (x, y) coordinate around a given circle center within the specified radius.

    Parameters:
        center_x (float): The x-coordinate of the circle center.
        center_y (float): The y-coordinate of the circle center.
        radius (float): The radius around the center where the point will be generated.

    Returns:
        tuple: A tuple (x, y) representing the random point.
    """
    # Generate a random angle in radians
    angle = random.uniform(0, 2 * math.pi)
    # Generate a random distance from the center, within the circle
    distance = random.uniform(0, radius)

    # Calculate the x and y coordinates
    random_x = center_x + distance * math.cos(angle)
    random_y = center_y + distance * math.sin(angle)

    return np.float64(random_x), np.float64(random_y)


def generate_random_coordinate_outside_box(
    offset: float, scale: float, x_boundary: float, y_boundary: float
):
    x_scaled = x_boundary * scale
    y_scaled = y_boundary * scale

    x_coord = random.uniform(-x_scaled, x_scaled)

    y_coord = random.uniform(-y_scaled, y_scaled)

    if x_coord > 0:
        x_coord += offset
    else:
        x_coord -= offset

    if y_coord > 0:
        y_coord += offset
    else:
        y_coord -= offset

    return np.float64(x_coord), np.float64(y_coord)


def generate_random_coordinate_coordinate_inside_box(
    offset_x: float, offset_y: float, x_boundary: float, y_boundary: float
):

    x_coord = random.uniform(-x_boundary, x_boundary) + offset_x

    y_coord = random.uniform(-y_boundary, y_boundary) + offset_y

    return np.float64(x_coord), np.float64(y_coord)


def get_neighbors(my_list, index):
    before = after = None
    try:
        before = my_list[index - 1]
    except IndexError:
        before = None

    try:
        after = my_list[index + 1]
    except IndexError:
        after = None

    return before, after
