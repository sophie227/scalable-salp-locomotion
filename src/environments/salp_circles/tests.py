from environments.salp.utils import (
    generate_target_points,
    rotate_points,
    generate_curve,
    generate_bending_curve,
)
import torch
import math
import matplotlib.pyplot as plt

# points = generate_target_points(0, 0, 10, [-45, 45], 1.0)

# Example usage
# pts = generate_curve(0.0, 0.0, n_points=10, radius=0.09, max_dist=0.05)
# print(pts[:3], "...")  # show first 3 points


# “S” curve starting at the origin, facing +x:
base_points = 3
n_points = 4
min_radius = 0.05
max_radius = 0.5
pts = generate_bending_curve(
    0.0,
    0.0,
    n_points=n_points,
    radius=0.5 * (n_points // base_points),
    max_dist=0.05,
    n_bends=1,  # two bends  → three alternating arcs
)

angle = math.radians(0.0)  # 30° counter‑clockwise
rot_xy = rotate_points(pts, angle)


# --- plot ---------------------------------------------------------
plt.scatter(rot_xy[:, 0].cpu(), rot_xy[:, 1].cpu(), marker="o")  # x‑coords  # y‑coords
plt.xlabel("x")
plt.ylabel("y")
plt.title("Point cloud")
plt.gca().set_aspect("equal")  # optional: square axes
plt.show()
