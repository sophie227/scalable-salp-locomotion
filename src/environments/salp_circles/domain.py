#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor
from vmas.simulator.joints import Joint
from vmas.simulator.core import Entity, Agent, Landmark, Box, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils
from vmas.simulator.sensors import Lidar

from environments.salp_circles.dynamics import SalpDynamics
from environments.salp_circles.utils import (
    COLOR_LIST,
    COLOR_MAP,
    generate_target_points,
    generate_bending_curve,
    batch_discrete_frechet_distance,
    generate_random_coordinate_coordinate_inside_box,
    rotate_points,
    calculate_moment,
    internal_angles_xy,
    wrap_to_pi,
    # menger_curvature,
    get_neighbor_angles,
    binary_encode,
)
from environments.salp_circles.rewards import (
    calculate_centroid_reward,
    # calculate_curvature_reward,
    calculate_distance_reward,
    calculate_frechet_reward,
)
from environments.salp_circles.types import GlobalObservation
import random
import math
from copy import deepcopy

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

torch.set_printoptions(precision=5)


class SalpCirclesDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # CONSTANTS
        self.agent_radius = 0.02
        self.agent_joint_length = 0.07
        self.agent_max_angle = 45
        self.agent_min_angle = -45
        self.u_multiplier = 1.0
        self.target_radius = self.agent_radius / 2
        self.frechet_thresh = .90
        self.min_n_agents = 8
        self.lidar_range = 0.8
        self.lidar_rays = 2
        self.n_collision_landmarks = kwargs.pop("n_collision_landmarks", 1)
        self.collision_radius = .2
        self.collision_penalty = -0.1

        if self.n_collision_landmarks < 1:
            raise ValueError("n_collision_landmarks must be >= 1")

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.45)

        # Agents
        self.n_agents = kwargs.pop("n_agents", self.min_n_agents)
        self.state_representation = kwargs.pop("state_representation", "local")
        self.agent_chains = [None for _ in range(batch_dim)]
        self.rotating_salps = kwargs.pop("rotating_salps", False)

        
        # Environment
        self.world_x_dim = self.agent_joint_length * self.n_agents * 4
        self.world_y_dim = self.agent_joint_length * self.n_agents * 4
        self.chain_length = (self.n_agents - 1) * self.agent_joint_length
        self.max_spawn_radius = self.chain_length + self.agent_radius

        self.goal_radius = self.agent_joint_length * self.n_agents

        self.free_y_dim = self.world_y_dim - self.goal_radius
        self.agent_starting_y = -self.world_y_dim + (self.free_y_dim / 2)

        # Targets
        self.target_starting_y = self.world_y_dim - (self.free_y_dim / 2)
        self.target_chains = [None for _ in range(batch_dim)]

        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        # Check if we are training or evaluating
        self.training = kwargs.pop("training", True)

        # Reward Shaping
        self.frechet_shaping_factor = 1.0
        self.centroid_shaping_factor = 1.0
        self.passage_entrance_factor = 1.0
        self.passage_exit_factor = 1.0
        # self.curvature_shaping_factor = 1.0
        self.prev_dist_factor = 1.0
        self.goal_bonus = 5.0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device
        # Make world
        world = World(
            batch_dim=batch_dim,
            x_semidim=self.world_x_dim,
            y_semidim=self.world_y_dim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            contact_margin=1e-3,
            torque_constraint_force=0.1,
        )



        self.targets = []
        target = Landmark(
            name=f"target_rad",
            shape=Sphere(self.goal_radius),
            color=COLOR_MAP["GREEN"],
            collide=False,
        )
        world.add_landmark(target)
        self.targets.append(target)
        # Add agents
        # entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith(
        #     "target"
        # )
        entity_filter_collisions: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "collision"
        )
        self.agents = []
        for n_agent in range(self.n_agents):
            agent = Agent(
                name=f"agent_{n_agent}",
                render_action=True,
                shape=Box(length=self.agent_radius * 2, width=self.agent_radius * 2.5),
                dynamics=SalpDynamics(),
                color=COLOR_LIST[n_agent],
                u_multiplier=self.u_multiplier,
                sensors=[
                    # Lidar(
                    #     world,
                    #     n_rays=self.lidar_rays,
                    #     max_range=self.lidar_range,
                    #     # entity_filter=entity_filter_targets,
                    #     angle_start=-0.5 * torch.pi,
                    #     angle_end=0.5 * torch.pi,
                    #     alpha=0.1,
                    # ),
                    Lidar(
                        world,
                        n_rays=self.lidar_rays,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_collisions,
                        angle_start=-0.5 * torch.pi,
                        angle_end=0.5 * torch.pi,
                        alpha=0.1,
                    ),
                ],
            )
            world.add_agent(agent)
            self.agents.append(agent)

        # Add joints
        self.joints = []
        for i in range(self.n_agents - 1):
            joint = Joint(
                world.agents[i],
                world.agents[i + 1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.agent_joint_length,
                rotate_a=self.rotating_salps,
                rotate_b=self.rotating_salps,
                collidable=False,
                width=0,
            )
            world.add_joint(joint)
            self.joints.append(joint)

        # Add collision landmarks
        self.collision_landmarks = []
        for i in range(self.n_collision_landmarks):
            collision = Landmark(
                name=f"collision_landmark_{i}",
                shape=Sphere(self.collision_radius),
                color=COLOR_MAP["RED"],
                collide=True,
            )
            world.add_landmark(collision)
            self.collision_landmarks.append(collision)

        # Initialize reward tensors
        self.global_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.centroid_rew = self.global_rew.clone()
        self.frechet_rew = self.global_rew.clone()
        self.distance_rew = self.global_rew.clone()

        # Initialize memory
        self.internal_angles_prev = torch.zeros(
            (batch_dim, self.n_agents - 2), device=device, dtype=torch.float32
        )  # n_agents-2 internal angles
        self.link_angles_prev = torch.zeros(
            (batch_dim, self.n_agents - 2), device=device, dtype=torch.float32
        )  # n_agents-1 link angles
        self.relative_angles_prev = torch.zeros(
            (batch_dim, self.n_agents, 2), device=device, dtype=torch.float32
        )  # n_agents-1 link angles

        world.zero_grad()

        # Step counter
        self.max_steps = 512
        self.steps = torch.zeros((batch_dim), device=device, dtype=torch.float32)

        return world

    def reset_world_at(self, env_index: int = None):

        print("RESET WORLD")
        # Rotation params
        agent_rotation_angles = [
            random.uniform(0, 2 * math.pi) for _ in range(self.world.batch_dim)
        ]
        agent_rotation_tensor = torch.tensor(
            agent_rotation_angles, device=self.device
        ).unsqueeze(-1)
        target_rotation_angle = random.uniform(0, 2 * math.pi)

        if env_index is None:
            self.steps = torch.zeros(
                (self.world.batch_dim), device=self.device, dtype=torch.float32
            )

            self.agent_chains = [
                self.create_agent_chain(
                    theta_min=0.0,
                    theta_max=0.0,
                    rotation_angle=agent_rotation_tensor[i],
                )
                for i in range(self.world.batch_dim)
            ]

            self.target_chains = [
                self.create_target_chain(
                    rotation_angle=target_rotation_angle,
                )
                for _ in range(self.world.batch_dim)
            ]

            agent_chain_tensor = torch.stack(self.agent_chains)
            for i, agent in enumerate(self.agents):
                pos = agent_chain_tensor[:, i, :]
                agent.set_pos(pos, batch_index=None)
                agent.set_rot(agent_rotation_tensor, batch_index=None)

            target_chain_tensor = torch.stack(self.target_chains)
            for i, target in enumerate(self.targets):
                pos = target_chain_tensor[:, i, :]
                target.set_pos(pos, batch_index=None)

            for i, joint in enumerate(self.joints):
                half_distance = (
                    self.agents[i].state.pos - self.agents[i + 1].state.pos
                ) / 2
                joint.landmark.set_pos(
                    self.agents[i].state.pos + half_distance, batch_index=None
                )

            self._reset_collision_landmarks(batch_index=None)

            a_pos = self.get_agent_chain_position()
            self.internal_angles_prev, self.link_angles_prev = internal_angles_xy(a_pos)
            relative_angles = [
                get_neighbor_angles(a_pos, self.world.agents.index(a), self.n_agents)
                for a in self.world.agents
            ]
            self.relative_angles_prev = torch.stack(relative_angles).transpose(1, 0)

            t_pos = self.get_target_chain_position()
            f_dist, _ = calculate_frechet_reward(a_pos, t_pos)
            c_dist, _ = calculate_centroid_reward(a_pos.mean(dim=1), t_pos.mean(dim=1))
            chain_centroid = a_pos.mean(dim=1)
            target_center = t_pos.mean(dim=1)
            centroid_dist = torch.norm(chain_centroid - target_center, dim=-1)
            self.prev_dist = centroid_dist * self.prev_dist_factor

        else:
            self.steps[env_index] = 0
            self.agent_chains[env_index] = self.create_agent_chain(
                theta_min=0.0,
                theta_max=0.0,
                rotation_angle=agent_rotation_tensor[env_index],
            )
            self.target_chains[env_index] = self.create_target_chain(
                rotation_angle=target_rotation_angle,
            )

            for n_agent, agent in enumerate(self.world.agents):
                pos = self.agent_chains[env_index][n_agent]
                agent.set_pos(pos, batch_index=env_index)
                agent.set_rot(agent_rotation_tensor[env_index], batch_index=env_index)

            for n_target, target in enumerate(self.targets):
                pos = self.target_chains[env_index][n_target]
                target.set_pos(pos, batch_index=env_index)

            for i, joint in enumerate(self.joints):
                half_distance = (
                    self.agents[i].state.pos - self.agents[i + 1].state.pos
                ) / 2
                joint.landmark.set_pos(
                    self.agents[i].state.pos[env_index] + half_distance[env_index],
                    batch_index=env_index,
                )

            self._reset_collision_landmarks(batch_index=env_index)

            a_pos = self.get_agent_chain_position()
            self.internal_angles_prev[env_index], self.link_angles_prev[env_index] = (
                internal_angles_xy(a_pos[env_index].unsqueeze(0))
            )
            relative_angles = [
                get_neighbor_angles(a_pos, self.world.agents.index(a), self.n_agents)
                for a in self.world.agents
            ]
            self.relative_angles_prev[env_index] = torch.stack(
                relative_angles
            ).transpose(1, 0)[env_index]

            t_pos = self.get_target_chain_position()
            f_dist, _ = calculate_frechet_reward(a_pos, t_pos)
            c_dist, _ = calculate_centroid_reward(a_pos.mean(dim=1), t_pos.mean(dim=1))
            self.prev_dist[env_index] = (
                torch.norm(a_pos.mean(dim=1)[env_index] - t_pos.mean(dim=1)[env_index], dim=-1)
                * self.prev_dist_factor
            )

    def _reset_collision_landmarks(self, batch_index=None):
        if batch_index is None:
            for landmark in self.collision_landmarks:
                positions = []
                for _ in range(self.world.batch_dim):
                    x_coord, y_coord = generate_random_coordinate_coordinate_inside_box(
                        0.0,
                        0.0,
                        self.world.x_semidim - self.collision_radius,
                        self.world.y_semidim - self.collision_radius,
                    )
                    positions.append([x_coord, y_coord])
                positions = torch.tensor(
                    positions, dtype=torch.float32, device=self.device
                )
                landmark.set_pos(positions, batch_index=None)
        else:
            for landmark in self.collision_landmarks:
                x_coord, y_coord = generate_random_coordinate_coordinate_inside_box(
                    0.0,
                    0.0,
                    self.world.x_semidim - self.collision_radius,
                    self.world.y_semidim - self.collision_radius,
                )
                pos = torch.tensor([x_coord, y_coord], dtype=torch.float32, device=self.device)
                landmark.set_pos(pos, batch_index=batch_index)

    def is_out_of_bounds(self, x_coord, y_coord):
        """Boolean mask of shape (n_envs,) – True if agent is out of bounds."""
        out_of_bounds = []

        for agent in self.agents:
            pos = agent.state.pos  # (n_envs, 2)
            x_ok = pos[..., 0].abs() <= x_coord - 1e-4
            y_ok = pos[..., 1].abs() <= y_coord - 1e-4
            out_of_bounds.append(~(x_ok & y_ok))

        out_of_bounds = torch.stack(out_of_bounds).transpose(1, 0).any(dim=-1)

        return out_of_bounds

    def _safe_spawn_boundary(self, offset: float, semidim: float) -> float:
        """Return a symmetric spawn half-width that keeps points inside the world."""
        return min(semidim - self.max_spawn_radius, semidim - abs(offset))

    def create_agent_chain(self, theta_min, theta_max, rotation_angle: float = 0.0):
        y_boundary = self._safe_spawn_boundary(self.agent_starting_y-.2, self.world.y_semidim)
        x_boundary = self.world.x_semidim - self.max_spawn_radius

        x_coord, y_coord = generate_random_coordinate_coordinate_inside_box(
            0.0,
            self.agent_starting_y,
            x_boundary,
            y_boundary,
        )

        chain = rotate_points(
            points=generate_target_points(
                x=x_coord,
                y=y_coord,
                n_points=self.n_agents,
                d_max=self.agent_joint_length,
                theta_range=[
                    theta_min,
                    theta_max,
                ],
            ),
            angle_rad=rotation_angle,
        ).to(self.device)

        
        return chain
    

    def create_target_chain(self, rotation_angle: float = 0.0):
        if hasattr(self, "fixed_target_chain"):
            return self.fixed_target_chain.clone()

        y_boundary = self._safe_spawn_boundary(self.target_starting_y, self.world.y_semidim)
        x_boundary = self.world.x_semidim - self.max_spawn_radius

        x_coord, y_coord = generate_random_coordinate_coordinate_inside_box(
            0.0,
            self.target_starting_y,
            x_boundary,
            y_boundary,
        )

        n_bends = random.choice([0, 1])
        radius = random.uniform(0.05, 0.3)
        radius_scaling = (
            self.n_agents // 3
        )  # 3 because it's the minimum amount of points for a curve

        chain = rotate_points(
            points=generate_bending_curve(
                x0=x_coord,
                y0=y_coord,
                n_points=self.n_agents,
                max_dist=self.agent_joint_length,
                radius=radius * radius_scaling,
                n_bends=n_bends,
            ),
            angle_rad=rotation_angle,
        ).to(self.device)

        # self.fixed_target_chain = chain.clone()

        # return self.fixed_target_chain
        return chain

    def interpolate(
        self,
        value,
        source_min=-1,
        source_max=1,
        target_min=-torch.pi,
        target_max=torch.pi,
    ):
        # Linear interpolation using PyTorch
        return target_min + (value - source_min) / (source_max - source_min) * (
            target_max - target_min
        )

    def process_action(self, agent: Agent):

        if self.rotating_salps:
            magnitude = agent.action.u[:, 0]

            # Set salp's rotation
            agent.state.rot += agent.action.u[:, 1].unsqueeze(-1)

        else:
            magnitude_pos = self.interpolate(
                agent.action.u[:, 0], target_min=0, target_max=1
            )

            magnitude_neg = self.interpolate(
                agent.action.u[:, 1], target_min=0, target_max=1
            )

            magnitude = magnitude_pos - magnitude_neg

        # Get heading vector
        agent_rot = agent.state.rot % (2 * torch.pi)
        heading_offset = agent_rot + torch.pi / 2

        theta = heading_offset % (2 * torch.pi)

        # Set salp's force vector
        x = torch.cos(theta).squeeze(-1) * magnitude
        y = torch.sin(theta).squeeze(-1) * magnitude

        agent.state.force = torch.stack((x, y), dim=-1)

    def get_targets(self):
        return [
            landmark
            for landmark in self.world.landmarks
            if landmark.name.startswith("target")
        ]

    def get_passages(self):
        return [
            landmark
            for landmark in self.world.landmarks
            if landmark.name.startswith("passage")
        ]

    def get_passages_positions(self):
        passages = self.get_passages()
        passage_pos = [p.state.pos for p in passages]

        return torch.stack(passage_pos).transpose(1, 0).float()

    def get_agent_chain_position(self):
        agent_pos = [a.state.pos for a in self.world.agents]
        return torch.stack(agent_pos).transpose(1, 0).float()

    def get_target_chain_position(self):
        targets = self.get_targets()
        target_pos = [t.state.pos for t in targets]
        return torch.stack(target_pos).transpose(1, 0).float()

    def check_collisions(self):
        collision_tensor = torch.zeros(
            self.world.batch_dim, device=self.device, dtype=torch.int
        )

        for agent in self.world.agents:
            overlap = torch.zeros(
                self.world.batch_dim, device=self.device, dtype=torch.int
            )
            for collision_landmark in self.collision_landmarks:
                overlap |= (
                    self.world.is_overlapping(agent, collision_landmark) > 0.0
                ).int()
            collision_tensor |= overlap

        return collision_tensor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            
           

            # Calculate rewards
            self.frechet_rew[:] = 0
            self.centroid_rew[:] = 0
            self.distance_rew[:] = 0

            # Get chain positions
            agent_pos = self.get_agent_chain_position()
            target_pos = self.get_target_chain_position()
            chain_centroid = agent_pos.mean(dim=1)
            target_center = target_pos[:,0,:]

            goal_dist = torch.norm(chain_centroid - target_center, dim=-1)  
            inside_goal = goal_dist < self.goal_radius
            print(f"goal_dist: {goal_dist}")
            # dist_rew = calculate_distance_reward(chain_centroid, target_center)
            current_dist = goal_dist * self.prev_dist_factor

            # Distance reward
            # dist_rew = calculate_distance_reward(agent_pos, target_pos)
            # raw_distance_reward = -1 * dist_rew
            # print(f"raw distance reward: {raw_distance_reward}")
            # current_dist = dist_rew * self.prev_dist_factor
            # print(f"dr {current_dist}")
            print(f"prev_dist: {self.prev_dist}")
            self.distance_rew = self.prev_dist - current_dist
            print(f"raw distance reward: {self.distance_rew}")
            self.distance_rew = torch.where(self.distance_rew < 0.001, self.distance_rew * 50, self.distance_rew )
            print(f"distance reward: {self.distance_rew}")
            # self.distance_rew = torch.exp(current_dist) 
            # print(f"distance reward {self.distance_rew}")
            self.prev_dist = current_dist
            # print(f"distance shaping {self.prev_dist}")
            

            # Passage entrance reward
            chain_centroid = agent_pos.mean(dim=1)
            centroid_y = chain_centroid[:, 1]

            
            
            _, f_rew = calculate_frechet_reward(agent_pos, target_pos)

            # Get reward for reaching the goal
            self.total_rew = f_rew
            goal_reached_rew = torch.zeros(
                self.world.batch_dim, device=self.device, dtype=torch.float32
            )
            # goal_reached_mask = self.total_rew > self.frechet_thresh
            # goal_reached_rew += self.reached_goal_bonus * goal_reached_mask.int()

            goal_reached_mask = inside_goal
            goal_reached_rew += self.goal_bonus * goal_reached_mask.int()
            print(f"goal reached reward: {goal_reached_rew}")

            

            # Check for collisions
            has_collided = self.check_collisions()
            collision_penalty = torch.zeros(
                self.world.batch_dim, device=self.device, dtype=torch.float32
            )
            collision_penalty += self.collision_penalty * has_collided
            print(f"Collision penalty: {collision_penalty}")
            # print(f"dist_rew: {self.distance_rew}")
      
            print(f"goal_reached_rew: {goal_reached_rew}")

           
            print("collision tensor:", has_collided[:5])
            print("any overlap raw:",
            [(self.world.is_overlapping(a, p).max().item())
            for a in self.world.agents[:2]
            for p in self.get_passages()[:2]])

            # Mix all rewards
            self.global_rew = (
                (5.0 * self.distance_rew)
                + collision_penalty
                + goal_reached_rew
              
            )

            print(f"Total reward: {self.global_rew}")
            
        return self.global_rew

    def get_heading(self, agent: Agent):
        x = torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        y = torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)

        return torch.stack((x, y), dim=-1)

    def agent_representation(self, agent: Agent, scope: str):

        # Agent specific
        a_pos_rel_2_t_centroid = (
            agent.state.pos - self.global_state.t_chain_centroid_pos
        )

        a_vel_rel_2_centroid = agent.state.vel - self.global_state.a_chain_centroid_vel

        a_pos_rel_2_centroid = agent.state.pos - self.global_state.a_chain_centroid_pos

        # Get agent information
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        idx = self.world.agents.index(agent)

        # Encode agent id
        encoding_len = 6
        encoded_idx = torch.zeros(
            (self.world.batch_dim, encoding_len),
            dtype=torch.float32,
            device=self.device,
        ) + binary_encode(idx, encoding_len)

        # Get neighbor forces
        neighbor_forces = torch.zeros(
            (self.world.batch_dim, 4), dtype=torch.float32, device=self.device
        )
        if is_first:
            neighbor_forces[:, 2:] = self.global_state.a_chain_all_forces[:, 1, :]
        elif is_last:
            neighbor_forces[:, :2] = self.global_state.a_chain_all_forces[:, -2, :]
        else:
            neighbor_forces = self.global_state.a_chain_all_forces[
                :, idx - 1 : idx + 2 : 2, :
            ].flatten(start_dim=1)

        # Get distance to assigned position
        # a_pos_2_t_pos_err = (
        #     self.global_state.t_chain_all_pos[:, idx, :]
        #     - self.global_state.a_chain_all_pos[:, idx, :]
        # )

        goal_center = self.global_state.t_chain_all_pos[:, 0, :]
        a_pos_2_t_pos_err = (
            goal_center
            - self.global_state.a_chain_all_pos[:, idx, :]
)

        a_pos_2_passage_pos_err = torch.zeros_like(
            self.global_state.a_chain_all_pos[:, idx, :]
        )
        a_pos_2_pen_pos_err = torch.zeros_like(
            self.global_state.a_chain_all_pos[:, idx, :]
        )
        a_pos_2_pex_pos_err = torch.zeros_like(
            self.global_state.a_chain_all_pos[:, idx, :]
        )

        # observation = torch.cat(
        #     [
        #         # Agent id
        #         encoded_idx,
        #         # Neighbor data
        #         torch.sin(self.global_state.a_chain_relative_angles[:, idx, :]),
        #         torch.cos(self.global_state.a_chain_relative_angles[:, idx, :]),
        #         self.global_state.a_chain_relative_angles_speed[:, idx, :],
        #         neighbor_forces,
        #         # Local data
        #         a_pos_rel_2_centroid,
        #         agent.state.pos,
        #         agent.state.vel,
        #         wrap_to_pi(agent.state.rot),
        #         agent.state.ang_vel,
        #         # Target data
        #         a_pos_rel_2_t_centroid,
        #         a_vel_rel_2_centroid,
        #         a_pos_2_t_pos_err,
        #         # Passage data,
        #         a_pos_2_pen_pos_err,
        #         a_pos_2_pex_pos_err,
        #         a_pos_2_passage_pos_err,
        #         # Lidar data,
        #         agent.sensors[0].measure(),
        #     ],
        #     dim=-1,
        # ).float()

        lidar_collision = agent.sensors[0].measure()
        # lidar_collision = agent.sensors[1].measure()
        # lidar = torch.cat([lidar_target, lidar_collision], dim=-1)

        if agent == self.world.agents[0] and int(self.steps[0].item()) % 5 == 0:
            print("lidar env0:", lidar_collision[0].detach().cpu().numpy())

        observation = torch.cat(
            [
                # Local data
                a_pos_rel_2_centroid,
                agent.state.pos,
                wrap_to_pi(agent.state.rot),
                # Target data
                a_pos_rel_2_t_centroid,
                a_pos_2_t_pos_err,
                # Lidar data,
                lidar_collision,
            ],
            dim=-1,
        ).float()

        return observation

    def observation(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # Calculate global state
            open_passages = torch.zeros(
                (self.world.batch_dim, 2), device=self.device, dtype=torch.float32
            )

            agent_pos = self.get_agent_chain_position()
            target_pos = self.get_target_chain_position()
            a_chain_centroid_pos = agent_pos.mean(dim=1)
            t_chain_centroid_pos = target_pos.mean(dim=1)

            aligned_agent_pos = agent_pos - agent_pos.mean(dim=1, keepdim=True)
            aligned_target_pos = target_pos - target_pos.mean(dim=1, keepdim=True)

            total_moment = 0

            vels = []
            ang_vels = []
            ang_pos = []
            forces = []
            relative_angles = []

            for a in self.world.agents:

                r = a.state.pos - a_chain_centroid_pos
                total_moment += calculate_moment(r, a.state.force)

                vels.append(a.state.vel)
                ang_vels.append(a.state.ang_vel)
                ang_pos.append(a.state.rot)
                forces.append(a.state.force)
                relative_angles.append(
                    get_neighbor_angles(
                        agent_pos, self.world.agents.index(a), self.n_agents
                    )
                )

            vels = torch.stack(vels).transpose(1, 0)
            ang_vels = torch.stack(ang_vels).transpose(1, 0)
            ang_pos = torch.stack(ang_pos).transpose(1, 0)
            forces = torch.stack(forces).transpose(1, 0)
            relative_angles = torch.stack(relative_angles).transpose(1, 0)

            internal_angles, link_angles = internal_angles_xy(agent_pos)

            # Calculate angle derivatives
            internal_angles_speed = (
                wrap_to_pi(internal_angles - self.internal_angles_prev) / self.world.dt
            )

            link_angles_speed = (
                wrap_to_pi(link_angles - self.link_angles_prev) / self.world.dt
            )

            relative_angles_speed = (
                wrap_to_pi(relative_angles - self.relative_angles_prev) / self.world.dt
            )

            # Store previous dtheta
            self.internal_angles_prev = internal_angles.clone()
            self.link_angles_prev = link_angles.clone()
            self.relative_angles_prev = relative_angles.clone()

            # Build global observation
            self.global_state = GlobalObservation(
                open_passages,
                # Menger curvature
                # menger_curvature(agent_pos, self.agent_joint_length)
                # - menger_curvature(target_pos, self.agent_joint_length),
                # Internal angle data
                internal_angles,
                internal_angles_speed,
                # Link angles
                link_angles,
                link_angles_speed,
                # Relative angles
                relative_angles,
                relative_angles_speed,
                # Raw obs
                target_pos,
                agent_pos,
                vels.flatten(start_dim=1),
                ang_pos.flatten(start_dim=1),
                ang_vels.flatten(start_dim=1),
                forces,
                # Condensed obs
                t_chain_centroid_pos,
                a_chain_centroid_pos,
                vels.mean(dim=1),
                wrap_to_pi(ang_pos.mean(dim=1)),
                ang_vels.mean(dim=1),
                forces.sum(dim=1),
                total_moment.unsqueeze(-1),
                batch_discrete_frechet_distance(
                    aligned_agent_pos, aligned_target_pos
                ).unsqueeze(-1),
            )

        return self.agent_representation(agent, self.state_representation)

    def done(self):
        # Update step count
        self.steps += 1

        # Check termination conditions
        # target_reached = self.total_rew > self.frechet_thresh

        agent_pos = self.get_agent_chain_position()
        target_pos = self.get_target_chain_position()

        chain_centroid = agent_pos.mean(dim=1)
        target_center = target_pos[:, 0, :]

        goal_dist = torch.norm(chain_centroid - target_center, dim=-1)
        target_reached = goal_dist <= self.goal_radius

        out_of_bounds = self.is_out_of_bounds(
            self.world.x_semidim, self.world.y_semidim
        )
        # has_collided = self.check_collisions()
        timeout = self.steps >= self.max_steps

        return target_reached | out_of_bounds  | timeout

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        chain_pos = self.get_agent_chain_position()
        target_pos = self.get_target_chain_position()
        return {
            "target_pose": (target_pos),
            "chain_pose": (chain_pos),
            
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        targets = self.get_targets()
        for i, target in enumerate(targets):
            range_circle = rendering.make_circle(self.target_radius, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*COLOR_LIST[i])
            geoms.append(range_circle)

        a_pos = self.get_agent_chain_position()

        range_circle = rendering.make_circle(self.target_radius, filled=False)
        xform = rendering.Transform()
        xform.set_translation(*a_pos[env_index].mean(dim=0))
        range_circle.add_attr(xform)
        range_circle.set_color(*COLOR_MAP["BLACK"].value)
        geoms.append(range_circle)

        t_pos = self.get_target_chain_position()

        range_circle = rendering.make_circle(self.target_radius, filled=False)
        xform = rendering.Transform()
        xform.set_translation(*t_pos[env_index].mean(dim=0))
        range_circle.add_attr(xform)
        range_circle.set_color(*COLOR_MAP["BLACK"].value)
        geoms.append(range_circle)

        return geoms
