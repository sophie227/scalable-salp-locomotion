#!/usr/bin/env python3
"""
Script to generate and save 3 target chains to a pkl file.
"""
import torch
import pickle
from environments.salp_navigate.domain import SalpNavigateDomain
import pprint
import random

# Setup
device = torch.device("cpu")
batch_dim = 1

# Create domain instance
domain = SalpNavigateDomain()
world = domain.make_world(batch_dim=batch_dim, device=device)

# Parameters for creating target chains (from reset_world_at)
target_inner_radius = domain.train_world_x_dim * (0.15 * 3)
target_outer_radius = (
    target_inner_radius + domain.n_agents * domain.agent_joint_length
)

# Generate 3 target chains
target_chains = []
for i in range(3):
    chain = domain.create_target_chain(
        target_inner_radius,
        target_outer_radius,
        rotation_angle=0.0,
    )
#     target_chains.append(chain.cpu())
#     print(f"Generated target chain {i+1}: shape {chain.shape}")

# # Save to pkl file
output_file = "target_chains.pkl"
# with open(output_file, "wb") as f:
#     pickle.dump(target_chains, f)

# print(f"\nSuccessfully saved 3 target chains to {output_file}")


with open(output_file, "rb") as f:
    loaded_chains = pickle.load(f)
pprint.pprint(loaded_chains)

chain_dict = {f"chain_{i}": chain for i, chain in enumerate(loaded_chains)}

value = random.choice(list(chain_dict.keys()))
print(value)

chain = chain_dict[value]
print(f"Selected chain: {value}, chain: {chain}")

# print(chain_dict)