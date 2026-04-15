#!/usr/bin/env python3
"""
Script to generate and save 3 target chains to a pkl file.
"""
import torch
import pickle
from environments.salp_navigate.domain import SalpNavigateDomain
from environments.salp_passage_curr.domain import SalpPassageDomain
import pprint
import random


def shift_chain_to_negative_y(chain: torch.Tensor, margin: float = 1e-1) -> torch.Tensor:
    max_y = chain[:, 1].max()
    if max_y < -0.2:
        return chain

    shifted_chain = chain.clone()
    shifted_chain[:, 1] -= max_y + random.uniform(.1, .5) 
    shifted_x = shifted_chain.clone()
    shifted_x[:, 0] -= random.uniform(-0.5, 0.5)  # Shift left by random margin

    return shifted_chain

# Setup
device = torch.device("cpu")
batch_dim = 1
n_agents = 8  # Change this to desired number of agents

# Create domain instance
domain = SalpNavigateDomain()
world = domain.make_world(batch_dim=batch_dim, device=device, n_agents=n_agents)

# Parameters for creating target chains (from reset_world_at)
target_inner_radius = 0.2
target_outer_radius = (
domain.world_y_dim 
)

# Generate 5 unique target chains
target_chains = []
while len(target_chains) < 5:
    chain = domain.create_target_chain(
        rotation_angle=0.0,
        inner_r=target_inner_radius,
        outer_r=target_outer_radius,
    )
    chain = shift_chain_to_negative_y(chain)
    # Skip if this chain is identical to any already collected chain
    if any(torch.allclose(chain, existing) for existing in target_chains):
        max_x = chain[:, 0].max()
        shifted_x_chain = chain.clone()
        shifted_x_chain[:, 0] -= max_x + random.uniform(-.5, 0.2)  # Shift left by max_x + random margin
        
        
    target_chains.append(chain.cpu())
    print(
        f"Generated target chain {len(target_chains)}: shape {chain.shape}, "
        f"y-range=({chain[:, 1].min().item():.3f}, {chain[:, 1].max().item():.3f})"
    )

# Save to pkl file
output_file = "target_chains_3.pkl"
with open(output_file, "wb") as f:
    pickle.dump(target_chains, f)

print(f"\nSuccessfully saved {len(target_chains)} target chains to {output_file}")


with open(output_file, "rb") as f:
    loaded_chains = pickle.load(f)
pprint.pprint(loaded_chains)

chain_dict = {f"chain_{i}": chain for i, chain in enumerate(loaded_chains)}

value = random.choice(list(chain_dict.keys()))
print(value)

chain = chain_dict[value]
print(f"Selected chain: {value}, chain: {chain}")

# print(chain_dict)