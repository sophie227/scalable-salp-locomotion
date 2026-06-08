import matplotlib.pyplot as plt
import dill

# file_path  = "/home/sophie/scalable-salp-locomotion/src/experiments/results/salp_navigate_varying_salp_curr/gcn_dim/0/logs/crumple_data_new.dat"
file_path2 = "/home/sophie/scalable-salp-locomotion/src/experiments/results/salp_navigate_varying_baseline/gcn_dim/6/logs/crumple_data_1025_1.dat"
file_path = "/home/sophie/scalable-salp-locomotion/src/experiments/results/salp_navigate_varying_salp_curr/gcn_dim/6/logs/crumple_data_1025.dat"

with open(file_path, "rb") as f:
    data = dill.load(f)

with open(file_path2, "rb") as f:
    data2 = dill.load(f)

n_agents = list(data.keys())[1]
mask = list(data[n_agents].keys())[0]
print(f"n_agents: {n_agents}, mask: {mask}")



chain_lengths = data[n_agents][mask]["chain_lengths"]
crumple_scores = data[n_agents][mask]["crumple_scores"]

chain_lengths2 = data2[n_agents][mask]["chain_lengths"]
crumple_scores2 = data2[n_agents][mask]["crumple_scores"]

plt.figure()
plt.plot(chain_lengths, marker='o', markersize=6, label="8 agent curriculum learning")
plt.plot(chain_lengths2, marker='d', color='red', markersize=6, label="8 agent baseline")
plt.title("Chain Length Over Time")
plt.xlabel("Timestep")
plt.ylabel("Length")
plt.legend()
plt.show()


plt.figure()
plt.plot(crumple_scores, marker='o', markersize=6, label="8 agent curriculum learning")
plt.plot(crumple_scores2, marker='d', color='red', markersize=6, label="8 agent baseline")
plt.title("Crumple Score Over Time")
plt.xlabel("Timestep")
plt.ylabel("Score")
plt.legend()
plt.show()