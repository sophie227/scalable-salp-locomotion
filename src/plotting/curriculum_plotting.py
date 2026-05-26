


import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle
import yaml

plt.rcParams.update({'font.size': 16})


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


plotting_dir = Path().resolve()
config_dir = plotting_dir / "ppo_config.yaml"

with open(config_dir, "r") as file:
    config = yaml.safe_load(file)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)

experiment_data = defaultdict(list)

max_len = 0

for batch in config["batches"]:
    for experiment in config["experiments"]:

        exp_key = f"{batch}-{experiment}"

        for trial in config["trials"]:

            all_stage_data = []
            stage_lengths = []

            paths = [
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_1/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_2/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_3/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_4/logs/train.dat")
            ]

            for stage_id, path in enumerate(paths):


                if path.is_file():

                    with open(path, "rb") as handle:
                        data = pickle.load(handle)

                    rewards = data["rewards_per_iteration"]

                    print(f"{path}")
                    print(f"Loaded {len(rewards)} points")

                    stage_lengths.append(len(rewards))
                    all_stage_data.extend(rewards)

            if len(all_stage_data) > config["moving_avg_window_size"]:

                smoothed = moving_average(
                    all_stage_data,
                    config["moving_avg_window_size"]
                )

                experiment_data[exp_key].append(smoothed)

                max_len = max(max_len, len(smoothed))

            else:
                print(f"Warning: not enough data for {trial}")

# Pad all trials to same length
for exp_key in experiment_data:

    padded = []

    for arr in experiment_data[exp_key]:

        if len(arr) < max_len:

            padding = np.full(max_len - len(arr), np.nan)

            arr = np.concatenate([arr, padding])

        padded.append(arr)

    experiment_data[exp_key] = np.array(padded)

# Plot
color_idx = 0

for exp_key, data_array in experiment_data.items():

    if len(data_array) == 0:
        continue

    mean_rewards = np.nanmean(data_array, axis=0)

    n_trials = np.sum(~np.isnan(data_array), axis=0)

    std_rewards = np.nanstd(data_array, axis=0)

    se_rewards = std_rewards / np.sqrt(np.maximum(n_trials, 1))

    x = np.arange(len(mean_rewards))

    color = plt.cm.tab10(color_idx % 10)

    color_idx += 1

    ax.plot(
        x,
        mean_rewards,
        linewidth=2,
        label=exp_key,
        color=color
    )

    ax.fill_between(
        x,
        mean_rewards - se_rewards,
        mean_rewards + se_rewards,
        alpha=0.3,
        color=color
    )

    print(f"{exp_key}: {len(data_array)} trials")

ax.legend(loc='best')

ax.set_xlabel("Iterations")
ax.set_ylabel("Average Global Reward")
ax.set_title("Learning Curves with Standard Error")

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# import matplotlib.pyplot as plt
# from collections import defaultdict
# from pathlib import Path
# import numpy as np
# import pickle
# import yaml

# plt.rcParams.update({'font.size': 16})


# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w


# plotting_dir = Path().resolve()
# config_dir = plotting_dir / "ppo_config.yaml"

# with open(config_dir, "r") as file:
#     config = yaml.safe_load(file)

# fig = plt.figure(figsize=(12, 7))
# ax = fig.add_subplot(111)

# experiment_data = defaultdict(list)

# max_len = 0


# for batch in config["batches"]:
#     for experiment in config["experiments"]:

#         exp_key = f"{batch}-{experiment}"

#         for trial in config["trials"]:

#             all_stage_data = []
#             stage_lengths = []

#             paths = [
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/train.dat"),
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_1/logs/train.dat"),
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_2/logs/train.dat"),
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_3/logs/train.dat"),
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_4/logs/train.dat"),
#                 Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_5/logs/train.dat")
#             ]

#             for path in paths:

#                 if path.is_file():

#                     with open(path, "rb") as handle:
#                         data = pickle.load(handle)

#                     rewards = data["rewards_per_iteration"]

#                     print(f"{path}")
#                     print(f"Loaded {len(rewards)} points")

#                     stage_lengths.append(len(rewards))
#                     all_stage_data.extend(rewards)

#             if len(all_stage_data) > config["moving_avg_window_size"]:

#                 smoothed = moving_average(
#                     all_stage_data,
#                     config["moving_avg_window_size"]
#                 )

#                 # compute stage boundaries
#                 stage_boundaries = np.cumsum(stage_lengths)
#                 stage_boundaries = stage_boundaries[stage_boundaries < len(smoothed)]

#                 experiment_data[exp_key].append((smoothed, stage_boundaries))

#                 max_len = max(max_len, len(smoothed))

#             else:
#                 print(f"Warning: not enough data for {trial}")


# # Pad all trials to same length
# for exp_key in experiment_data:

#     padded = []

#     for smoothed, boundaries in experiment_data[exp_key]:

#         if len(smoothed) < max_len:
#             padding = np.full(max_len - len(smoothed), np.nan)
#             smoothed = np.concatenate([smoothed, padding])

#         padded.append((smoothed, boundaries))

#     experiment_data[exp_key] = np.array(padded, dtype=object)


# # Plot
# for exp_key, data_array in experiment_data.items():

#     if len(data_array) == 0:
#         continue

#     all_curves = np.vstack([d[0] for d in data_array])

#     mean_rewards = np.nanmean(all_curves, axis=0)
#     std_rewards = np.nanstd(all_curves, axis=0)

#     n_trials = np.sum(~np.isnan(all_curves), axis=0)
#     se_rewards = std_rewards / np.sqrt(np.maximum(n_trials, 1))

#     x = np.arange(len(mean_rewards))

#     ax.plot(
#         x,
#         mean_rewards,
#         linewidth=2,
#         label=exp_key
#     )

#     ax.fill_between(
#         x,
#         mean_rewards - se_rewards,
#         mean_rewards + se_rewards,
#         alpha=0.3
#     )

#     print(f"{exp_key}: {len(data_array)} trials")

#     # ---- stage separators (from first trial) ----
#     _, boundaries = data_array[0]

#     for b in boundaries:
#         ax.axvline(
#             x=b,
#             color='black',
#             linestyle='--',
#             alpha=0.5,
#             linewidth=1
#         )


# ax.legend(loc='best')

# ax.set_xlabel("Iterations")
# ax.set_ylabel("Average Global Reward")
# ax.set_title("Learning Curves with Standard Error + Stage Boundaries")

# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()