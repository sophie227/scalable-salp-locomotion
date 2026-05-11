


# import matplotlib.pyplot as plt
# from collections import defaultdict
# from pathlib import Path
# import numpy as np
# import pickle
# import yaml
# import pandas as pd

# plt.rcParams.update({'font.size': 16})

# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# plotting_dir = Path().resolve()
# config_dir = plotting_dir / "ppo_config.yaml"
# print(config_dir)


# with open(config_dir, "r") as file:
#     config = yaml.safe_load(file)
#     print(config)
    

# fig = plt.figure(figsize=(12, 7))
# ax = fig.add_subplot(111)

# # Store data by experiment group
# experiment_data = defaultdict(list)
# experiment_data1 = defaultdict(list)
# experiment_data2 = defaultdict(list)
# experiment_data3 = defaultdict(list)        

# max_len = 0

# # First pass: collect data for each experiment
# for batch in config["batches"]:
#     for experiment in config["experiments"]:
#         exp_key = f"{batch}-{experiment}"
        
#         for trial in config["trials"]:
#             checkpoint_path = Path(f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/train.dat")
#             curriculum_checkpoint_path_1 = Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_1/logs/train.dat")
#             curriculum_checkpoint_path_2 = Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_2/logs/train.dat")
#             curriculum_checkpoint_path_3 = Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_3/logs/train.dat")
#             # print(checkpoint_path)
            

#             if checkpoint_path.is_file():
#                 with open(checkpoint_path, "rb") as handle:
#                     data = pickle.load(handle)
#                     # all_keys = data.keys()

#                     # for key in all_keys:
#                     #     print(key)
#                     full_data = data["rewards_per_iteration"]
#                     print(f"Trial {trial}: {len(full_data)} total data points")
#                     data = full_data
#                     print(f"Trial {trial}: using {len(data)} data points after slicing")
                    
#                     # Apply moving average
#                     if len(data) > config["moving_avg_window_size"]:
#                         smoothed_data = moving_average(data, config["moving_avg_window_size"])
#                         experiment_data[exp_key].append(smoothed_data)
#                         max_len = max(max_len, len(smoothed_data))
#                     else:
#                         print(f"Warning: Trial {trial} has too few data points for smoothing")
#             if curriculum_checkpoint_path_1.is_file():
#                 with open(curriculum_checkpoint_path_1, "rb") as handle:
#                     curr1 = pickle.load(handle)
#                     full_curr1 = curr1["rewards_per_iteration"]
#                     print(f"Trial {trial}_stage_1: {len(full_curr1)} total data points")
#                     data1 = full_curr1
#                     print(f"Trial {trial}_stage_1: using {len(data1)} data points after slicing")
                    
#                     # Apply moving average
#                     if len(data1) > config["moving_avg_window_size"]:
#                         smoothed_data = moving_average(data1, config["moving_avg_window_size"])
#                         experiment_data1[exp_key].append(smoothed_data)
#                         max_len = max(max_len, len(smoothed_data))
#                     else:
#                         print(f"Warning: Trial {trial}_stage_1 has too few data points for smoothing")
#             if curriculum_checkpoint_path_2.is_file():
#                 with open(curriculum_checkpoint_path_2, "rb") as handle:
#                     curr2 = pickle.load(handle)
#                     full_curr2 = curr2["rewards_per_iteration"]
#                     print(f"Trial {trial}_stage_2: {len(full_curr2)} total data points")
#                     data2 = full_curr2
#                     print(f"Trial {trial}_stage_2: using {len(data2)} data points after slicing")
                    
#                     # Apply moving average
#                     if len(data2) > config["moving_avg_window_size"]:
#                         smoothed_data = moving_average(data2, config["moving_avg_window_size"])
#                         experiment_data2[exp_key].append(smoothed_data)
#                         max_len = max(max_len, len(smoothed_data))
#                     else:
#                         print(f"Warning: Trial {trial}_stage_2 has too few data points for smoothing")
#             if curriculum_checkpoint_path_3.is_file():  
#                 with open(curriculum_checkpoint_path_3, "rb") as handle:
#                     curr3 = pickle.load(handle)
#                     full_curr3 = curr3["rewards_per_iteration"]
#                     print(f"Trial {trial}_stage_3: {len(full_curr3)} total data points")
#                     data3 = full_curr3
#                     print(f"Trial {trial}_stage_3: using {len(data3)} data points after slicing")
                    
#                     # Apply moving average
#                     if len(data3) > config["moving_avg_window_size"]:
#                         smoothed_data = moving_average(data3, config["moving_avg_window_size"])
#                         experiment_data3[exp_key].append(smoothed_data)
#                         max_len = max(max_len, len(smoothed_data))
#                     else:
#                         print(f"Warning: Trial {trial}_stage_3 has too few data points for smoothing")
#             combined_df = pd.DataFrame({experiment_data,experiment_data1, experiment_data2, experiment_data3})
                
# # Second pass: pad arrays to the same length if needed
# for exp_key in combined_df.columns:
#     padded_data = []
#     for trial_data in experiment_data[exp_key]:
#         if len(trial_data) < max_len:
#             # Pad with last value or NaN
#             padding = np.full(max_len - len(trial_data), np.nan)
#             padded_data.append(np.concatenate([trial_data, padding]))
#         else:
#             padded_data.append(trial_data)
    
#     experiment_data[exp_key] = padded_data

# # Plotting with standard error
# color_idx = 0
# for exp_key in experiment_data:
#     if not experiment_data[exp_key]:
#         print(f"No data for {exp_key}")
#         continue
        
#     # Convert list of arrays to 2D numpy array
#     data_array = np.array(experiment_data[exp_key])
    
#     # Calculate mean and standard error across trials
#     mean_rewards = np.nanmean(data_array, axis=0)
    
#     # Standard Error = StdDev / sqrt(n)
#     n_trials = np.sum(~np.isnan(data_array), axis=0)  # Count non-NaN values at each step
#     std_rewards = np.nanstd(data_array, axis=0)
#     se_rewards = std_rewards / np.sqrt(np.maximum(n_trials, 1))  # Avoid division by zero
    
#     # X axis
#     x = np.arange(len(mean_rewards))
    
#     # Get color for this experiment
#     color = plt.cm.tab10(color_idx % 10)
#     color_idx += 1
    
#     # Plot mean line
#     ax.plot(x, mean_rewards, linewidth=1, label=exp_key, color=color)
    
#     # Plot standard error band
#     ax.fill_between(
#         x,
#         mean_rewards - se_rewards,
#         mean_rewards + se_rewards,
#         alpha=0.3,
#         color=color
#     )
    
#     print(f"{exp_key}: {len(experiment_data[exp_key])} trials")

# ax.legend(loc='best')
# ax.set_xlabel("Iterations")
# ax.set_ylabel("Average Reward")
# ax.set_title("Learning Curves with Standard Error Across Trials")
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


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

            paths = [
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_1/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_2/logs/train.dat"),
                Path(f"{config['base_path']}/{batch}/{experiment}/{trial}_stage_3/logs/train.dat"),
            ]

            for path in paths:

                if path.is_file():

                    with open(path, "rb") as handle:
                        data = pickle.load(handle)

                    rewards = data["rewards_per_iteration"]

                    print(f"{path}")
                    print(f"Loaded {len(rewards)} points")

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
ax.set_ylabel("Average Reward")
ax.set_title("Learning Curves with Standard Error")

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()