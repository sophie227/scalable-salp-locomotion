import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle
import yaml

try:
    import dill
except ImportError:
    dill = None

plt.rcParams.update({'font.size': 16})


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def load_yaml(path: Path):
    if not path.is_file():
        return None
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_pickle_or_dill(path: Path):
    with open(path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception:
            if dill is not None:
                handle.seek(0)
                return dill.load(handle)
            raise


def infer_batch_env_config(batch_name: str):
    yaml_path = Path(__file__).resolve().parents[1] / 'experiments' / 'yamls' / batch_name / '_env.yaml'
    env_config = load_yaml(yaml_path)
    if not env_config:
        return None
    return {
        'n_agents': env_config.get('n_agents'),
        'state_representation': env_config.get('state_representation'),
    }


def extract_positions_from_state_array(array: np.ndarray, n_agents: int | None = None):
    array = np.asarray(array)

    if array.ndim == 3 and array.shape[-1] == 2:
        return array

    if array.ndim == 2 and array.shape[1] == 2 and n_agents is None:
        return array[:, None, :]

    if n_agents is not None:
        if array.ndim == 2 and array.shape[1] == n_agents * 2:
            return array.reshape(-1, n_agents, 2)

        if array.ndim == 3 and array.shape[1] == n_agents and array.shape[2] >= 4:
            return array[..., 2:4]

        if array.ndim == 2 and array.shape[1] == n_agents * 11:
            return array.reshape(-1, n_agents, 11)[..., 2:4]

    if array.ndim == 3 and array.shape[2] >= 4:
        return array[..., 2:4]

    raise ValueError(
        f"Unable to infer positions from state array with shape {array.shape}. "
        "Expected (timesteps, n_agents, 2) or a state tensor containing agent_pos at indices 2:4."
    )


# ============================================================
# CRUMPLE METRICS
# ============================================================

def compute_radius_of_gyration(positions):

    center = np.mean(positions, axis=0)

    rg = np.sqrt(
        np.mean(
            np.sum((positions - center) ** 2, axis=1)
        )
    )

    return rg


def compute_anisotropy(positions, eps=1e-8):

    center = np.mean(positions, axis=0)

    centered = positions - center

    cov = np.cov(centered.T)

    eigvals = np.linalg.eigvals(cov)
    eigvals = np.sort(np.real(eigvals))[::-1]

    return eigvals[0] / (eigvals[1] + eps)


# ============================================================
# CONFIG
# ============================================================

plotting_dir = Path().resolve()
config_dir = plotting_dir / "ppo_config.yaml"

print(config_dir)

with open(config_dir, "r") as file:
    config = yaml.safe_load(file)
    print(config)


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)

# Store data by experiment group
experiment_data = defaultdict(list)

max_len = 0


def get_positions_from_loaded_data(data, n_agents=None):
    if isinstance(data, dict):
        if 'positions' in data:
            return np.asarray(data['positions'])
        if 'state' in data:
            return extract_positions_from_state_array(data['state'], n_agents)
        if 'training_data' in data and isinstance(data['training_data'], dict):
            nested = data['training_data']
            if 'positions' in nested:
                return np.asarray(nested['positions'])
            if 'state' in nested:
                return extract_positions_from_state_array(nested['state'], n_agents)
        if 'observations' in data:
            return extract_positions_from_state_array(data['observations'], n_agents)
        if 'states' in data:
            return extract_positions_from_state_array(data['states'], n_agents)

    if isinstance(data, list):
        return extract_positions_from_state_array(np.asarray(data), n_agents)

    if isinstance(data, np.ndarray):
        return extract_positions_from_state_array(data, n_agents)

    raise ValueError(f'Unsupported data type for position extraction: {type(data)}')


# ============================================================
# LOAD DATA
# ============================================================

for batch in config['batches']:
    batch_config = infer_batch_env_config(batch)
    n_agents = batch_config.get('n_agents') if batch_config else None

    for experiment in config['experiments']:
        exp_key = f"{batch}-{experiment}"

        for trial in config['trials']:
            # checkpoint_path = Path(
            #     f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/train.dat"
            # )

            # if not checkpoint_path.is_file():
            checkpoint_path = Path(
                f"{config['base_path']}/{batch}/{experiment}/{trial}/logs/evaluation.dat"
            )

            if not checkpoint_path.is_file():
                print(f"Missing file: {checkpoint_path}")
                continue

            try:
                data = load_pickle_or_dill(checkpoint_path)
            except Exception as exc:
                print(f"Failed to load {checkpoint_path}: {exc}")
                continue

            try:
                full_data = get_positions_from_loaded_data(data, n_agents)
            except Exception as exc:
                print(
                    f"Trial {trial}: could not extract positions from {checkpoint_path}: {exc}"
                )
                continue

            full_data = np.asarray(full_data)
            if full_data.ndim == 2:
                full_data = full_data[:, None, :]

            if full_data.ndim == 3 and full_data.shape[-1] != 2:
                try:
                    full_data = extract_positions_from_state_array(full_data, n_agents)
                except Exception as exc:
                    print(
                        f"Trial {trial}: position extraction produced wrong shape {full_data.shape}: {exc}"
                    )
                    continue

            if full_data.ndim != 3 or full_data.shape[-1] != 2:
                print(
                    f"Trial {trial}: positions should be (timesteps, n_agents, 2), got {full_data.shape}"
                )
                continue

            print(
                f"Trial {trial}: {len(full_data)} total timesteps"
            )

            full_data = full_data[: config['datapoints'][0]]

            print(
                f"Trial {trial}: using {len(full_data)} timesteps"
            )

            crumple_metric = []

            for positions in full_data:
                positions = np.array(positions)

                # Radius of gyration
                rg = compute_radius_of_gyration(positions)

                # Shape anisotropy
                anisotropy = compute_anisotropy(positions)

                # Combined anti-crumple metric
                # Higher = elongated chain
                # Lower = compact/crumpled
                metric = rg * anisotropy
                crumple_metric.append(metric)

            # Apply smoothing
            if len(crumple_metric) > config["moving_avg_window_size"]:

                smoothed_data = moving_average(
                    crumple_metric,
                    config["moving_avg_window_size"]
                )

                experiment_data[exp_key].append(
                    smoothed_data
                )

                max_len = max(
                    max_len,
                    len(smoothed_data)
                )

            else:

                print(
                    f"Warning: Trial {trial} "
                    f"has too few data points"
                )


# ============================================================
# PAD ARRAYS
# ============================================================

for exp_key in experiment_data:

    padded_data = []

    for trial_data in experiment_data[exp_key]:

        if len(trial_data) < max_len:

            padding = np.full(
                max_len - len(trial_data),
                np.nan
            )

            padded_data.append(
                np.concatenate(
                    [trial_data, padding]
                )
            )

        else:

            padded_data.append(trial_data)

    experiment_data[exp_key] = padded_data


# ============================================================
# PLOT
# ============================================================

color_idx = 0

for exp_key in experiment_data:

    if not experiment_data[exp_key]:

        print(f"No data for {exp_key}")

        continue

    # Convert list to numpy array
    data_array = np.array(
        experiment_data[exp_key]
    )

    # Mean across trials
    mean_metric = np.nanmean(
        data_array,
        axis=0
    )

    # Standard error
    n_trials = np.sum(
        ~np.isnan(data_array),
        axis=0
    )

    std_metric = np.nanstd(
        data_array,
        axis=0
    )

    se_metric = std_metric / np.sqrt(
        np.maximum(n_trials, 1)
    )

    x = np.arange(len(mean_metric))

    color = plt.cm.tab10(
        color_idx % 10
    )

    color_idx += 1

    # Mean line
    ax.plot(
        x,
        mean_metric,
        linewidth=1,
        label=exp_key,
        color=color
    )

    # Error band
    ax.fill_between(
        x,
        mean_metric - se_metric,
        mean_metric + se_metric,
        alpha=0.3,
        color=color
    )

    print(
        f"{exp_key}: "
        f"{len(experiment_data[exp_key])} trials"
    )


# ============================================================
# FINALIZE
# ============================================================

ax.legend(loc='best')

ax.set_xlabel("Iterations")

ax.set_ylabel("Anti-Crumple Metric")

ax.set_title(
    "Salp Chain Crumpling"
)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()