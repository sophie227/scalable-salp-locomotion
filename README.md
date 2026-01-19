# scalable-salp-locomotion
This repository contains the code and experiments of the Learning Scalable Salp Locomotion Paper. 

### To install
Project requires Python 3.12

Run the following commands inside after cloning:
1. `sudo apt install swig build-essential python3-dev`
2. `git submodule init`
3. `git submodule update`
4. `cd VectorizedMultiAgentSimulator`
5. `pip install -e .`
6. `cd ..`
7. `pip install -r requirements.txt`

### To build an experiment

Run the jupyter notebook `src/learning/builders/experiment_builder.ipynb`

### To run experiment

Run the following command for running experiments in different modalities:

- `python3 src/run_trial.py --batch=salp_navigate_8a --name=gcn --algorithm=ppo --environment=salp_navigate`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel bash run_trial.sh salp_navigate_8a ppo salp_navigate test_id ::: gcn gat graph_transformer transformer_full transformer_encoder transformer_decoder`

### To reproduce paper's results

Figure 3
- `parallel bash run_trial.sh salp_navigate_8a ppo salp_navigate test_id ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full`
- `parallel bash run_trial.sh salp_navigate_16a ppo salp_navigate test_id ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full`

Figure 4
- `parallel bash evaluate.sh salp_navigate_8a ppo salp_navigate test_id ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full`
- `parallel bash evaluate.sh salp_navigate_16a ppo salp_navigate test_id ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full`

To plot Figure 3 run the notebook `src/plotting/plot_learning_curves_std_error_multiple.ipynb`

To plot Figure 4 run the notebook `src/plotting/plot_zero_shot_curves_mult.ipynb`