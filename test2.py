''' main driver '''
import os
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
import optuna
from sumo_mmrl import Agent,Env
import sqlalchemy
import wandb 
from optuna.pruners import HyperbandPruner

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Configuration parameters
EPISODES = config['training_settings']['episodes']
BATCH_SIZE = config['training_settings']['batch_size']
EXPERIMENT_PATH = config['training_settings']['experiment_path']
SUMOCONFIG = config['training_settings']['sumoconfig']
NUM_VEHIC = config['training_settings']['num_vehic']
TYPES = config['training_settings']['types']
LOG_PATH = config['training_settings']['log_dir_path']


# Define the objective function for Optuna
def objective(trial):
    # Suggested hyperparameters

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.999, 0.9999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024, 2048])
    memory_size = trial.suggest_categorical("memory_size", [10000, 20000, 50000, 100000])
    epsilon_max = trial.suggest_float("epsilon_max", 0.9, 1.0,log=True)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.1,log=True)
    
    n_layers = trial.suggest_int('n_layers', 1, 5, log=True)  # For example, between 1 and 5 layers
    layer_sizes = [trial.suggest_int(f'n_units_l{i}', 16, 128) for i in range(n_layers)]  # Each layer size between 16 and 128


    activation = trial.suggest_categorical("activation", ["relu", "tanh","leaky_relu"])
    env = Env(EXPERIMENT_PATH, SUMOCONFIG, NUM_VEHIC, TYPES)
    dagent = Agent(12, 4, EXPERIMENT_PATH,wandb , learning_rate, gamma, epsilon_decay, epsilon_max, epsilon_min, memory_size, layer_sizes, activation, batch_size)

    wandb.init(project='sumo_mmrl', entity='aaronrls')
    wandb.config.update({"learning_rate": learning_rate, "gamma": gamma, "epsilon_decay": epsilon_decay, "batch_size": batch_size, "memory_size": memory_size, "epsilon_max": epsilon_max, "epsilon_min": epsilon_min, "n_layers": n_layers, "layer_sizes": layer_sizes, "activation": activation})
    
    best_reward = float('-inf')  # Track the best reward for early stopping
    no_improvement_count = 0  # Count episodes with no improvement

    for episode in range(EPISODES):
        cumulative_reward = 0
        env.render("libsumo")
        state, stage, legal_actions = env.reset()
        while stage != "done":
            action, action_index, validator, q_values = dagent.choose_action(state, legal_actions)
            next_state, new_reward, stage, legal_actions = env.step(action, validator)
            if env.get_global_step() % 2 == 0:
                wandb.log({"location": env.get_vehicle_location_edge_id(),
                        "best_choice": env.get_best_choice(),
                        "agent choice": action,
                        "q_values": q_values,
                        "out lanes": env.get_out_lanes(),})
            dagent.remember(state, action_index, new_reward, next_state, done=(stage == "done"))
            cumulative_reward += new_reward
            if len(dagent.memory) > batch_size:
                dagent.replay(batch_size, episode, env.get_global_step())
                dagent.soft_update()
            state = next_state
            

        # Log episode details
        wandb.log({
                            "cumulative_reward": cumulative_reward,
                            "epsilon": dagent.get_epsilon(),
                                "explore_ratio": dagent.get_exploration_stats()[0],
                                "exploit_ratio": dagent.get_exploration_stats()[1],
                                    "episode": episode,
                                    "agent_steps": env.get_global_step(),
                                    "simulation_steps": env.get_steps_per_episode(),
                                        })
        trial.report(cumulative_reward, episode)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon())  # Close the environment properly
        if episode % 30 == 0:
            dagent.hard_update()

        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            dagent.save_model(str(episode))
        

        
    wandb.finish()   
    return cumulative_reward  # The objective value to maximize

def get_next_study_name(storage_url, base_name="study"):
    try:
        # Attempt to connect to the Optuna storage database.
        storage = optuna.storages.RDBStorage(storage_url)
    except Exception as e:
        print(f"Failed to connect to the storage database: {e}")
        return base_name + "_1"

    # Get all existing study names.
    try:
        all_study_summaries = optuna.study.get_all_study_summaries(storage=storage)
    except sqlalchemy.exc.OperationalError:
        # If the table doesn't exist yet, this is likely the first study.
        return base_name + "_1"
    
    all_study_names = [summary.study_name for summary in all_study_summaries]

    # Find the highest existing study number.
    max_num = 0
    for name in all_study_names:
        if name.startswith(base_name):
            try:
                num = int(name.split("_")[-1])  # Assumes the format "study_X"
                max_num = max(max_num, num)
            except ValueError:
                continue

    # Create a new unique study name.
    new_study_name = f"{base_name}_{max_num + 1}"
    return new_study_name

    
def main():
    # Set up Optuna study
    pruner = optuna.pruners.MedianPruner()
    storage_path = "sqlite:///db.sqlite3"
    new_study_name = get_next_study_name(storage_path, base_name="study",)
    study = optuna.create_study(
        storage=storage_path,  # Specify the storage URL here.
        study_name=new_study_name,
        direction="maximize",
        pruner=pruner,
    )

    study.optimize(objective, n_trials=100)

    print(f"Best value: {study.best_value} (params: {study.best_params})")

if __name__ == "__main__":
    main()
