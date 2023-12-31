# sumo_optimization.py

import wandb
import optuna
from sumo_mmrl import Agent, Env

def create_env(config):
    """
    Create and return the simulation environment based on the provided configuration.

    :param config: The configuration dictionary with environment settings.
    :type config: dict
    :return: An instance of the SUMO simulation environment.
    :rtype: Env
    """
    experiment_path = config['training_settings']['experiment_path']
    sumoconfig = config['training_settings']['sumoconfig']
    num_vehic = config['training_settings']['num_vehic']
    types = config['training_settings']['types']

    return Env(experiment_path, sumoconfig, num_vehic, types)


def create_agent(trial,wandb_run, config):
    """
    Create and return the agent with hyperparameters from the trial and configuration.

    :param trial: The trial object from Optuna for hyperparameter optimization.
    :type trial: optuna.trial.Trial
    :param config: The configuration dictionary with agent settings.
    :type config: dict
    :return: An instance of the agent.
    :rtype: Agent
    """
    # Extract hyperparameters from the trial object
    experiment_path = config['training_settings']['experiment_path']
    learning_rate = trial.suggest_categorical("learning_rate", config['agent_hyperparameters']['learning_rate'])
    learning_rate = trial.suggest_categorical("learning_rate", config['agent_hyperparameters']['learning_rate'])
    gamma = trial.suggest_float("gamma", *config['agent_hyperparameters']['gamma'], log=True)
    epsilon_decay = trial.suggest_categorical("epsilon_decay", config['agent_hyperparameters']['epsilon_decay'])
    batch_size = trial.suggest_categorical("batch_size", config['agent_hyperparameters']['batch_size'])
    memory_size = trial.suggest_categorical("memory_size", config['agent_hyperparameters']['memory_size'])
    epsilon_max = trial.suggest_categorical("epsilon_max", config['agent_hyperparameters']['epsilon_max'])
    epsilon_min = trial.suggest_categorical("epsilon_min", config['agent_hyperparameters']['epsilon_min'])
    n_layers = trial.suggest_categorical("n_layers", config['agent_hyperparameters']['n_layers'])
    layer_sizes = [trial.suggest_categorical(f"layer_{i}_size", config['agent_hyperparameters']['layer_sizes']) for i in range(n_layers)]
    activation = trial.suggest_categorical("activation", config['agent_hyperparameters']['activation'])
    soft_update_factor = trial.suggest_categorical("soft_update_factor", config['agent_hyperparameters']['soft_update_factor'])

    return Agent(12, 4, experiment_path,
                    wandb_run,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    layer_sizes, 
                    activation, 
                    batch_size,
                    soft_update_factor)


def log_performance_metrics(metrics):
    """
    Log performance metrics to Weights & Biases.

    :param metrics: A dictionary containing the metrics to log.
    :type metrics: dict
    """
    wandb.log(metrics)
    
def prepare_wandb_config(trial, config):
    """
    Prepare the configuration dictionary for updating wandb.config with the values from the current trial.

    :param trial: The trial object from Optuna for hyperparameter optimization.
    :type trial: optuna.trial.Trial
    :param config: The configuration dictionary with agent settings.
    :type config: dict
    :return: A dictionary with the current values to update wandb.config.
    :rtype: dict
    """
    # Extracting the current values from the trial
    hyperparameters = {
        "learning_rate": trial.params['learning_rate'],
        "gamma": trial.params['gamma'],
        "epsilon_decay": trial.params['epsilon_decay'],
        "batch_size": trial.params['batch_size'],
        "memory_size": trial.params['memory_size'],
        "epsilon_max": trial.params['epsilon_max'],
        "epsilon_min": trial.params['epsilon_min'],
        "n_layers": trial.params['n_layers'],
        "layer_sizes": [trial.params[f"layer_{i}_size"] for i in range(config['agent_hyperparameters']['n_layers'][0])],  # Assuming n_layers is a list with one element
        "activation": trial.params['activation']
    }

    return hyperparameters
def log_environment_details(env, action, q_values):
    """
    Log environment details and agent actions to Weights & Biases.

    :param env: The simulation environment.
    :type env: Env
    :param action: The action taken by the agent.
    :type action: int
    :param q_values: The Q-values from the agent's decision.
    :type q_values: list
    """
    step = env.get_global_step()
    if step % 2 == 0:
        wandb.log({
            "location": env.get_vehicle_location_edge_id(),
            "best_choice": env.get_best_choice(),
            "agent choice": action,
            "q_values": q_values,
            "out lanes": env.get_out_lanes(),
        })
def log_episode_summary(episode,env, cumulative_reward, agent):
    """
    Log the summary of the episode to Weights & Biases.

    :param episode: The current episode number.
    :type episode: int
    :param cumulative_reward: The cumulative reward for the episode.
    :type cumulative_reward: float
    :param agent: The agent used in the simulation.
    :type agent: Agent
    """
    wandb.log({
        "cumulative_reward": cumulative_reward,
        "epsilon": agent.get_epsilon(),
        "explore_ratio": agent.get_exploration_stats()[0],
        "exploit_ratio": agent.get_exploration_stats()[1],
        "episode": episode,
        "agent_steps": env.get_global_step(),
        "simulation_steps": env.get_steps_per_episode(),
    })
