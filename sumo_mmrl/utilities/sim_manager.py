# sumo_optimization.py

import wandb
import optuna
from sumo_mmrl import Agent, Env
from ..environment.net_parser import NetParser



def create_env(config):
    """
    Create and return the simulation environment based on the provided configuration.
    :param config: The configuration dictionary with environment settings.
    :type config: dict
    :return: An instance of the SUMO simulation environment.
    :rtype: Env
    """
    path = config['training_settings']['experiment_path']
    sumo_config_path = path + config['training_settings']['sumoconfig']
    parser = NetParser(sumo_config_path)
    edge_locations = (
            parser.get_edge_pos_dic()
        )  
    bussroute = parser.get_route_edges()
    out_dict = parser.get_out_dic()
    index_dict = parser.get_edge_index()

    return Env(config, edge_locations, bussroute, out_dict, index_dict)


def create_agent(config):
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
    learning_rate = config['agent_hyperparameters']['learning_rate']
    gamma = config['agent_hyperparameters']['gamma']
    epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
    batch_size = config['agent_hyperparameters']['batch_size']
    memory_size = config['agent_hyperparameters']['memory_size']
    epsilon_max = config['agent_hyperparameters']['epsilon_max']
    epsilon_min = config['agent_hyperparameters']['epsilon_min']
    n_layers = config['agent_hyperparameters']['n_layers']
    layer_sizes = config['agent_hyperparameters']['layer_sizes']
    activation = config['agent_hyperparameters']['activation']

    return Agent(14, 6, experiment_path,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    batch_size,)

def create_trial_agent(config):
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
    learning_rate = config['agent_hyperparameters']['learning_rate']
    gamma = config['agent_hyperparameters']['gamma']
    epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
    batch_size = config['agent_hyperparameters']['batch_size']
    memory_size = config['agent_hyperparameters']['memory_size']
    epsilon_max = config['agent_hyperparameters']['epsilon_max']
    epsilon_min = config['agent_hyperparameters']['epsilon_min']
    n_layers = config['agent_hyperparameters']['n_layers']
    layer_sizes = config['agent_hyperparameters']['layer_sizes']
    activation = config['agent_hyperparameters']['activation']

    return Agent(14, 6, experiment_path,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    batch_size,)


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

    wandb.log({
        "location": env.get_vehicle_location_edge_id(),
        "best_choice": env.get_best_choice(),
        "agent choice": action,
        "q_values": q_values,
        "out lanes": env.get_out_lanes(),
    })
def log_episode_summary(episode,env, cumulative_reward, agent, batch_avg_reward):
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
        "episode": episode,
        "agent_steps": env.get_global_step(),
        "simulation_steps": env.get_steps_per_episode(),
        "batch_avg_reward": batch_avg_reward

    })


