"""Main driver for the SUMO MMRL project.

This module sets up and runs the training environment using Optuna for hyperparameter optimization
and Weights & Biases for experiment tracking. It defines an objective function for the optimization
process and a main function to initiate the study.
"""
import sys
import optuna
from sumo_mmrl import  Utils
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from sumo_mmrl import sumo_opotimize as so

config = Utils.load_yaml_config('config.yaml')
EPISODES = config['training_settings']['episodes']
wandb_kwargs= {"project":"sumo_mmrl","entity":"aaronrls"}
wandbc = WeightsAndBiasesCallback(metric_name='cumulative_reward',wandb_kwargs=wandb_kwargs)

@wandbc.track_in_wandb()
def objective(trial):
    """
    Objective function for Optuna optimization.

    This function sets up the environment, agent, and performs the training loop, logging information
    and tracking the best reward obtained. It's used by the Optuna study to evaluate the performance
    of different sets of hyperparameters.

    :param trial: An individual trial object with hyperparameters suggested by Optuna.
    :type trial: optuna.trial.Trial
    :return: The best cumulative reward achieved during the training.
    :rtype: float
    """

    env = so.create_env(config=config)
    dagent= so.create_agent(trial,wandb,config=config)
    wandb_trial_config = so.prepare_wandb_config(trial, config)
    wandb.config.update(wandb_trial_config)
    best_reward = float('-inf') 
    
    for episode in range(EPISODES):
        
        cumulative_reward = 0
        env.render("no_gui")
        state, stage, legal_actions = env.reset()
        while stage != "done":
            action, action_index, validator, q_values = dagent.choose_action(state, legal_actions)
            next_state, new_reward, stage, legal_actions = env.step(action, validator)
            so.log_environment_details(env, action, q_values)
            dagent.remember(state, action_index, new_reward, next_state, done=(stage == "done"))
            cumulative_reward += new_reward
            if len(dagent.memory) > dagent.batch_size:
                dagent.replay(dagent.batch_size)
                dagent.soft_update()
            state = next_state

        so.log_episode_summary(episode, env, cumulative_reward, dagent)
        
        trial.report(cumulative_reward, episode)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon())
        
        if episode % 30 == 0:
            dagent.hard_update()

        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            dagent.save_model(str(episode))
  
    wandb.finish()   
    return cumulative_reward 

 
def main():
    """
    Main function to execute the optimization process.

    This function parses the command line arguments for the study name, sets up the Optuna study,
    and starts the optimization process. The results are printed to the console.
    """

    study_name = sys.argv[1] if len(sys.argv) > 1 else None
    storage_path = config['optuna']['storage_path']
    pruner = optuna.pruners.MedianPruner()
    
    study = Utils.setup_study(study_name, storage_path, pruner)
    study.optimize(objective, n_trials=100, callbacks=[wandbc])

    print(f"Best value: {study.best_value} (params: {study.best_params})")


if __name__ == "__main__":
    main()

