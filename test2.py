"""Main driver for the SUMO MMRL project.

This module sets up and runs the training environment using Optuna for hyperparameter optimization
and Weights & Biases for experiment tracking. It defines an objective function for the optimization
process and a main function to initiate the study.
"""
import sys
import optuna
from sumo_mmrl import  Utils
from sumo_mmrl.utilities import sim_manager as so
import numpy as np
from functools import wraps
import time
import os
from csv import writer
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback


config = Utils.load_yaml_config('config.yaml')
EPISODES = config['training_settings']['episodes']
logs_path = config['logs']['path']
experiment_path = config['training_settings']['experiment_path']

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
    dagent= so.create_trial_agent(trial, config=config)

    best_reward = float('-inf') 
    batch_rewards = []
    batch_avg_reward = 0
    headers = ['Episodes','Reward','Epsilon','Steps']

    with open(os.path.join(experiment_path,logs_path),'a') as logger:
        write = writer(logger)
        write.writerow(headers)
        for episode in range(EPISODES):
            
            cumulative_reward = 0

            env.render("libsumo")
            # env.render("gui")
            state, stage, legal_actions = env.reset()
            
            while stage != "done":
                action, action_index, validator = dagent.choose_action(state, legal_actions)


                next_state, new_reward, stage, legal_actions = env.step(action, validator)

                # dagent.remember(state, action_index, next_state,new_reward)
                dagent.remember(state, action_index, next_state, new_reward, done=(stage == "done"))
                cumulative_reward += new_reward
                if len(dagent.memory) > dagent.batch_size:
                    dagent.replay(dagent.batch_size)
                    dagent.hard_update()
                state = next_state

            

            dagent.decay()

            batch_rewards.append(cumulative_reward)
            batch_np = np.array(batch_rewards)
            batch_avg_reward = batch_np.mean() 
            write.writerow([int(episode),
                            round(float(cumulative_reward),4),
                            round(float(dagent.get_epsilon()),4),
                            int(env.agent_step)])
            

            # env.close(episode, cumulative_reward, dagent.get_epsilon())
            env.quiet_close()


            if cumulative_reward > best_reward:
                best_reward = cumulative_reward
                dagent.save_model(str(episode))
            
            trial.report(batch_avg_reward, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        logger.close()  
    return batch_avg_reward

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

