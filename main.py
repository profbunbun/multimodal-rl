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
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback


config = Utils.load_yaml_config('config.yaml')
EPISODES = config['training_settings']['episodes']


wandb_kwargs= {"project":config['global']['project_name'],
               "entity":config['global']['entity']}
wandbc = WeightsAndBiasesCallback(metric_name=config['wandb']['metric_name'],wandb_kwargs=wandb_kwargs)

@wandbc.track_in_wandb()
def objective(trial):

    env = so.create_env(config=config)
    dagent= so.create_trial_agent(trial, config=config)
  
    batch_rewards = []
    batch_avg_reward = 0


    for episode in range(EPISODES):
        
        cumulative_reward = 0

        env.render("libsumo")
        # env.render("gui")
        state, stage, legal_actions = env.reset()
        
        while stage != "done":
            action, action_index, validator = dagent.choose_action(state, legal_actions)


            next_state, new_reward, stage, legal_actions = env.step(action, validator)


            dagent.remember(state, action_index, next_state, new_reward, done=(stage == "done"))
            cumulative_reward += new_reward

            if len(dagent.memory) > dagent.batch_size:
                dagent.replay(dagent.batch_size)
                dagent.hard_update()
            state = next_state

        


        batch_rewards.append(cumulative_reward)
        batch_np = np.array(batch_rewards)
        batch_avg_reward = batch_np.mean()
        
        wandb.log({
        "cumulative_reward": cumulative_reward,
        "epsilon": dagent.get_epsilon(),
        "episode": episode,
        "agent_steps": env.agent_step,
        "simulation_steps": env.sumo.simulation.getTime(),
        "batch_avg_reward": batch_avg_reward
        })

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon())
        # env.quiet_close()
        
        trial.report(cumulative_reward, episode)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    wandb.finish()

    print("Batch Complete")      
    return cumulative_reward

def main():
    """
    Main function to execute the optimization process.

    This function parses the command line arguments for the study name, sets up the Optuna study,
    and starts the optimization process. The results are printed to the console.
    """

    study_name = sys.argv[1] if len(sys.argv) > 1 else None
    storage_path = config['optuna']['storage_path']
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    
    study = Utils.setup_study(study_name, storage_path, pruner)
    study.optimize(objective, n_trials=16, callbacks=[wandbc])

    print(f"Best value: {study.best_value} (params: {study.best_params})")

if __name__ == "__main__":
    main()

