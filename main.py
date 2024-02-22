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
wandbc = WeightsAndBiasesCallback(metric_name=config['wandb']['metric_name'],wandb_kwargs=wandb_kwargs,as_multirun=True)

@wandbc.track_in_wandb()
def objective(trial):

    env = so.create_env(config=config)
    dagent= so.create_trial_agent(trial, config=config)
  
    batch_rewards = []
    batch_avg_reward = 0
    


    for episode in range(EPISODES):
        route_taken = []
        
        cumulative_reward = 0

        env.render("libsumo")
        # env.render("gui")
        state, stage, legal_actions, edge = env.reset()
        route_taken.append(edge)
        
        while stage != "done":
            action, action_index, validator = dagent.choose_action(state, legal_actions)


            next_state, new_reward, stage, legal_actions, edge = env.step(action, validator)
            route_taken.append(edge)


            dagent.remember(state, action_index, next_state, new_reward, done=(stage == "done"))
            cumulative_reward += new_reward

            if len(dagent.memory) > dagent.batch_size:
                dagent.replay(dagent.batch_size)
                dagent.hard_update()
            state = next_state

        


        batch_rewards.append(cumulative_reward)
        batch_np = np.array(batch_rewards)
        batch_avg_reward = batch_np.mean()
        distance_travled = env.get_route_length(route_taken)
        
        wandb.log({
        "cumulative_reward": cumulative_reward,
        "epsilon": dagent.get_epsilon(),
        "episode": episode,
        "agent_steps": env.agent_step,
        "simulation_steps": env.sumo.simulation.getTime(),
        "batch_avg_reward": batch_avg_reward,
        "Distance": distance_travled
        })

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon(), distance_travled)
        # env.quiet_close()
        
        if episode > (EPISODES//2):
            trial.report(cumulative_reward, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
    wandb.finish()

    # print("Batch Complete")      
    return cumulative_reward, distance_travled

def main():
    

    study_name = sys.argv[1] if len(sys.argv) > 1 else None
    storage_path = config['optuna']['storage_path']
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(
                    storage=storage_path,
                    study_name=study_name,
                    direction=["maximize","minimize"],
                    pruner=pruner,
                )
    study.optimize(objective, n_trials=10, callbacks=[wandbc])



if __name__ == "__main__":
    main()

