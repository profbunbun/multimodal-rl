"""Main driver for the SUMO MMRL project.

This module sets up and runs the training environment using Weights & Biases for experiment tracking.
It runs the main training loop without hyperparameter optimization.
"""
from sumo_mmrl import Utils
from sumo_mmrl.environment import sim_manager as so
import wandb

config = Utils.load_yaml_config('config.yaml')
EPISODES = config['training_settings']['episodes']

wandb.init(project=config['wandb']['project_name'],
           entity=config['wandb']['entity'],
           name=config['wandb']['name'],
           group=config['wandb']['group'],
           config=config)

def main_training_loop():
    env = so.create_env(config=config)
    dagent = so.create_agent(config=config)

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

        distance_traveled = env.get_route_length(route_taken)
        
        wandb.log({
            "cumulative_reward": cumulative_reward,
            "epsilon": dagent.get_epsilon(),
            "episode": episode,
            "agent_steps": env.agent_step,
            "simulation_steps": env.sumo.simulation.getTime(),
            "Distance": distance_traveled
        })

        dagent.decay()
        env.close(episode, cumulative_reward, dagent.get_epsilon(), distance_traveled)
    
    wandb.finish()

if __name__ == "__main__":
    main_training_loop()
