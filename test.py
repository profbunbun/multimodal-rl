''' main driver '''
import os
import json
import traceback
from sumo_mmrl import Agent , DQN , Env , Logger


with open('config.json') as f:
    config = json.load(f)

EPISODES = config['training_settings']['episodes']
BATCH_SIZE = config['training_settings']['batch_size']
EXPERIMENT_PATH = config['training_settings']['experiment_path']
SUMOCONFIG = config['training_settings']['sumoconfig']
NUM_VEHIC = config['training_settings']['num_vehic']
TYPES = config['training_settings']['types']
LOG_PATH= config['training_settings']['log_dir_path']


def main():
    logger = Logger(LOG_PATH, 'config.json')
    env = Env(EXPERIMENT_PATH, SUMOCONFIG, NUM_VEHIC, TYPES)
    dagent = Agent(12, 4, EXPERIMENT_PATH,logger)
    logger.log_config()
    if os.path.exists(LOG_PATH + "/model.pt"):
        model_state, optimizer_state = dagent.get_model_info()
        logger.log_model_and_optimizer_info(model_state , optimizer_state)
    

    try:
        for episode in range(EPISODES + 1):
            accumulated_reward = 0
            steps_per_episode = []


            env.render("libsumo" if episode % 100 != 0 else "libsumo")
            state, stage, legal_actions = env.reset()

            while stage != "done":
                action, action_index, validator, q_values = dagent.choose_action(state, legal_actions)
                
                next_state, new_reward, stage, legal_actions = env.step(action, validator)

                accumulated_reward += new_reward

                if stage == "done":
                    dagent.remember(state, action_index, new_reward, next_state, done=1)
                else:

                    dagent.remember(state, action_index, new_reward, next_state, done=0)

                # step_data = {
                #     'episode': episode,
                #     'step': env.get_global_step(),
                #     'reward': new_reward,
                #     'epsilon': dagent.get_epsilon(),  
                #     'vehicle_location_edge_id': env.get_vehicle_location_edge_id(),  
                #     'destination_edge_id': env.get_destination_edge_id(),
                #     'out_lanes': env.get_out_lanes(), 
                #     'action_chosen': action,
                #     'best_choice': env.get_best_choice(),  
                #     'q_values': q_values,  
                #     'stage': stage,
                #     'done': 1 if stage == "done" else 0
                #     }
                # logger.log_step(step_data)

                if len(dagent.memory) > BATCH_SIZE:
                    dagent.replay(BATCH_SIZE, episode, env.get_global_step())
                    dagent.soft_update()
                    
                state = next_state

            dagent.decay()
            steps_per_episode.append(env.get_steps_per_episode())
            env.close(episode, accumulated_reward, dagent.get_epsilon())
            
            episode_data = {
                'episode': episode,
                'epsilon': dagent.get_epsilon(),
                'episode_reward': accumulated_reward,
                'simulation_steps': steps_per_episode[-1],
                'agent_steps': env.get_global_step(),
                'life': env.get_life()
            }
            logger.log_episode(episode_data)

            if episode % 30 == 0:
                dagent.hard_update()
            


    except Exception as e:
        trace = traceback.format_exc()
        print(f"An exception occurred: {e}\nTraceback: {trace}")
    finally:
        print("Training complete.")
        dagent.save()

if __name__ == "__main__":
    main()
