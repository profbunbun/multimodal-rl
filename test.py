from sumo_mmrl import Basic, Dagent

# import time
EPISODES = 1_000
STEPS = 1000
BATCH_SIZE = 64
MIN_MEMORY = 1000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"
NUM_VEHIC = 2
TYPES = 2


def main():

    env = Basic(EXPERIMENT_PATH, SUMOCONFIG, STEPS, NUM_VEHIC, TYPES)
    dagent = Dagent(10, 4, EXPERIMENT_PATH)
    
    for episode in range(EPISODES + 1):
        accumulated_reward = 0

        if (episode) % 100 == 0:
            env.render("gui")
        else:
            env.render("libsumo")

        state, done, legal_actions = env.reset()

        while  env.done != True:
            (action, action_index, validator) = dagent.choose_action(
                state, legal_actions
            )

            (next_state, new_reward, done, legal_actions) = env.step(action,
                                                                     validator)

            accumulated_reward += new_reward
            dagent.remember(state, action_index, new_reward, next_state, done)
            state = next_state
            
            if len(dagent.memory) > BATCH_SIZE:
                dagent.replay(BATCH_SIZE)

        if episode < (0.5 * EPISODES):
            dagent.epsilon_decay_3(episode, (0.5 * EPISODES))

        else:
            dagent.epsilon_decay_2(episode, (0.5 * EPISODES))

        env.close(episode, dagent.epsilon)


if __name__ == "__main__":
    main()
