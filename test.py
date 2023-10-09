from sumo_mmrl import Basic, Dagent

# import time
EPISODES = 1_000
STEPS = 1000
BATCH_SIZE = 32
MIN_MEMORY = 1000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"
NUM_VEHIC = 1
TYPES = 2


def main():

    env = Basic(EXPERIMENT_PATH, SUMOCONFIG, STEPS, NUM_VEHIC, TYPES, "A")
    dagent = Dagent(10, 4, EXPERIMENT_PATH)
    
    for episode in range(EPISODES + 1):
        accumulated_reward = 0

        # if (episode) % 100 == 0:
        #     env.render("gui")
        # else:
        #     env.render("libsumo")
        env.render("libsumo")

        state, stage, legal_actions = env.reset()

        while stage != "done":
            (action, action_index, validator) = dagent.choose_action(
                state, legal_actions
            )

            (next_state,
             new_reward,
             stage,
             legal_actions) = env.step(action, validator)

            accumulated_reward += new_reward
            dagent.remember(state, action_index, new_reward, next_state, stage)
            state = next_state
            
            if len(dagent.memory) > BATCH_SIZE:
                dagent.replay(BATCH_SIZE)

        # dagent.eps_linear(EPISODES-100)

        if episode < (0.5 * EPISODES):
            dagent.epsilon_decay_3(episode, (0.5 * EPISODES))

        else:
            dagent.epsilon_decay_2(episode, (0.5 * EPISODES))

        env.close(episode, dagent.epsilon)
        dagent.save()


if __name__ == "__main__":
    main()
