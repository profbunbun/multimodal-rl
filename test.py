""" import stuff """
from sumo_mmrl import Basic, Dagent, Vagent

# import time
EPISODES = 50_000
STEPS = 1000
BATCH_SIZE = 64
MIN_MEMORY = 1000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"
NUM_VEHIC = 4
TYPES = 2


def main():
    """
    main _summary_

    _extended_summary_
    """

    env = Basic(EXPERIMENT_PATH, SUMOCONFIG, STEPS, NUM_VEHIC, TYPES)
    dagent = Dagent(10, 4, EXPERIMENT_PATH)
    # vagent = Vagent(10, NUM_VEHIC, EXPERIMENT_PATH)

    for episode in range(EPISODES + 1):
        accumulated_reward = 0

        # if (episode) % 100 == 0:
        #     env.render("gui")
        # else:
        #     env.render("libsumo")
        env.render("libsumo")

        state, done, legal_actions = env.reset()
        # (vaction) = vagent.choose_action(state)

        while not env.done:
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

            # agent.epsilon_null()
            # agent.epsilon_decay()
        if episode < (0.5 * EPISODES):
            dagent.epsilon_decay_3(episode, (0.5 * EPISODES))

        else:
            dagent.epsilon_decay_2(episode, (0.5 * EPISODES))

        env.close(episode, dagent.epsilon)


if __name__ == "__main__":
    main()
