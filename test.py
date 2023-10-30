# import os
from sumo_mmrl import Dagent
from sumo_mmrl.environment.env import Env

# import time
EPISODES = 5_000
STEPS = 500
BATCH_SIZE = 32
# MIN_MEMORY = 2000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"
NUM_VEHIC = 1
TYPES = 1


def main():
    env = Env(EXPERIMENT_PATH, SUMOCONFIG, STEPS, NUM_VEHIC, TYPES, "A")
    dagent = Dagent(11, 4, EXPERIMENT_PATH)

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
                state, legal_actions)

            (next_state, new_reward, stage,
             legal_actions) = env.step(action, validator)

            accumulated_reward += new_reward

            dagent.remember(state, action_index, new_reward, next_state, done=0)

            state = next_state

            if len(dagent.memory) > BATCH_SIZE:
                dagent.replay(BATCH_SIZE)

        dagent.remember(state, action_index, new_reward, next_state, done=1)
        if episode % 2 == 0:
            dagent.soft_update()
        if episode % BATCH_SIZE == 0:
            dagent.hard_update()
        # dagent.eps_linear(EPISODES)

        # dagent.epsilon_null()

        if episode < (0.8 * EPISODES):
            dagent.eps_linear(0.8 * EPISODES)

        else:
            dagent.epsilon_decay_2(episode, (0.2 * EPISODES))

        env.close(episode, dagent.epsilon,
                  accumulated_reward)
        dagent.save()
    # os.system("beep -f 555 -l 460")


if __name__ == "__main__":
    main()
