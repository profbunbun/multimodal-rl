from sumo_mmrl import Dagent
from sumo_mmrl.environment.env2 import Env

# import time
EPISODES = 10_000
STEPS = 1000
BATCH_SIZE = 64
MIN_MEMORY = 1000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"
NUM_VEHIC = 1
TYPES = 1


def main():
    env = Env(EXPERIMENT_PATH, SUMOCONFIG, STEPS, NUM_VEHIC, TYPES, "B")
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
                state, legal_actions)
        

            (next_state, new_reward, stage,
             legal_actions) = env.step(action, validator)

            accumulated_reward += new_reward
            dagent.remember(state, action_index, new_reward, next_state, stage)
            state = next_state

            if len(dagent.memory) > BATCH_SIZE:
                dagent.replay(BATCH_SIZE)

        # dagent.epsilon_decay()

        if episode < (0.1 * EPISODES):
            dagent.epsilon_decay_3(episode, (0.1 * EPISODES))

        else:
            dagent.epsilon_decay_2(episode, (0.1 * EPISODES))

        env.close(episode, dagent.epsilon, accumulated_reward)
        dagent.save()


if __name__ == "__main__":
    main()
