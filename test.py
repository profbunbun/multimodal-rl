""" import stuff """
from sumo_mmrl import Basic, Agent

EPISODES = 20000
STEPS = 1000
BATCH_SIZE = 32
MIN_MEMORY = 1000
EXPERIMENT_PATH = "Experiments/3x3"
SUMOCONFIG = "/Nets/3x3b.sumocfg"


def main():
    """
    main _summary_

    _extended_summary_
    """
    env = Basic(EXPERIMENT_PATH , SUMOCONFIG, STEPS)
    agent = Agent(10, 4,EXPERIMENT_PATH)

    for episode in range(EPISODES):

        accumulated_reward = 0

        if (episode) % 100 == 0:
            env.render("gui")
        else:
            env.render("libsumo")

        state, done, legal_actions, distance_mask = env.reset()

        while not env.done:

            action, action_index, validator = agent.choose_action(state, legal_actions)
            # if validator != -1:
            (next_state,
            new_reward,
            done,
            legal_actions,
            distance_mask) = env.step(action, validator)

            accumulated_reward += new_reward

            agent.remember(state, action_index,
                           new_reward, next_state, done, distance_mask)

            state = next_state
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        agent.epsilon_decay_3(episode, EPISODES)
        # agent.epsilon_null()

        env.close(episode, agent.epsilon)


if __name__ == "__main__":
    main()
