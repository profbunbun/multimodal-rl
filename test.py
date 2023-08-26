""" import stuff """
import numpy as np
from sumo_mmrl import Basic, Utility, Agent

# from Connector.utility import Utility
# from Agent.agent import Agent


EPISODES = 100
STEPS = 1000
BATCH_SIZE = 32
MIN_MEMORY = 1000
SUMOCONFIG = "Nets/3x3b.sumocfg"


def main():
    """
    main _summary_

    _extended_summary_
    """

    env = Basic(SUMOCONFIG, STEPS)
    agent = Agent(6, 4)
    util = Utility()
    rewards, eps_history = [], []

    for episode in range(EPISODES):

        accumulated_reward = 0

        if (episode) % 1000 == 0:
            env.render("gui")
        else:
            env.render("libsumo")

        state, done, legal_actions = env.reset()

        while not env.done:

            action, action_index = agent.choose_action(state, legal_actions)
            (next_state,
             new_reward,
             done,
             legal_actions) = env.step(action)

            accumulated_reward += new_reward

            agent.remember(state, action_index, new_reward, next_state, done)

            state = next_state
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        # agent.epsilon_decay()
        agent.epsilon_decay_2(episode, EPISODES)

        acc_r = float(accumulated_reward)
        # r = float(new_reward)
        rewards.append(acc_r)

        eps_history.append(agent.epsilon)
        avg_reward = np.mean(rewards[-100:])

        # avg_reward = np.mean(rewards)

        print(
            "EP: ",
            episode,
            f"Reward: {acc_r:.3}",
            f" Average Reward  {avg_reward:.3}",
            f"epsilon {agent.epsilon:.5}",
            f" **** step: {env.steps}",
            f"*** Agent steps: {env.agent_step}",
        )
        x = [i + 1 for i in range(len(rewards))]
        file_name = "sumo-agent.png"

        util.plot_learning(x, rewards, eps_history, file_name)
        env.close()


if __name__ == "__main__":
    main()
