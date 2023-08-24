""" import stuff """
import torch as T
import numpy as np
from Env.env import Basic

from Connector.utility import Utility
from Agent.agent import Agent


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
        if (episode) % 10 == 0:
            env.render("gui")
        else:
            env.render("libsumo")

        state, done, options = env.reset()

        state = T.from_numpy(state)  # pylint: disable=E1101

        step = 0
        agent_step = 0

        accumulated_reward = 0
        action, action_index = agent.act(state, options)
        # next_state,new_reward, done,out_mask = env.step(action)

        while not env.done:
            if env.make_choice_flag:
                next_state, new_reward, done, options = env.step(action)
                next_state = T.from_numpy(next_state)
                accumulated_reward += new_reward
                agent_step += 1
                agent.remember(state, action_index, new_reward, next_state, done)
                state = next_state
                action, action_index = agent.act(state, options)
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
            else:
                env.nullstep()

            step += 1

        # agent.epsilon_decay()
        agent.epsilon_decay_3(episode, EPISODES)

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
            f" **** step: {step}",
            f"*** Agent steps: {agent_step}",
        )
        x = [i + 1 for i in range(len(rewards))]
        file_name = "sumo-agent.png"

        util.plot_learning(x, rewards, eps_history, file_name)
        env.close()


if __name__ == "__main__":
    main()
