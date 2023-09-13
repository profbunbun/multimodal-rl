"""Module providing deque data structure"""
from collections import deque
import os
import random
import math
import torch as T
from torch import nn
from torch import optim
import numpy as np
from . import dqn

random.seed(0)


STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"

PATH = "/Models/model.pt"


class Agent:

    """
     _summary_

    _extended_summary_
    """

    def __init__(self, state_size, action_size, path) -> None:
        self.path = path
        self.direction_choices = [STRAIGHT, TURN_AROUND, RIGHT, LEFT]
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 0.997
        self.epsilon_max = 0.9997
        self.decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        device = T.device(  # pylint: disable=E1101
            "cuda" if T.cuda.is_available() else "cpu"
        )
        # device = T.device("cuda:1")
        self.policy_net = dqn.DQN(state_size, action_size)
        # net = dqn.DQN(state_size, action_size)
        if T.cuda.device_count() > 1:
            # print("Let's use", T.cuda.device_count(), "GPUs!")
            self.policy_net = nn.DataParallel(self.policy_net)
        self.policy_net.to(device)
        self.loss = nn.HuberLoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.learning_rate)

        if os.path.exists(path + PATH):
            self.policy_net = dqn.DQN(state_size, action_size)
            self.policy_net.load_state_dict(T.load(path + PATH))
            self.policy_net.eval()
        else:
            self.policy_net = dqn.DQN(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        """
        remember _summary_

        _extended_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        self.memory.append((state, action, reward, next_state, done))

    def explore(self, options):
        """
        explore _summary_
        _extended_summary_

        """
        action = np.random.choice(self.direction_choices)
        if action in options:
            return action, 1
        return action, -1

    def exploit(self, state, options):
        """
        exploit _summary_

        _extended_summary_
        """

        act_values = self.policy_net(state)

        action = self.direction_choices[
            T.argmax(act_values)]  # pylint: disable=E1101

        if action in options:
            return action, 1
        return action, -1

    def choose_action(self, state, options):
        """
        act _summary_

        _extended_summary_

        Args:
            state (_type_): _description_
            options (_type_): _description_

        Returns:
            _type_: _description_
        """
        available_choices = list(options.keys())
        rando = np.random.rand()

        if rando < self.epsilon:
            action, valid = self.explore(available_choices)
            return action, self.direction_choices.index(action), valid

        action, valid = self.exploit(state, available_choices)

        if valid != -1:
            return action, self.direction_choices.index(action), valid
        return action, self.direction_choices.index(action), valid

    def replay(self, batch_size):
        """
        replay _summary_

        _extended_summary_

        Args:
            batch_size (_type_): _description_
        """

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, new_state, done in minibatch:
            if not done:
                new_state_policy = self.policy_net(new_state)

                adjusted_reward = reward + self.gamma * max(new_state_policy)

                output = self.policy_net(state)
                target = output.detach().clone()
                target[action] = adjusted_reward

            else:
                output = self.policy_net(state)
                target = output.detach().clone()
                target[action] = reward

                out = self.loss(output, target)
                self.optimizer.zero_grad()
                out.mean().backward()
                self.optimizer.step()
                T.save(self.policy_net.state_dict(), self.path + PATH)

    def epsilon_decay(self):
        """
        epsilon_decay _summary_

        _extended_summary_
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        else:
            self.epsilon = self.epsilon_min

    def epsilon_decay_2(self, episode, episodes):
        """
        epsilon_decay_2 _summary_

        _extended_summary_

        Args:
            episode (_type_): _description_
            episodes (_type_): _description_
        """
        if self.epsilon > self.epsilon_min:
            if episode > (5 / 10 * episodes):
                self.epsilon *= self.decay

            else:
                self.epsilon = self.epsilon_max - 1.01 ** (
                    10 * episode - ((5.4 / 10 * episodes) * 10)
                )
        else:
            self.epsilon = self.epsilon_min

    def epsilon_decay_3(self, episode, episodes):
        """
        epsilon_decay_3 _summary_

        _extended_summary_

        Args:
            episode (_type_): _description_
            episodes (_type_): _description_
        """
        if self.epsilon > self.epsilon_min:
            episode += 1

            self.epsilon = (1 / 9.5) * math.log(((-episode) + episodes + 1))
            # self.epsilon_max-1.01**(10*episode-((4.4/10 * episodes)*10))
        else:
            self.epsilon = self.epsilon_min

    def epsilon_null(self):
        """
        epsilon_null _summary_

        _extended_summary_
        """
        self.epsilon = self.epsilon_min
