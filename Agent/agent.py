"""Module providing deque data structure"""
from collections import deque
import os
import random
import math
import torch as T
from torch import nn
from torch import optim
import numpy as np
from Agent import dqn

random.seed(0)
T.autograd.set_detect_anomaly(True)

STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"

PATH = "Models/model.pt"


class Agent:

    """
     _summary_

    _extended_summary_
    """

    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.direction_choices = [STRAIGHT, TURN_AROUND, RIGHT, LEFT]
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 0.997
        self.epsilon_max = 0.9997
        self.decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        # pylint: disable=E1101
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        # pylint: enable=E1101
        if os.path.exists(PATH):
            self.policy_net = dqn.DQN(self.state_size,
                                      self.action_size).to(self.device)
            self.policy_net.load_state_dict(T.load(PATH))
            self.policy_net.eval()
        else:
            self.policy_net = dqn.DQN(self.state_size,
                                      self.action_size).to(self.device)

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

    def explore(self):
        """
        explore _summary_
        _extended_summary_

        """
        return

    def exploit(self):
        """
        exploit _summary_

        _extended_summary_
        """
        return

    # make choice function
    def act(self, state, options):
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
            act = np.random.choice(available_choices)
            action_index = self.direction_choices.index(act)
            return options[act], action_index

        act_values = self.policy_net.forward(state)

        # q-val
        # act=T.argmax(act_values)
        # pylint: disable=E1101
        act = self.direction_choices[T.argmax(act_values)]
        # pylint: enable=E1101
        action_index = self.direction_choices.index(act)
        return act, action_index

    # Train the model
    def replay(self, batch_size):
        """
        replay _summary_

        _extended_summary_

        Args:
            batch_size (_type_): _description_
        """
        # T.cuda.empty_cache()
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, new_state, done in minibatch:
            reward = reward.float()
            reward = reward.to(self.device)

            if not done:
                new_state_policy = self.policy_net.forward(
                    new_state).to(self.device)

                # pylint: disable=E1101
                adjusted_reward = reward + self.gamma * T.max(new_state_policy)
                # pylint: enable=E1101
                output = self.policy_net.forward(state).to(self.device)

                target = output.detach().clone()
                target[action] = adjusted_reward
                # out_mask=out_mask.detach().clone()

                # for i in enumerate(target):
                #     if out_mask[i[0]]==0:
                #         target[i[0]]=-1000

                target = target.to(self.device)

            else:
                output = self.policy_net.forward(state).to(self.device)
                target = output.detach().clone()
                target[action] = reward
                # out_mask=out_mask.detach().clone()

                # for i in enumerate(target):
                #     if out_mask[i[0]]==0:
                #         target[i[0]]=-1000

                target = target.to(self.device)

                # loss function
                # loss =  nn.MSELoss()
                # loss = nn.L1Loss()
                loss = nn.HuberLoss()

                # optimize parameters
                optimizer = optim.Adam(
                    self.policy_net.parameters(), lr=self.learning_rate
                )

                out = loss(output, target)

                # out=self.loss(output,target)
                optimizer.zero_grad()
                # out.backward()
                out.backward(retain_graph=True)
                optimizer.step()
                # T.cuda.empty_cache()

                T.save(self.policy_net.state_dict(), PATH)
                # return loss.item()

    # trying differnt epsilon decay
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
            self.epsilon = self.epsilon

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
            self.epsilon = self.epsilon
