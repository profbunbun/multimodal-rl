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

TAU = 0.005


class Dagent:

    def __init__(self, state_size, action_size, path) -> None:
        self.path = path
        self.direction_choices = [STRAIGHT, TURN_AROUND, RIGHT, LEFT]
        self.memory = deque(maxlen=100_000)
        self.gamma = 0.98
        self.epsilon = 1
        self.epsilon_max = 1
        self.decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        device = T.device(  # pylint: disable=E1101
            "cuda" if T.cuda.is_available() else "cpu"
        )

        self.policy_net = dqn.DQN(state_size, action_size)
        self.target_net = dqn.DQN(state_size, action_size)
        
        if os.path.exists(path + PATH):
            self.target_net.load_state_dict(T.load(path + PATH))
            # self.policy_net.eval()

        # self.policy_net = dqn.DQN(state_size, action_size)
        # self.target_net = dqn.DQN(state_size, action_size)
        self.policy_net.to(device)
        self.target_net.to(device)

        
        self.loss_fn = nn.HuberLoss()
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(),
        #                           lr=self.learning_rate,)
        # self.optimizer = optim.Adam(self.policy_net.parameters(),
        #                        lr=self.learning_rate,)
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.learning_rate, amsgrad=True)
        
    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def explore(self, options):

        action = np.random.choice(self.direction_choices)
        if action in options:
            return action, 1
        return action, -1

    def exploit(self, state, options):

        act_values = self.policy_net(state)

        action = self.direction_choices[
            T.argmax(act_values)]  # pylint: disable=E1101

        if action in options:
            return action, 1
        return action, -1

    def choose_action(self, state, options):

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

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, new_state, stage in minibatch:
            self.optimizer.zero_grad(set_to_none=True)
            if stage != "done":
                new_state_policy = self.target_net(new_state)
                adjusted_reward = reward + self.gamma * max(new_state_policy)
                output = self.policy_net(state)
                # output = new_state_policy
                target = new_state_policy
                target[action] = adjusted_reward

            else:
                output = self.policy_net(state)
                # target = output.clone()
                target = self.target_net(state)
                target[action] = reward

            loss = self.loss_fn(output, target)
            loss.backward()
            # T.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            # self.optimizer.zero_grad()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def hard_update(self):
       
        policy_net_state_dict = self.policy_net.state_dict()
        
        self.target_net.load_state_dict(policy_net_state_dict)
        
    def save(self):
        T.save(self.policy_net.state_dict(), self.path + PATH)

    def epsilon_decay(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decay
        else:
            self.epsilon = self.epsilon_min

    def epsilon_decay_2(self, episode, episodes):

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

        if self.epsilon > self.epsilon_min:
            episode += 1

            self.epsilon = (1 / 9.5) * math.log(((-episode) + episodes + 1))
            
        else:
            self.epsilon = self.epsilon_min

    def epsilon_null(self):

        self.epsilon = 0.0

    def eps_linear(self, episodes):
        
        if self.epsilon >= self.epsilon_min:
            
            self.epsilon = self.epsilon - 1/(episodes)
            
        else:
            self.epsilon = self.epsilon_min
