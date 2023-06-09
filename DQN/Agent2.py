import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DQN.DQNetwork import DQN
from DQN.ReplayMemory import ReplayMemory
import math 
import random


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class Agent2:
    
    def __init__(self,n_observations,n_actions) -> None:
        
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)
        pass
    
    
    
    
    





    def select_action(self,state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with T.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return T.tensor([[T.action_space.sample()]], device= self.device, dtype=T.long)
    
    



