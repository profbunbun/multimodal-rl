import torch
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacaty):
        
        self.memory = deque([],maxlen=capacaty)

    def remember(self, *args):
        self.memory.append(Transition(*args))

    def replay_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
     
    
    def __len__(self):
        return len(self.memory)