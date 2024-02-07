
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
          
            nn.Linear(64,n_actions)
            )


    def forward(self, x):
        
        return self.layers(x)
    