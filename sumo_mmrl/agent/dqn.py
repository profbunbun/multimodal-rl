import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class for approximating Q-values.

    :param int state_size: Size of the input state space.
    :param int action_size: Size of the output action space.
    :param list layer_sizes: Sizes of the hidden layers.
    :param str activation_function: Activation function to use in the layers.
    """


    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 8)
        # self.layer2 = nn.Linear(8, 6)
        self.layer3 = nn.Linear(8, n_actions)


    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        # x = F.leaky_relu(self.layer2(x))
        out = self.layer3(x)
        return out
    
    def init_weights(self, m):
        """
        Initialize the weights of the network.

        :param torch.nn.Module m: The module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)
    
    def get_activation_function(self, name):
        """
        Get the specified activation function.

        :param str name: Name of the activation function.
        :return: The corresponding activation function.
        :rtype: function
        """
        if name == 'relu':
            return F.relu
        elif name == 'leaky_relu':
            return F.leaky_relu
        elif name == 'tanh':
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {name}")
