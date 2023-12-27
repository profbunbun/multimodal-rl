import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, n_layers, layer_size, activation_function):
        super(DQN, self).__init__()
        
        # Create a list to hold all the layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_size, layer_size))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))
        
        # Output layer
        self.layers.append(nn.Linear(layer_size, action_size))

        # Initialize weights
        self.layers.apply(self.init_weights)
        
        # Set activation function
        self.activation_function = self.get_activation_function(activation_function)

    def forward(self, x_net):
        # Pass the input through each layer
        for layer in self.layers[:-1]:
            x_net = self.activation_function(layer(x_net))
        
        # No activation for the last layer
        x_net = self.layers[-1](x_net)
        
        return x_net
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)
    
    def get_activation_function(self, name):
        if name == 'relu':
            return F.relu
        elif name == 'leaky_relu':
            return F.leaky_relu
        elif name == 'tanh':
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {name}")
