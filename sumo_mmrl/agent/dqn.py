import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, layer_sizes, activation_function):
        super(DQN, self).__init__()
        
        # Verify that layer_sizes is a list and has at least one layer size
        assert isinstance(layer_sizes, list) and len(layer_sizes) > 0, "layer_sizes must be a list with at least one element"

        # Create a list to hold all the layers
        self.layers = nn.ModuleList()
        
        # Input layer
        previous_layer_size = state_size
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_layer_size = layer_size
        
        # Output layer
        self.layers.append(nn.Linear(previous_layer_size, action_size))

        # Initialize weights
        self.layers.apply(self.init_weights)
        
        # Set activation function
        self.activation_function = self.get_activation_function(activation_function)

    def forward(self, x_net):
        # Pass the input through each layer except the last
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
