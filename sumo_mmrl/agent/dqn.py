import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer2b = nn.Linear(64, 32)
        self.layer2c = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, action_size)

        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)
        # self.dropout3 = nn.Dropout(p=0.1)

        # self.apply(self.init_weights)

    def forward(self, x_net):
        x_net = F.relu(self.layer1(x_net))
        # x_net = self.dropout1(x_net)
        x_net = F.relu(self.layer2(x_net))
        # x_net = self.dropout2(x_net)
        x_net = F.relu(self.layer2b(x_net))
        # x_net = self.dropout3(x_net)
        x_net = F.relu(self.layer2c(x_net))
        x_net = self.layer3(x_net)
        # x_net = F.log_softmax(x_net, dim=1) 
        return x_net
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)
