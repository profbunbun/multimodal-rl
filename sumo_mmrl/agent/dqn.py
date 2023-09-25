
import torch as T
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
  
    def __init__(self, state_size, action_size):
        super().__init__()

        self.device = T.device(  # pylint: disable=E1101
            "cuda" if T.cuda.is_available() else "cpu"
        )
        self.layer1 = nn.Linear(state_size, 32)

        self.layer2 = nn.Linear(32, 8)

        self.layer3 = nn.Linear(8, action_size)

    def forward(self, x_net):

        x_net = T.Tensor(x_net)
        # x_net = x_net.to(self.device)
        x_net = F.relu(self.layer1(x_net))  # pylint: disable=E1102
        x_net = F.relu(self.layer2(x_net))  # pylint: disable=E1102
        return self.layer3(x_net)  # pylint: disable=E1102