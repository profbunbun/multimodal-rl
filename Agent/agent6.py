from collections import deque
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQN(nn.Module):

    def __init__(self, state_size,action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    
    def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      return self.layer3(x)



 
class Agent6:
    def __init__(self,state_size,action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory= deque(maxlen=20000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.decay = 0.995
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size,self.action_size).to(self.device)
        
             
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        
        if  np.random.rand() < self.epsilon:
            return np.random.randint(1,size=self.action_size)
        act_values = self.policy_net(state)
        return np.argmax(act_values[0])

    def replay(self,batch_size):
        
        minibatch=np.random.sample(self.memory,batch_size)
        
        for state,action,reward,next_state,done in minibatch:
            target = reward
            if not done:
                target= (reward+self.gamma *np.amax(self.policy_net(next_state)[0]))
            target_f=self.policy_net(state)
            target_f[0][action]=target
            criterion = nn.SmoothL1Loss()
            loss = criterion(target_f,target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay