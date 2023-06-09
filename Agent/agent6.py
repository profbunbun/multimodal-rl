from collections import deque
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
random.seed(0)

class DQN(nn.Module):

    def __init__(self, state_size,action_size):
        super(DQN, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    
    def forward(self, x):
        x=x.float()
        x=x.to(self.device)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



 
class Agent6:
    def __init__(self,state_size,action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory= deque(maxlen=10000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.decay = 0.995
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size,self.action_size).to(self.device)
        self.target_net = DQN(self.state_size,self.action_size).to(self.device)
        
             
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        rando=np.random.rand()
        if  rando < self.epsilon:
            # print(self.epsilon)
            act=np.random.randint(0,high=self.action_size-1)
            return act
        act_values = self.policy_net(state)
        act_values = act_values.detach().cpu().numpy()
        act=np.argmax(act_values)
        return act

    def replay(self,batch_size):
        T.cuda.empty_cache()
        minibatch=random.sample(self.memory,batch_size)
        
        for state,action,reward,next_state,done in minibatch:
            target = reward
            if not done:
                target_net=self.target_net(next_state).detach().cpu().numpy()
                target= (reward + self.gamma * np.amax(target_net))
            policy=self.policy_net(state)
            # policy=self.policy_net(state).detach().cpu().numpy()
            target_f=policy
            # target_f[action-1]=target.clone().detach().requires_grad_(True)
            t=target.clone().detach().requires_grad_(True)
            t=t.float()
            t=t.to(self.device)
            t=t.squeeze(-1)
        
            tf=target_f[action-1]
            self.loss =  nn.MSELoss()
            self.optimizer=optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)
            output=self.loss(tf,t)
            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()
            
            T.cuda.empty_cache()
    
    def epsilon_decay(self):   
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        pass
    