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
        # 128 out to 32
        self.layer2 = nn.Linear(128, 128)
        
        self.layer3 = nn.Linear(128, action_size)

    
    def forward(self, x):
        x=x.float()
        
        x=x.to(self.device)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # add softmax for output
        return self.layer3(x)



 
class Agent7:
    def __init__(self,state_size,action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory= deque(maxlen=10000)
        
        self.gamma = 0.95
        self.epsilon = .9999
        self.epsilon_max = .9999
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
        # q-val
        act=T.argmax(act_values)
        return act


    def replay(self,batch_size):
        T.cuda.empty_cache()
        minibatch=random.sample(self.memory,batch_size)
        
        for state,action,reward,next_state,done in minibatch:
            # update targets to same vector size
            reward= reward.float()
            reward=reward.to(self.device)
            target = reward
            
            if not done:
                target_net=self.target_net(next_state).to(self.device)
                target= (reward + self.gamma * T.argmax(target_net))
            
            policy=self.policy_net(state)
            target_f = policy
            t=target
            t=t.float()
            t=t.squeeze(-1)
        
            tf=target_f[action-1]
            # confirm this is how to train in pytorch
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
    def epsilon_decay_2(self,episode,episodes):   
        if self.epsilon > self.epsilon_min:
            
            if (episode > (9/10 * episodes) ):           
                self.epsilon *= self.decay
            
            else:
                self.epsilon= self.epsilon_max-1.01**(10*episode-((9.1/10 * episodes)*10))
                
        pass
    