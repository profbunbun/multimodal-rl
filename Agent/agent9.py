from collections import deque
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
random.seed(0)
T.autograd.set_detect_anomaly(True)
class DQN(nn.Module):

    def __init__(self, state_size,action_size):
        super(DQN, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        self.layer1 = nn.Linear(state_size, 32)
        # 128 out to 32
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 16)
        # add softmax
        self.layer4 = nn.Linear(16, action_size)

    
    def forward(self, x):
        x=x.float()
        
        x=x.to(self.device)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # x = F.softmax(self.layer4(x))
        # add softmax for output
        return self.layer4(x)
        # return x



 
class Agent9:
    def __init__(self,state_size,action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory= deque(maxlen=2000)
        
        self.gamma = 0.95
        self.epsilon = .998
        self.epsilon_max = .998
        self.decay = 0.995
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.state_size,self.action_size).to(self.device)
        # print(self.policy_net)
        # self.target_net = DQN(self.state_size,self.action_size).to(self.device)
        
             
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
        
        for state,action,reward,new_state,done in minibatch:
          
            
            if not done:
                # estimated_target_DQN=self.policy_net(next_state).to(self.device)
                # estimated_target= (reward + self.gamma * T.argmax(estimated_target_DQN))
                reward= reward.float()
                reward=reward.to(self.device)
                
                #get estimated rewards for next step
                est=self.policy_net(new_state).to(self.device)#rename these
                # replace estimated reward for action with reward plus gamma version of the estimate
                adjusted_reward = (reward + self.gamma * T.argmax(est))# max not argmax
                # update estimated actin reward 
                est[action]=adjusted_reward
                
                est=est.float()
                # get the current policy ----- use label 
                policy=self.policy_net(state).to(self.device)
                # update with actual
                policy[action]=adjusted_reward
                t=policy
                t=t.float()
                
            else:
                # if its done, ther is no prediction
                est=self.policy_net(new_state).to(self.device)
                
                est=est.float()
                reward= reward.float()
                reward=reward.to(self.device)
                policy=self.policy_net(state).to(self.device)
                # so update the policy with the actual reward
                policy[action]=reward
                t=policy
                t=t.float()
               
            
            
            # confirm this is how to train in pytorch
            # self.loss =  nn.MSELoss()
            # self.loss = nn.CrossEntropyLoss()
            # self.loss = nn.BCEWithLogitsLoss()
            self.loss = nn.MSELoss()
            # self.loss = nn.L1Loss()
            # self.loss = nn.BCELoss()
            
            
            self.optimizer=optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)
            
            output=self.loss(est,t)
            
            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()
            
            T.cuda.empty_cache()
            
            params = list(self.policy_net.parameters())
            # print(len(params))
            # print(params[0].size())
    
    
    def epsilon_decay(self):   
        if self.epsilon > self.epsilon_min:
            
            self.epsilon *= self.decay
        pass
    def epsilon_decay_2(self,episode,episodes):   
        if self.epsilon > self.epsilon_min:
            
            if (episode > (5/10 * episodes) ):           
                self.epsilon *= self.decay
            
            else:
                self.epsilon= self.epsilon_max-1.01**(10*episode-((5.4/10 * episodes)*10))
                
        pass
    