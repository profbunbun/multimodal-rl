from collections import deque
import os
import random
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
random.seed(0)
T.autograd.set_detect_anomaly(True)

STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

PATH="Models/model.pt"
class DQN(nn.Module):

    def __init__(self, state_size,action_size):
        super(DQN, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        self.layer1 = nn.Linear(state_size, 16)
        nn.init.kaiming_normal_(self.layer1.weight,mode="fan_out")
        # nn.init.zeros_(self.layer1.weight)
        # nn.init.zeros_(self.layer1.bias)
        self.layer2 = nn.Linear(16, 32)
        nn.init.kaiming_normal_(self.layer1.weight,mode="fan_out")
        # nn.init.zeros_(self.layer2.weight)
        # nn.init.zeros_(self.layer2.bias)
        self.layer3 = nn.Linear(32, 16)
        nn.init.kaiming_normal_(self.layer1.weight,mode="fan_out")
        
        # nn.init.zeros_(self.layer3.weight)
        # nn.init.zeros_(self.layer3.bias)
        self.layer4 = nn.Linear(16, action_size)
        nn.init.kaiming_normal_(self.layer1.weight)
        # nn.init.zeros_(self.layer4.weight)
        # nn.init.zeros_(self.layer4.bias)

    def forward(self, x):
        x=x.float()
        x=x.to(self.device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

 
class Agent:
    def __init__(self,state_size,action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.direction_choices = [STRAIGHT, TURN_AROUND,  SLIGHT_RIGHT, RIGHT, SLIGHT_LEFT, LEFT]
        self.memory= deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = .997
        self.epsilon_max = .9997
        self.decay = 0.99
        self.epsilon_min=0.01
        self.learning_rate=0.001
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        if os.path.exists(PATH):
            self.policy_net = DQN(self.state_size,self.action_size).to(self.device)
            self.policy_net.load_state_dict(T.load(PATH))
            self.policy_net.eval()
        else:
            self.policy_net = DQN(self.state_size,self.action_size).to(self.device)

    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        pass
        
    def explore(self,available_choices):
        
        return
    
    def exploit(self):
        return
        
# make choice function
    def act(self,state, options):
        
     
        rando=np.random.rand()
        if  rando < self.epsilon:
            
            available_choices=list(options.keys())
            
            act=np.random.choice(available_choices)
            
            return act
            
        else:
            act_values = self.policy_net(state)
            # q-val
            # act=T.argmax(act_values)
            act=self.direction_choices[T.argmax(act_values)]
            return act
        
        
 
# Train the model
    def replay(self,batch_size):
        # T.cuda.empty_cache()
        minibatch=random.sample(self.memory,batch_size)
        
        for state,action,reward,new_state,done in minibatch:
          
            reward= reward.float()
            reward=reward.to(self.device)
            
            if not done:
               
                new_state_policy=self.policy_net(new_state).to(self.device)
                
                adjusted_reward = (reward + self.gamma * T.max(new_state_policy))
                
                output=self.policy_net(state).to(self.device)
               
                
                target=output.detach().clone()
                target[action]=adjusted_reward
                out_mask=out_mask.detach().clone()
                
               
                for i in enumerate(target):
                    if out_mask[i[0]]==0:
                        target[i[0]]=-1000
                        
                
                target=target.to(self.device)
               
                
            else: 
                
                
                output=self.policy_net(state).to(self.device)
                target=output.detach().clone()
                target[action]=reward
                out_mask=out_mask.detach().clone()
                
                
                for i in enumerate(target):
                    if out_mask[i[0]]==0:
                        target[i[0]]=-1000
                
                target=target.to(self.device)
               
            
               
               
            
                # loss function
                # loss =  nn.MSELoss()
                # loss = nn.L1Loss()
                loss = nn.HuberLoss()
                
                # optimize parameters
                optimizer=optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)
                
                out=loss(output,target)
                
                # out=self.loss(output,target)
                optimizer.zero_grad()
                # out.backward()
                out.backward(retain_graph=True)
                optimizer.step()
                # T.cuda.empty_cache()
                
                T.save(self.policy_net.state_dict(),PATH)
                # return loss.item()
            

    
    
    # trying differnt epsilon decay
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
        else:
            self.epsilon=self.epsilon       
        pass
    def epsilon_decay_3(self,episode,episodes):   
        if self.epsilon > self.epsilon_min:
            episode+=1
            
            self.epsilon= (1/9.5)*math.log(((-episode)+episodes+1))
            # self.epsilon_max-1.01**(10*episode-((4.4/10 * episodes)*10))
        else:
            self.epsilon=self.epsilon       
        pass
    