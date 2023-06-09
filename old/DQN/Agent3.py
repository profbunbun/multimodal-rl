import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from DQN.DQNetwork import DQN
from DQN.ReplayMemory import ReplayMemory, Transition
import math 
import random

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        
        x=x.to(self.device)
        # x.reshape([2,64])
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent3:
    
    def __init__(self,n_observations,n_actions,gamma,epsilon,eps_max,eps_end, 
                 eps_decay,tau,learning_rate,batch_size) -> None:
        
        
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
       
        self.gamma = gamma
        self.gamma = T.tensor([self.gamma]).to(self.device)
        self.eps_max = eps_max
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.epsilon=epsilon
        
        
        
        
        
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.policy_net=self.policy_net.double()
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net=self.target_net.double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.explore=0
        self.exploit=0
        pass
    
    
    
    
    
    
    
    





    def select_action(self,state,steps_done,episode):
        
        state=state.to(self.device)
        # print("state: "+str(state.shape))
        self.steps_done=steps_done
        sample = random.random()
        self.epsilon = self.eps_end + (self.eps_max - self.eps_end) * \
            math.exp(-1. * episode/self.eps_decay)
        # print("epsilon: "+str(self.epsilon)+" sample: "+str(sample)+"exploit: "+str(self.exploit)+" explore: "+str(self.explore))
        
       
     
        if sample > self.epsilon:
            self.exploit+=1
            with T.no_grad():
                pnet=   self.policy_net(state).max(0)[1].view(1, 1)
                
            return pnet ,self.epsilon,self.exploit,self.explore
            
        else:
            self.explore+=1
            return T.tensor([[random.randint(0,2)]], device= self.device, dtype=T.long),self.epsilon,self.exploit,self.explore
        
        
        
        
        
        
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in batch.next_state
                                                    if s is not None])
        
        non_final_next_states_reshaped=non_final_next_states.reshape([self.batch_size,self.n_observations])
        non_final_next_states=non_final_next_states_reshaped
        
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)
        reward_batch=reward_batch.to(self.device)

        state_batch_reshaped=state_batch.reshape([self.batch_size,self.n_observations])
        state_batch=state_batch_reshaped
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        next_state_values = T.zeros(self.batch_size, device=self.device)
        
        with T.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].float()
            next_state_values.to(self.device)
        # Compute the expected Q values
        # self.gamma=self.gamma.to(self.device)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        T.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    



