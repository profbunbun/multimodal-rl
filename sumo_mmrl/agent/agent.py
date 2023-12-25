import os
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn import DQN
from . import exploration, memory
import json 

PATH = "/Models/model.pt"

class Agent:
    def __init__(self, state_size, action_size, path):
        self.path = path
        self.direction_choices = ['r', 's', 'l', 't']
        with open("config.json", "r") as config_file:
            config = json.load(config_file)

        self.memory_size = config["training_settings"]["memory_size"]
        self.gamma = config["hyperparameters"]["gamma"]
        self.learning_rate = config["hyperparameters"]["learning_rate"]
        self.soft_update_factor = config["hyperparameters"]["soft_update_factor"]
        self.batch_size = config["training_settings"]["batch_size"]
        self.epsilon_max = config["hyperparameters"]["epsilon_max"]
        self.epsilon_min = config["hyperparameters"]["epsilon_min"]
        self.epsilon_decay = config["hyperparameters"]["epsilon_decay"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)

        if os.path.exists(path + PATH):
            self.target_net.load_state_dict(torch.load(path + PATH))


        self.criterion = nn.HuberLoss()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                       lr=self.learning_rate, momentum=0.9)

        self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)
        self.memory = memory.Memory(self.memory_size)

    def remember(self, state, action, reward, next_state, done):
        
        self.memory.remember(state, action, reward, next_state, done)


    def choose_action(self, state, options):

     
       

        action, index, valid, q_values = self.exploration_strategy.choose_action(state, options)
        return action, index, valid, q_values

    def replay(self, batch_size):
        minibatch = self.memory.replay_batch(batch_size)
        if len(minibatch) == 0:
            return None

        # Unpack the minibatch into separate lists
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists to tensors
        states = [torch.tensor(s, device=self.device, dtype=torch.float32) for s in states]
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        loss_item = self.perform_training_step(states, actions, rewards, next_states, dones)
        
        return loss_item

    def perform_training_step(self, state, action, reward, next_state, done):
        # Check if the inputs are already tensors. If not, convert them.
        if not isinstance(state, torch.Tensor):
            state = torch.stack(state).to(self.device).float()
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.stack(next_state).to(self.device).float()
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device).long()
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device).float()
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, device=self.device).float()
        
        
        # Ensure the tensors are in the correct shape
        action = action.unsqueeze(-1) if action.dim() == 1 else action
        reward = reward.unsqueeze(-1) if reward.dim() == 1 else reward
        done = done.unsqueeze(-1) if done.dim() == 1 else done

        # self.policy_net.train()
        # self.target_net.eval()
        # print("state", state.shape)


        # Get current Q values from the policy network
        current_q_values = self.policy_net(state).gather(1, action)

        # Get next Q values from the target network
        next_q_values = self.target_net(next_state).detach()
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        # Compute the expected Q values
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)

        # Compute loss
        loss = self.criterion(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        max_grad_norm = max(p.grad.data.norm(2).item() for p in self.policy_net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), .5)
        self.optimizer.step()

        return loss.item(), max_grad_norm



    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.soft_update_factor + target_net_state_dict[key] * (1 - self.soft_update_factor)
        self.target_net.load_state_dict(target_net_state_dict)

    def hard_update(self):
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)

    def save(self):
        torch.save(self.policy_net.state_dict(), self.path + PATH)


    def decay(self):
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        return self.exploration_strategy.epsilon
