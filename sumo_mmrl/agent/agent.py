import os
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn import DQN
from . import exploration, memory
import json 
import hashlib





# PATH = "/logger/model.pt"

class Agent:
    def __init__(self, state_size, action_size, path, learning_rate=None, gamma=None, epsilon_decay=None, epsilon_max=None, epsilon_min=None, memory_size=None, n_layers=None, layer_size=None, activation=None, batch_size=None):
        self.logger = None
        self.path = path
        self.direction_choices = ['r', 's', 'l', 't']
        with open("config.json", "r") as config_file:
            config = json.load(config_file)

        self.memory_size = memory_size if memory_size is not None else config["training_settings"]["memory_size"]
        self.gamma = gamma if gamma is not None else config["hyperparameters"]["gamma"]
        self.learning_rate = learning_rate if learning_rate is not None else config["hyperparameters"]["learning_rate"]
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config["hyperparameters"]["epsilon_decay"]

        self.soft_update_factor = config["hyperparameters"]["soft_update_factor"]
        self.batch_size = batch_size if batch_size is not None else config["training_settings"]["batch_size"]
        self.epsilon_max = epsilon_max if epsilon_max is not None else config["hyperparameters"]["epsilon_max"]
        self.epsilon_min = epsilon_min if epsilon_min is not None else config["hyperparameters"]["epsilon_min"]

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size, n_layers, layer_size, activation).to(self.device)
        self.target_net = DQN(state_size, action_size, n_layers, layer_size, activation).to(self.device)

        # if os.path.exists(path + PATH):
        #     self.target_net.load_state_dict(torch.load(path + PATH))
        self.config_id = self.generate_config_id(learning_rate, gamma, epsilon_decay, epsilon_max, epsilon_min, memory_size, n_layers, layer_size, activation, batch_size)
        self.load_model()

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

    def replay(self, batch_size, current_episode, current_step):
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

        loss_item, max_grad_norm, current_q_values, expected_q_values = self.perform_training_step(states, actions, rewards, next_states, dones)
        # if self.logger:
        #     training_data = {
        #         'episode': current_episode,  # Current episode
        #         'agent_step': current_step,  # Current step of the agent
        #         'batch_size': batch_size,  # Size of the batch
        #         'loss': loss_item,  # Loss from the current training step
        #         'q_values': [q.tolist() for q in current_q_values],  # Q values from the policy network
        #         'target_q_values': [q.tolist() for q in expected_q_values],  # Target Q values for loss calculation
        #         'epsilon': self.get_epsilon(),  # Current epsilon for exploration
        #         'learning_rate': self.learning_rate,  # Current learning rate
        #         'gradient_norms': [p.grad.data.norm(2).item() for p in self.policy_net.parameters() if p.grad is not None],  # Gradient norms
        #         'max_gradient_norm': max_grad_norm,  # Maximum gradient norm
        #         'replay_memory_size': len(self.memory)  # Size of the replay memory
        #     }
        #     self.logger.log_training(training_data)
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

        return loss.item(), max_grad_norm, current_q_values, expected_q_values



    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.soft_update_factor + target_net_state_dict[key] * (1 - self.soft_update_factor)
        self.target_net.load_state_dict(target_net_state_dict)

    def hard_update(self):
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)

    def generate_config_id(self, *args):
        """Generate a unique hash for the given configuration."""
        config_str = '_'.join(map(str, args))
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_model(self, episode_num=None):
        """Save the model uniquely based on its configuration and episode number."""
        filename = f"model_{self.config_id}"
        if episode_num is not None:
            filename += f"_ep{episode_num}"
        filename += ".pt"
        save_path = os.path.join(self.path, filename)
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self):
        """Load the model if a file matching the configuration exists."""
        filename = f"model_{self.config_id}.pt"
        load_path = os.path.join(self.path, filename)
        if os.path.exists(load_path):
            self.policy_net.load_state_dict(torch.load(load_path, map_location=self.device))
            self.policy_net.to(self.device)
            print(f"Model loaded from {load_path}")


    def decay(self):
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        return self.exploration_strategy.epsilon
    
    # def get_model_info(self):
    #     """Returns detailed information about the policy network."""
    #     model = DQN(12, 4)
    #     model.load_state_dict(torch.load(self.path + PATH))
    #     state_info = self.get_model_state_dict_as_string(model)
    #     opt_info = self.get_optimizer_state_dict_as_string(self.optimizer)
        

        return state_info, opt_info

    def get_model_state_dict_as_string(self, model):
        model_info = "Model's state_dict:\n"
        for param_tensor in model.state_dict():
            model_info += f"{param_tensor}\t{model.state_dict()[param_tensor].size()}\n"
        return model_info

    def get_optimizer_state_dict_as_string(self, optimizer):
        optimizer_info = "Optimizer's state_dict:\n"
        for var_name in optimizer.state_dict():
            optimizer_info += f"{var_name}\t{optimizer.state_dict()[var_name]}\n"
        return optimizer_info
    
    def set_hyperparameters(self, learning_rate, gamma, epsilon_decay):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate, momentum=0.9)
        self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)
