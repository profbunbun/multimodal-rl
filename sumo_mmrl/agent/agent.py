import os
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn import DQN
from . import exploration, memory
import json 
import hashlib
import wandb


class Agent:
    def __init__(self, state_size, 
                 action_size, 
                 path,
                 wandb_run,
                 learning_rate=None, 
                 gamma=None, 
                 epsilon_decay=None, 
                 epsilon_max=None, 
                 epsilon_min=None, 
                 memory_size=None, 
                 layer_sizes=None, 
                 activation=None, 
                 batch_size=None,
                 soft_update_factor=None,
                 ):
            self.wandb_run = wandb_run
            self.path = path
            self.direction_choices = ['r', 's', 'l', 't']

            self.memory_size = memory_size 
            self.gamma = gamma 
            self.learning_rate = learning_rate 
            self.epsilon_decay = epsilon_decay 

            self.soft_update_factor = soft_update_factor
            self.batch_size = batch_size 
            self.epsilon_max = epsilon_max 
            self.epsilon_min = epsilon_min 
            self.soft_update_factor = soft_update_factor 
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.policy_net = DQN(state_size, action_size, layer_sizes, activation).to(self.device)
            self.target_net = DQN(state_size, action_size, layer_sizes, activation).to(self.device)


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

        self.perform_training_step(states, actions, rewards, next_states, dones)



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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        if self.wandb_run:
            self.wandb_run.log({
            "Loss": loss.item(),
            "Max Gradient Norm": max_grad_norm
            })
        if self.wandb_run:
            for name, param in self.policy_net.named_parameters():
                self.wandb_run.log({f"Policy Gradients/{name}":wandb.Histogram(param.grad.cpu().detach().numpy())})






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
        return config_str

    def save_model(self, episode_num):
        # Define the filename for the model
        filename = f"model"
        
        filename += f"_ep{episode_num}"
        filename += ".pt"


        # Save model to a temporary file
        temp_model_path = os.path.join(wandb.run.dir, filename)
        torch.save(self.policy_net.state_dict(), temp_model_path)

        # Create a new artifact
        artifact = wandb.Artifact(name=f"model-{episode_num}", type='model', description="Trained model")

        # Add the file to the artifact's contents
        artifact.add_file(temp_model_path)

        # Log the artifact to W&B and associate it with the current run
        wandb.run.log_artifact(artifact)

        # Optionally, remove the temporary file if you don't want it saved locally
        os.remove(temp_model_path)


    def load_model(self, artifact_name=None):
        """
        Load the model state dictionary from a WandB artifact.

        Parameters:
        - artifact_name: The name of the artifact to load. If not provided, it tries to load
                         using the configuration ID.
        """

        # Construct the artifact name if not provided
        artifact_name = artifact_name or f"model-{self.config_id}:latest"

        # Use the WandB API to retrieve the artifact
        artifact = self.wandb_run.use_artifact(artifact_name)

        # Download the artifact's files and get the model file path
        artifact_dir = artifact.download()

        # In most cases, the file is saved as "model.pt" or a similar name. Update this accordingly.
        model_file = 'model.pt'

        # Construct the full path to the model file
        model_path = os.path.join(artifact_dir, model_file)

        # Load the model state
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.to(self.device)


    def decay(self):
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        return self.exploration_strategy.epsilon
    
   

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

    def get_exploration_stats(self):
        """Calculate the exploration vs exploitation statistics."""
        return self.exploration_strategy.get_exploration_stats()