import os
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn import DQN
from . import exploration, memory
import wandb


class Agent:
    """
    Agent class representing a reinforcement learning agent with a Deep Q-Network.

    :param int state_size: Size of the input state space.
    :param int action_size: Size of the output action space.
    :param str path: Path for saving and loading models.
    :param wandb_run: Weight and Biases run object for logging.
    :param float learning_rate: Learning rate for the optimizer.
    :param float gamma: Discount factor for future rewards.
    :param float epsilon_decay: Decay rate for exploration probability.
    :param float epsilon_max: Maximum exploration probability.
    :param float epsilon_min: Minimum exploration probability.
    :param int memory_size: Size of the replay memory.
    :param list layer_sizes: Sizes of the hidden layers in the DQN.
    :param str activation: Activation function to use in the DQN.
    :param int batch_size: Batch size for training.
    :param float soft_update_factor: Factor for soft updating the target network.
    """
    def __init__(self, state_size, 
                 action_size, 
                 path,

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

            self.policy_net = DQN(state_size, action_size).to(self.device)
            self.target_net = DQN(state_size, action_size).to(self.device)


            self.criterion = nn.HuberLoss()

            self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                        lr=self.learning_rate, momentum=0.9)

            self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)
            self.memory = memory.Memory(self.memory_size)

    def remember(self, state, action, reward, next_state, done):
        """
        Remember a new experience.

        :param list state: The current state of the environment.
        :param int action: The action taken.
        :param float reward: The reward received.
        :param list next_state: The next state of the environment.
        :param bool done: Whether the episode is finished.
        """
        
        self.memory.remember(state, action, reward, next_state, done)


    def choose_action(self, state, options):
        """
        Choose an action based on the current state and available options using an epsilon-greedy strategy.

        :param list state: The current state of the environment.
        :param list options: Available action options.
        :return: A tuple containing the chosen action, its index, a validity flag, and Q values.
        :rtype: tuple
        """

        action, index, valid, q_values = self.exploration_strategy.choose_action(state, options)
        return action, index, valid

    def replay(self, batch_size):
        """
        Replay experiences from memory and train the network.

        :param int batch_size: Size of the batch to replay.
        :return: None
        """
        minibatch = self.memory.replay_batch(batch_size)
        if len(minibatch) == 0:
            return None

        states, actions, next_states, rewards, dones = zip(*minibatch)

        states = [torch.tensor(s, device=self.device, dtype=torch.float32) for s in states]
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        self.perform_training_step(states, actions, next_states, rewards, dones)



    def perform_training_step(self, state, action, next_state, reward, done):
        """
        Perform a training step using the given experience batch.

        :param torch.Tensor state: Tensor of states.
        :param torch.Tensor action: Tensor of actions.
        :param torch.Tensor reward: Tensor of rewards.
        :param torch.Tensor next_state: Tensor of next states.
        :param torch.Tensor done: Tensor of done flags.
        """
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

        q_values = self.policy_net(state)



        current_q_values = torch.gather(q_values, 1, action)
        next_q_values = self.target_net(next_state).detach()
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)
        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
     


    def hard_update(self):
        """
        Hard update the target network's weights with the policy network's weights.
        """
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)

    def generate_config_id(self, *args):
        """
        Generate a unique hash for the given configuration.

        :param args: Configuration parameters.
        :return: A unique configuration identifier.
        :rtype: str
        """
        config_str = '_'.join(map(str, args))
        return config_str

    def save_model(self, episode_num):
        """
        Save the current model state.

        :param int episode_num: Current episode number.
        """
        filename = f"model"
        filename += f"_ep{episode_num}"
        filename += ".pt"

        temp_model_path = os.path.join(self.path, filename)
        torch.save(self.policy_net.state_dict(), temp_model_path)



    def load_model(self, ep_num):
        """
        Load the model state dictionary from a WandB artifact.

        :param str artifact_name: The name of the artifact to load, defaults to None.
        """

        filename = f"model"
        filename += f"_ep{ep_num}"
        filename += ".pt"

        model_path = os.path.join(self.path, filename)

        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.to(self.device)


    def decay(self):
        """
        Decay the exploration probability.
        """
        self.exploration_strategy.update_epsilon()
    
    def get_epsilon(self):
        """
        Get the current exploration probability.

        :return: Current exploration probability.
        :rtype: float
        """
        return self.exploration_strategy.epsilon
    
   

    def set_hyperparameters(self, learning_rate, gamma, epsilon_decay):
        """
        Set new values for the agent's hyperparameters.

        :param float learning_rate: New learning rate.
        :param float gamma: New discount factor.
        :param float epsilon_decay: New epsilon decay rate.
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate, momentum=0.9)
        self.exploration_strategy = exploration.Explorer(self.policy_net, self.epsilon_max, self.epsilon_decay, self.epsilon_min)

    def get_exploration_stats(self):
        """
        Calculate and return the exploration vs exploitation statistics.

        :return: Tuple containing exploration ratio and exploitation ratio.
        :rtype: tuple
        """
        return self.exploration_strategy.get_exploration_stats()