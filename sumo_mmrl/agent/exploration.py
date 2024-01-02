import numpy as np
import torch as T

class Explorer:
    """
    Explorer class for managing exploration and exploitation in a reinforcement learning agent.

    :param torch.nn.Module policy: The policy network used for exploitation.
    :param float epsilon_max: The maximum epsilon for exploration.
    :param float decay_rate: The rate at which epsilon decays.
    :param float epsilon_min: The minimum epsilon for exploration.
    """

    def __init__(self, policy, epsilon_max=1, decay_rate=0.999, epsilon_min=0.1):
        """
        Initialize the explorer.

        Args:
            policy: The policy network used for exploitation.
            epsilon_max (float): The maximum epsilon for exploration.
            decay_rate (float): The rate at which epsilon decays.
            epsilon_min (float): The minimum epsilon for exploration.
        """
        self.epsilon = epsilon_max
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min

        self.direction_choices = ['r', 's', 'l', 't']
        self.policy_net = policy
        self.explore_count = 0
        self.exploit_count = 0
        np.random.seed(0)
        self.last_reward = None

    def explore(self):
        """
        Randomly select an action.

        :return: The selected action.
        :rtype: str
        """
        action = np.random.choice(self.direction_choices)
        self.explore_count += 1
        return action

    def exploit(self, state):
        """
        Select action based on the policy network.

        :param list state: The current state in an appropriate format (e.g., list, array, tensor).
        :return: The chosen action based on the policy network's output and the action values.
        :rtype: tuple
        """
        # Convert state to tensor if it's not already and ensure it's on the correct device
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)

        # Infer the device from the policy network's parameters
        device = next(self.policy_net.parameters()).device
        state = state.to(device)

        # Unsqueeze the tensor to add a batch dimension, necessary for batch normalization
        state = state.unsqueeze(0)

        # Use the network in evaluation mode for single-sample inference
        # self.policy_net.eval()
        # with T.no_grad():
        act_values = self.policy_net(state)
        # self.policy_net.train()  # Revert to training mode

        # Choose the action with the highest value
        action = self.direction_choices[T.argmax(act_values, dim=1)]
        self.exploit_count += 1
        return action, act_values

    def choose_action(self, state, options):
        """
        Choose an action based on epsilon-greedy strategy.

        :param list state: The current state.
        :param list options: Available actions.
        :return: The chosen action, its index, and a validity flag.
        :rtype: tuple
        """
        q_values = None
        randy = np.random.rand()
        if randy < self.epsilon:
            action = self.explore()
        else:
            action, q_values = self.exploit(state)

        valid = int(action in options)
        return action, self.direction_choices.index(action), valid, q_values

    def update_epsilon(self):
        """
        Update epsilon value using exponential decay.
        """
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)

    def get_exploration_stats(self):
        """
        Calculate the exploration vs exploitation statistics.

        :return: Tuple containing explore ratio and exploit ratio.
        :rtype: tuple
        """
        total = self.explore_count + self.exploit_count
        if total > 0:
            explore_ratio = self.explore_count / total
            exploit_ratio = self.exploit_count / total
        else:
            explore_ratio = exploit_ratio = 0
        return explore_ratio, exploit_ratio

    def save_stats(self, filename):
        """
        Save the current exploration vs exploitation stats to a file.

        :param str filename: The name of the file to save the stats to.
        """
        explore_ratio, exploit_ratio = self.get_exploration_stats()
        with open(filename, 'a') as file:
            file.write(f"{explore_ratio}, {exploit_ratio}\n")

