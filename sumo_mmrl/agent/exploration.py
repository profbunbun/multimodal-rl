import numpy as np
import torch as T

class Explorer:

    def __init__(self, policy, epsilon_max=1, decay_rate=0.999, epsilon_min=0.1):
   
        self.epsilon = epsilon_max
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min

        self.direction_choices = ['R', 'r', 's', 'L', 'l', 't']
        self.policy_net = policy
        self.explore_count = 0
        self.exploit_count = 0
        # np.random.seed(0)
        np.random.seed(1)
        # np.random.seed(9)
        # np.random.seed(20)
        # np.random.seed(42)
        # np.random.seed(66)
        # np.random.seed(616)
        # np.random.seed(9000)
        # np.random.seed(8675309)
        self.last_reward = None

    def explore(self):
   
        action = np.random.choice(self.direction_choices)
        self.explore_count += 1
        return action

    def exploit(self, state):
        
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)

        device = next(self.policy_net.parameters()).device
        state = state.to(device)

        state = state.unsqueeze(0)

        self.policy_net.eval()
        with T.no_grad():
            act_values = self.policy_net(state)
        self.policy_net.train()

        index_choice = T.argmax(act_values)
        num_of_choices = len(self.direction_choices)

        if (index_choice + 1) <= num_of_choices:
            action = self.direction_choices[index_choice]
        else:
            action  = None
        self.exploit_count += 1
        return action, index_choice

    def choose_action(self, state, options):

        randy = np.random.rand()
        if randy < self.epsilon:
            action= self.explore()
        else:
            action, index = self.exploit(state)

        if action == None:
            valid = 0

        else:
            valid = int(action in options)
            index = self.direction_choices.index(action)

        
        return action, index, valid

    def update_epsilon(self):

        if self.epsilon < self.epsilon_min:
            self.epsilon = 0.0
        else:
            self.epsilon = self.epsilon * self.decay_rate
