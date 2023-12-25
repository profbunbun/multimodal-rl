import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:  
            self.write = 0  

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):  
                leaf_idx = parent_idx
                break
            else:  
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]  

class Memory:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  
        self.max_priority = 1.0  

    def add(self, error, experience):
        priority = (error + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        self.tree.add(self.max_priority, experience)  

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total_priority() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -1)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update(self, idx, error):
        priority = (error + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        self.tree.update(idx, priority)

# Usage Example:
# memory = Memory(capacity=1000)
# for _ in range(100):
#     experience = ...  # Obtain an experience from the environment
#     error = ...  # Calculate the error from the learning algorithm
#     memory.add(error, experience)
# batch, idxs, is_weights = memory.sample(32)
# ...  # Perform learning using the batch
# for idx in idxs:
#     ...  # Update the priorities based on the learning
#     new_error = ...
#     memory.update(idx, new_error)
