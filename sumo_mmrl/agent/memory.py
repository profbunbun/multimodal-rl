import random


class Memory:
    '''Circular buffer memory using a list'''
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = [None] * memory_size  # Preallocate memory
        self.position = 0  # Current position in the circular buffer

    def remember(self, state, action, reward, next_state, done):
        '''Add memory to the circular buffer'''
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.memory_size  # Wrap-around

    def replay_batch(self, batch_size):
        '''minibatch of randomly selected memories using indices'''
        # Calculate the current number of items in the memory
        current_memory_length = self.position if None not in self.memory else self.memory.index(None)
        
        # Ensure we don't sample more items than we have in memory
        actual_batch_size = min(batch_size, current_memory_length)
        
        # Select a batch of random indices from the memory
        indices = random.sample(range(current_memory_length), actual_batch_size)
        
        # Create the batch using the random indices
        batch = [self.memory[i] for i in indices]
        return batch

    def __len__(self):
        '''Return the current size of internal memory'''
        # Since the buffer is circular, we return the memory size when it's full
        return self.memory_size if None not in self.memory else self.memory.index(None)
