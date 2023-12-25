import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        pass

    def find_first_valid_index(self, data):
        """Find the index of the first non-None value in a list."""
        for i, value in enumerate(data):
            if value is not None:
                return i
        return -1  # Return -1 if no valid index is found

    def plot_learning(self, x, smoothed_rewards, epsilons, filename):
        fig, ax = plt.subplots(figsize=(15, 10))

        # Shared X-axis for all plots
        shared_x = np.arange(1, len(x) + 1)

        # Plot Smoothed Rewards and Epsilons on the same plot
        ax.plot(shared_x, smoothed_rewards, label="Smoothed Reward per Episode", color='C0')
        ax.plot(shared_x, epsilons, label="Epsilon Decay", color='C1')
        ax.set_title('Episode Rewards & Epsilon Decay')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
