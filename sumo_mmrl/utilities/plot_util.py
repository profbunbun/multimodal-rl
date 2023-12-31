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
        fig, ax1 = plt.subplots(figsize=(15, 10))

        # Shared X-axis for all plots
        shared_x = np.arange(1, len(x) + 1)

        # Plot Smoothed Rewards on the primary y-axis
        color = 'C0'
        ax1.plot(shared_x, smoothed_rewards, label="Smoothed Reward per Episode", color=color)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Smoothed Reward', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a twin y-axis to plot Epsilon Decay
        ax2 = ax1.twinx()
        color = 'C1'
        ax2.plot(shared_x, epsilons, label="Epsilon Decay", color=color)
        ax2.set_ylabel('Epsilon', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_title('Episode Rewards & Epsilon Decay')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
