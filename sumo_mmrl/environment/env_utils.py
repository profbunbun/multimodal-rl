import numpy as np
import matplotlib.pyplot as plt
import os

class Utils:
    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        :param x1: X-coordinate of the first point.
        :param y1: Y-coordinate of the first point.
        :param x2: X-coordinate of the second point.
        :param y2: Y-coordinate of the second point.
        :return: Manhattan distance between the points.
        """
        return abs(x1 - x2) + abs(y1 - y2)

    @staticmethod
    def smooth_data(data, window_size):
        """
        Smooth data using a simple moving average.

        :param data: The input data to smooth.
        :param window_size: The number of data points to consider for the moving average.
        :return: Smoothed data.
        """
        if not data:
            return []

        smoothed = []
        running_total = 0
        non_numeric_count = 0

        for i, value in enumerate(data):
            if value is not None:
                running_total += value
            else:
                non_numeric_count += 1

            if i >= window_size:
                if data[i - window_size] is not None:
                    running_total -= data[i - window_size]
                else:
                    non_numeric_count -= 1

            count = min(i + 1, window_size) - non_numeric_count
            if count > 0:
                smoothed.append(running_total / count)
            else:
                smoothed.append(None)

        if non_numeric_count > 0:
            print(f"Warning: Found {non_numeric_count} non-numeric values. Some smoothing results may be None.")

        return smoothed

    @staticmethod
    def plot_learning_curve(x, rewards, epsilons, file_name):
        """
        Plot the learning curve of the agent.

        :param x: Episodes or time steps.
        :param rewards: List of rewards per episode or time step.
        :param epsilons: List of epsilon values per episode or time step.
        :param file_name: Path to save the plot.
        """
        fig, ax1 = plt.subplots(figsize=(15, 10))

        # Shared X-axis for all plots
        shared_x = np.arange(1, len(x) + 1)

        # Plot Smoothed Rewards on the primary y-axis
        color = 'C0'
        ax1.plot(shared_x, rewards, label="Smoothed Reward per Episode", color=color)
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
        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def ensure_directory_exists(directory):
        """
        Ensure that a directory exists. If it doesn't, create it.

        :param directory: Path of the directory to check.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
