import csv
import os
import json
from datetime import datetime

class Logger:
    def __init__(self, base_log_dir, config_path):
        # Read hyperparameters and training settings from the config file
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        hyperparameters = config.get("hyperparameters", {})
        training_settings = config.get("training_settings", {})

        # Extract relevant values
        episodes = training_settings.get("episodes", "")
        learning_rate = self.format_value(hyperparameters.get("learning_rate", ""))
        decay = self.format_value(hyperparameters.get("epsilon_decay", ""))

        # Construct directory name based on the formatted values
        dir_name = f"eps{episodes}_lr{learning_rate}_d{decay}"
        log_dir = os.path.join(base_log_dir, dir_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Define filenames within the constructed directory
        self.step_filename = os.path.join(log_dir, 'step_log.csv')
        self.episode_filename = os.path.join(log_dir, 'episode_log.csv')
        
        # Fields for step and episode logs
        self.step_fields = [
             'episode', 'step', 'reward', 'epsilon', 'vehicle_location_edge_id',
            'destination_edge_id', 'out_lanes', 'action_chosen', 'best_choice', 'q_values','stage', 'done'
        ]
        self.episode_fields = [
             'episode', 'epsilon', 'cumulative_reward', 'total_steps', 'action_distribution',
            'action_probabilities', 'td_error', 'life'
        ]
        
        # Initialize log files
        self.init_step_log()
        self.init_episode_log()

    def format_value(self, value):
        """Format the value by removing unnecessary leading zeros and preserving decimals."""
        if isinstance(value, float):
            return f"{value:.10f}".rstrip('0').rstrip('.')  # Adjust precision as needed
        return str(value).lstrip('0')

    def init_step_log(self):
        """Initialize the step log file and write the header."""
        with open(self.step_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.step_fields)
            writer.writeheader()

    def init_episode_log(self):
        """Initialize the episode log file and write the header."""
        with open(self.episode_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.episode_fields)
            writer.writeheader()

    def log_step(self, data):
        """Log the provided step data to the CSV file."""
        with open(self.step_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.step_fields)
            writer.writerow(data)

    def log_episode(self, data):
        """Log the provided episode aggregate data to the CSV file."""
        with open(self.episode_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.episode_fields)
            writer.writerow(data)
