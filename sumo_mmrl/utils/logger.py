import csv
import os
import json
from datetime import datetime
import torch

class Logger:
    def __init__(self, base_log_dir, config_path):
        # Read hyperparameters and training settings from the config file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        hyperparameters = self.config.get("hyperparameters", {})
        training_settings = self.config.get("training_settings", {})
        


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
         # Define a new filename for the training log within the constructed directory
        self.training_filename = os.path.join(log_dir, 'training_log.csv')
        self.config_filename = os.path.join(log_dir, 'config_log.csv')
        self.model_info_filename = os.path.join(log_dir, 'model_info.txt') 


        
        # Fields for the training log
        self.training_fields = [
            'episode',              # Tracked manually in the training loop
            'agent_step',           # Tracked manually in the training loop
            'batch_size',           # Determined by your training settings
            'loss',                 # Calculated during the training step in Agent
            'q_values',             # Output from the policy network for the current states
            'target_q_values',      # Calculated as part of the target for the loss function
            'epsilon',              # Current epsilon from the exploration strategy in Agent
            'learning_rate',        # Current learning rate (if using a scheduler or static value)
            'gradient_norms',       # Calculated post-backpropagation
            'max_gradient_norm',    # Also calculated post-backpropagation, especially if applying gradient clipping
            'replay_memory_size'    # The current size of the replay memory, obtainable from Memory class
        ]
        
        # Fields for step and episode logs
        self.step_fields = [
             'episode', 'step', 'reward', 'epsilon', 'vehicle_location_edge_id',
            'destination_edge_id', 'out_lanes', 'action_chosen', 'best_choice', 'q_values','stage', 'done'
        ]
        self.episode_fields = [
             'episode', 'epsilon', 'episode_reward', 'simulation_steps','agent_steps', 'life'
        ]
        
        # Initialize log files
        self.init_step_log()
        self.init_episode_log()
        self.init_config_log()  
        self.init_training_log()
    
    def log_model_info(self, model):
        """Log information about a PyTorch model."""
        with open(self.model_info_filename, 'w') as f:
            # Log the model's architecture
            f.write("Model Architecture:\n")
            f.write(str(model))
            f.write("\n\n")

            # Log the details of each layer
            f.write("Layer Details:\n")
            for name, module in model.named_modules():
                f.write(f"{name} ({module.__class__.__name__}): \n")
                
                # Log the shape of the parameters
                for param_name, param in module.named_parameters():
                    f.write(f" - {param_name}: {param.size()}\n")
            
            # Log the total number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"\nTotal Parameters: {total_params}\n")

    def init_config_log(self):
        """Initialize the config log file and write the header."""
        with open(self.config_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.config.keys())
            writer.writeheader()

    def log_config(self):
        """Log the configuration data to the CSV file."""
        with open(self.config_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.config.keys())
            writer.writerow(self.config)

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
    
    def init_training_log(self):
        """Initialize the training log file and write the header."""
        with open(self.training_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.training_fields)
            writer.writeheader()

    def log_training(self, data):
        """Log the provided training data to the CSV file."""
        with open(self.training_filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.training_fields)
            writer.writerow(data)
            
    def log_model_info(self, model_info):
        """Log the model information."""
        with open(self.model_info_filename, 'w') as f:
            # Log the total number of parameters
            f.write(f"Total Parameters: {model_info['total_parameters']}\n\n")
            
            # Log details of each layer
            for layer in model_info['layers']:
                f.write(f"Layer Name: {layer['name']}\n")
                f.write(f"Layer Type: {layer['type']}\n")
                f.write(f"Parameters: {layer['parameters']}\n\n")