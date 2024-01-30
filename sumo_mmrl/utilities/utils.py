import yaml
import optuna
import sqlalchemy
import argparse
from functools import wraps
import time

class Utils:
    """
    A utility class containing various static methods for common tasks.

    This class provides a central location for methods that are used across
    various parts of the application, helping to reduce code duplication
    and improve maintainability.
    """

    @staticmethod
    def load_yaml_config(config_path):
        """
    Loads a YAML configuration file and returns the settings as a dictionary.

    :param file_path: The path to the YAML file to load.
    :type file_path: str
    :return: A dictionary containing the configuration settings.
    :rtype: dict

    :Example:

    >>> config = load_yaml_config('config.yaml')
    >>> print(config['training_settings']['episodes'])
    20000
    """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_next_study_name(storage_url, base_name="study"):
        """
        Generates a new unique study name based on existing studies in the Optuna storage database.
        
        Attempts to connect to the specified Optuna storage database and retrieves all existing study names.
        It then generates a new study name with an incremented highest number based on the specified base name.
        
        :param storage_url: The URL of the Optuna storage database.
        :type storage_url: str
        :param base_name: The base name for the study, defaults to "study".
        :type base_name: str, optional
        :return: A unique study name based on the existing studies in the database.
        :rtype: str
        
        :raises Exception: If connection to the storage database fails.
        :raises sqlalchemy.exc.OperationalError: If there's an issue accessing the study summaries, typically if the table doesn't exist.
        
        :Example:

        >>> get_next_study_name("sqlite:///example.db")
        'study_1'
        """
        try:
        # Attempt to connect to the Optuna storage database.
            storage = optuna.storages.RDBStorage(storage_url)
        except Exception as e:
            print(f"Failed to connect to the storage database: {e}")
            return base_name + "_1"

        # Get all existing study names.
        try:
            all_study_summaries = optuna.study.get_all_study_summaries(storage=storage)
        except sqlalchemy.exc.OperationalError:
            # If the table doesn't exist yet, this is likely the first study.
            return base_name + "_1"
        
        all_study_names = [summary.study_name for summary in all_study_summaries]

        # Find the highest existing study number.
        max_num = 0
        for name in all_study_names:
            if name.startswith(base_name):
                try:
                    num = int(name.split("_")[-1])  # Assumes the format "study_X"
                    max_num = max(max_num, num)
                except ValueError:
                    continue

        # Create a new unique study name.
        new_study_name = f"{base_name}_{max_num + 1}"
        return new_study_name
    
    def setup_study(study_name, storage_path, pruner):
        """
        Set up the Optuna study by loading an existing study or creating a new one.

        :param study_name: The name of the study to load or create.
        :type study_name: str or None
        :param storage_path: The path to the database storage for Optuna.
        :type storage_path: str
        :param pruner: The pruner instance to use for the new study.
        :type pruner: optuna.pruners.BasePruner
        :return: The loaded or newly created Optuna study.
        :rtype: optuna.study.Study
        """
        try:
            if study_name:
                # Try to load the existing study
                study = optuna.load_study(study_name=study_name, storage=storage_path)
                print(f"Resuming study {study_name}")
            else:
                # No study name provided, create a new one
                study_name = Utils.get_next_study_name(storage_path, base_name="study")
                study = optuna.create_study(
                    storage=storage_path,
                    study_name=study_name,
                    direction="maximize",
                    pruner=pruner,
                )
        except Exception as e:
            # Handle exceptions, such as if the study doesn't exist
            print(f"Failed to load or create study {study_name}: {e}")
            print("Creating a new study instead.")
            study_name = Utils.get_next_study_name(storage_path, base_name="study")
            study = optuna.create_study(
                storage=storage_path,
                study_name=study_name,
                direction="maximize",
                pruner=pruner,
            )
        return study

    
    def parse_arguments():
        """
        Parse command-line arguments.

        :return: The parsed arguments.
        :rtype: argparse.Namespace
        """
        parser = argparse.ArgumentParser(description="SUMO MMRL Optimization")
        parser.add_argument('--study_name', type=str, help='Name of the Optuna study.')
        # Add more arguments as needed
        return parser.parse_args()
    

