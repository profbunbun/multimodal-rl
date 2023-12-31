import yaml

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
        Load a YAML configuration file.

        Given the path to a YAML file, this method opens the file, loads the contents
        using PyYAML, and returns the resulting dictionary.

        :param config_path: The file path to the YAML configuration file.
        :type config_path: str
        :return: A dictionary representation of the YAML configuration.
        :rtype: dict
        :raises FileNotFoundError: If the specified file does not exist.
        :raises yaml.YAMLError: If the file cannot be parsed as YAML.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Add more utility methods here with similar documentation style.
