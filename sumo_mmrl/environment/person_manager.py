from .person import Person

class PersonManager:
    """
    PersonManager class to manage and create multiple Person instances.

    :param int num_of_people: The number of people to manage.
    :param dict edge_position: Dictionary of edge positions.
    :param sumo: SUMO simulation instance.
    :param dict index_dict: Dictionary of edge indices.
    """

    def __init__(self, num_of_people, edge_position, sumo, index_dict):
        """
        Initialize the PersonManager with the given parameters.
        """
        self.num_of_people = num_of_people
        self.edge_position = edge_position
        self.sumo = sumo
        self.index_dict = index_dict

    def create_people(self):
        """
        Create and return a list of Person instances.

        :return: List of Person instances.
        :rtype: list
        """
        people = []
        for p_id in range(self.num_of_people):
            people.append(
                Person(
                    str(p_id),
                    self.sumo,
                    self.edge_position,
                    self.index_dict,
                    p_id + 1,
                )
            )
        return people
