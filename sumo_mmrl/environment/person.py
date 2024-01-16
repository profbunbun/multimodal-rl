import random


class Person:
    """
    Person class representing an individual in the simulation.

    :param str person_id: Unique identifier for the person.
    :param sumo: SUMO simulation instance.
    :param dict edge_position: Dictionary of edge positions.
    :param dict index_dict: Dictionary of edge indices.
    :param int types: Type identifier for the person.
    """

    def __init__(self, person_id, sumo, edge_position, index_dict, types) -> None:
        """
        Initialize a Person instance with the given parameters.
        """
        self.person_id = person_id
        # start_edge = "-E1_0"
        self.sumo = sumo
        self.index_dict = index_dict
        self.edge_position = edge_position
        # self.new_lane = list(self.index_dict.keys())[21]
        self.new_lane = random.choice(list(self.index_dict.keys()))
        self.destination = "-463787633#1"
        # self.destination = list(self.index_dict.keys())[0]
        # self.destination = random.choice(list(self.index_dict.keys()))

        self.sumo.person.add(person_id, "547782537#2", 20)

        self.sumo.person.appendDrivingStage(person_id,
                                            self.destination,
                                            lines="taxi")

        self.sumo.person.setParameter(person_id,
                                      "type", str(2))
        #   str(random.randint(1, types))

    def location(self):
        """
        Get the current location of the person.

        :return: The current (x, y) position of the person.
        :rtype: tuple
        """

        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def get_destination(self):
        """
        Get the destination of the person.

        :return: Destination edge ID.
        :rtype: str
        """

        return self.destination

    def remove_person(self):
        """
        Remove the person from the simulation.
        """
        self.sumo.person.remove(self.person_id)
     
    def get_road(self):
        """
        Get the current road ID of the person.

        :return: Current road ID.
        :rtype: str
        """

        return self.sumo.person.getRoadID(self.person_id)

    def get_type(self):
        """
        Get the type parameter of the person.

        :return: Type of the person.
        :rtype: str
        """
        return self.sumo.person.getParameter(self.person_id,
                                             "type")
