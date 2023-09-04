"""module stuff"""
from ..connector.utility import Utility
import random
util = Utility()


class Person:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, person_id, sumo, index_dict) -> None:
        self.person_id = person_id
        self.sumo = sumo
        self.index_dict = index_dict
        # new_lane = random.choice(list(self.index_dict.keys()))
        new_lane = list(self.index_dict.keys())[0]
        destination = random.choice(list(self.index_dict.keys()))
        self.sumo.person.add(person_id, "1w", 20)
        self.sumo.person.appendDrivingStage(person_id, "24e", lines="taxi")
        # self.sumo.person.add(person_id, new_lane, 20)
        # self.sumo.person.appendDrivingStage(person_id, destination, lines="ANY")

    def location(self):
        """
        location _summary_

        _extended_summary_

        Returns:
            _description_
        """
        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def set_destination(self):
        """
        set_destination _summary_

        _extended_summary_
        """


    def remove_person(self):
        """
        remove_person _summary_

        _extended_summary_
        """    
        self.sumo.person.remove(self.person_id)
        return
                

