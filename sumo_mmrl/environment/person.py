"""module stuff"""
import random



class Person:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, person_id, sumo, edge_position, index_dict) -> None:
        self.person_id = person_id
        self.sumo = sumo
        self.index_dict = index_dict
        self.edge_position = edge_position
        self.new_lane = random.choice(list(self.index_dict.keys()))
        # new_lane = list(self.index_dict.keys())[0]
        self.destination = random.choice(list(self.index_dict.keys()))
        # self.sumo.person.add(person_id, "1w", 20)
        # self.sumo.person.appendDrivingStage(person_id, "24e", lines="taxi")
        self.sumo.person.add(person_id, self.new_lane, 20)
        self.sumo.person.appendDrivingStage(person_id, self.destination, lines="taxi")

    def location(self):
        """
        location _summary_

        _extended_summary_

        Returns:
            _description_
        """
        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def get_destination(self):
        """
        set_destination _summary_

        _extended_summary_
        """

        return self.destination


    def remove_person(self):
        """
        remove_person _summary_

        _extended_summary_
        """    
        self.sumo.person.remove(self.person_id)
        return
                
    def get_road(self):
        '''
        get_road _summary_

        _extended_summary_

        :return: _description_
        :rtype: _type_
        '''        
        return self.sumo.person.getRoadID(self.person_id)
