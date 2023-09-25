"""module stuff"""
import random


class Person:

    def __init__(self, person_id, sumo, edge_position, index_dict, types) -> None:
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
        
        self.sumo.person.appendDrivingStage(person_id,
                                            self.destination,
                                            lines="taxi")
        
        self.sumo.person.setParameter(person_id,
                                      "type",
                                      str(random.randint(1, types)))
        # print(self.sumo.person.getParameter(person_id, "type"))

    def location(self):
   
        ppos = self.sumo.person.getPosition(self.person_id)

        return ppos

    def get_destination(self):
   
        return self.destination


    def remove_person(self):

        self.sumo.person.remove(self.person_id)
        return
                
    def get_road(self):
     
        return self.sumo.person.getRoadID(self.person_id)
