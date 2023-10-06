
# import numpy as np
import random


STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"


class Vehicle:

    def __init__(self, vehicle_id, out_dict, index_dict, edge_position, sumo, i) -> None:

        self.direction_choices = [STRAIGHT, TURN_AROUND, RIGHT, LEFT]
        self.vehicle_id = vehicle_id
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.sumo = sumo
        self.edge_position = edge_position
        self.sumo.vehicle.add(self.vehicle_id, "r_0", typeID="taxi")
        self.sumo.vehicle.setParameter(vehicle_id,
                                       "type", str(2)
                                    #    str(random.randint(1, i))
                                       )


        
        self.random_relocate()
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]

    def get_lane(self):
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        return self.cur_loc

    def location(self):

        vpos = self.sumo.vehicle.getPosition(self.vehicle_id)
        return [vpos[0], vpos[1]]

    def get_out_dict(self):
 
        lane = self.get_lane()
        options = self.out_dict[lane]
        return options

    def set_destination(self, action):

        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        outlist = list(self.out_dict[self.cur_loc].keys())
        if action in outlist:
            target_lane = self.out_dict[self.cur_loc][action]
            self.sumo.vehicle.changeTarget(self.vehicle_id, target_lane)

    def pickup(self):

        reservation = self.sumo.person.getTaxiReservations(0)
        reservation_id = reservation[0]
        self.sumo.vehicle.dispatchTaxi(self.vehicle_id,"0")
        # print(reservation_id)
        
    def get_road(self):    
        return self.sumo.vehicle.getRoadID(self.vehicle_id)

    def random_relocate(self):    
        new_lane=random.choice(list(self.index_dict.keys()))      
        self.sumo.vehicle.changeTarget(self.vehicle_id,edgeID=new_lane)
        self.sumo.vehicle.moveTo(self.vehicle_id,new_lane+"_0",5)

    def get_type(self):
        return self.sumo.vehicle.getParameter(self.vehicle_id,
                                              "type")
        
    def teleport(self, dest):
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)
        self.sumo.vehicle.moveTo(self.vehicle_id, dest+"_0", 5)