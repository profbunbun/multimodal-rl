"""module stuff"""
# import numpy as np
import random
from ..connector.utility import Utility


util = Utility()


STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"

# out_mask=[]


class Vehicle:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, vehicle_id, out_dict, index_dict, sumo) -> None:
        """
        __init__ _summary_

        _extended_summary_

        Args:
            vehicle_id (_type_): _description_
            out_dict (_type_): _description_
            index_dict (_type_): _description_
            sumo (_type_): _description_
        """
        self.direction_choices = [STRAIGHT, TURN_AROUND, RIGHT, LEFT]
        self.vehicle_id = vehicle_id
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.sumo = sumo
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]

        # if ':' not in self.cur_loc:
        #     self.outlist=list(self.out_dict[self.cur_loc].keys())
        #     self.outlane=list(self.out_dict[self.cur_loc].values())

    def get_lane(self):
        """
        get_lane _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """

        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        return self.cur_loc

    def location(self):
        """
        location _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """
        vpos = self.sumo.vehicle.getPosition(self.vehicle_id)
        return [vpos[0], vpos[1]]

    def is_indexed_lane(self):
        """
        is_indexed_lane _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """
        current_lane = self.get_lane()
        if current_lane == "":
            return False

        if ":" not in current_lane:
            return True
        else:
            return False

        # def get_stats(self):
        #     current_lane=self.get_lane()
        #     current_location=current_lane.partition("_")[0]
        #     if ':' not in current_lane :
        #         return

        return

    def get_out_dict(self):
        """
        get_out_dict _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """
        lane = self.get_lane()
        options = self.out_dict[lane]
        return options

    def set_destination(self, action):
        """
        set_destination _summary_

        _extended_summary_

        Args:
            action (_type_): _description_
        """
        # print(action)

        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        outlist = list(self.out_dict[self.cur_loc].keys())
        if action in outlist:
            target_lane = self.out_dict[self.cur_loc][action]
            self.sumo.vehicle.changeTarget(self.vehicle_id, target_lane)

    def set_pickup(self):
        """
        set_pickup _summary_

        _extended_summary_
        """

    def pickup(self):
        """
        pickup _summary_

        _extended_summary_
        """
        # reservation = self.sumo.person.getTaxiReservations(0)
        # reservation_id = reservation[0]
        # self.sumo.vehicle.dispatchTaxi("1","0")
        # print(reservation_id)

    def close(self):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo.close()
        return

    def random_relocate(self):
        """
        random_relocate _summary_

        _extended_summary_
        """
        new_lane = random.choice(list(self.index_dict.keys()))
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=new_lane)
        self.sumo.vehicle.moveTo(self.vehicle_id, new_lane + "_0", 5)
        return
