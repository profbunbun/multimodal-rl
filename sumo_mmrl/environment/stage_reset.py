"""
     _summary_

    _extended_summary_

    Returns:
        _type_: _description_
    """
from .outmask import OutMask
from .find_stop import StopFinder
from .routemask import RouteMask
from .vehicle import Vehicle
from .person import Person


class StageReset:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, out_dict, index_dict):
        # # self.out_mask = OutMask()
        # # self.finder = StopFinder()
        # # self.route_mask = RouteMask()
        # # self.make_choice_flag = False
        # # self.done = False
        # # self.old_edge = None
        # self.agent_step = 0
        # # self.edge_position_dic = edge_position_dic
        # # self.route_flag = 0
        # # self.state = []
        self.out_dict = out_dict
        self.index_dict = index_dict
        
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.done = False
        self.make_choice_flag = True
        self.vehicle = None
        self.person = None
        self.sumo = None
    
    def step(self, vehicle, person, sumo):
        """
        step _summary_

        _extended_summary_

        Keyword Arguments:
            action -- _description_ (default: {None})

        Returns:
            _description_
        """
        self.sumo = sumo
        self.person = person
        self.vehicle = vehicle

        self.vehicle.random_relocate()
        self.sumo.simulationStep()
        vedge = self.vehicle.get_road()
        pedge = self.person.get_road()
        choices = self.vehicle.get_out_dict()
        (
            vedge_loc,
            dest_edge_loc,
            outmask,
            self.edge_distance,
        ) = self.out_mask.get_outmask(
            vedge, pedge, choices, self.edge_position
        )

        new_dist_check = 1
       
        self.state = []
        self.state.extend(vedge_loc)
        self.state.extend(dest_edge_loc)
        self.state.append(self.sumo.simulation.getTime())
        self.state.append(new_dist_check)
        self.state.extend(outmask)
        self.old_edge = vedge
        return self.state, self.done, choices


    def manhat_dist(self, x1, y1, x2, y2):
        """
        manhat_dist _summary_

        _extended_summary_

        Args:
            x1 (_type_): _description_
            y1 (_type_): _description_
            x2 (_type_): _description_
            y2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return abs(x1 - x2) + abs(y1 - y2)
