from .outmask import OutMask
from .bus_stop import StopFinder


class Stage3:
    def __init__(self, edge_position_dic, bus_route):
        
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.agent_step = 0
        self.bus_route = bus_route
        self.make_choice_flag = True
        self.stage = "onbus"
        self.edge_position_dic = edge_position_dic
        self.route_flag = 0

    def step(self, action, validator, vehicle, person, sumo):

        self.agent_step += 1
        reward = 0
        vedge = vehicle.get_road()
        begin_stop = self.finder.find_begin_stop(
            person.get_road(), self.edge_position_dic, sumo
            ).partition("_")[0]
        
        if vedge != begin_stop:
            vehicle.teleport(begin_stop)
            sumo.simulationStep()
            vedge = vehicle.get_road()
            
        next_route_edge = self.bus_route[1]
        pers_dest = person.destination
        end_stop = self.finder.find_end_stop(pers_dest, self.edge_position_dic,sumo).partition("_")[0]
        vedge_loc = self.edge_position_dic[vedge]

        if validator == 1:

            choices = vehicle.get_out_dict()
            
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                edge_distance,
            ) = self.out_mask.get_outmask(vedge, next_route_edge, choices, self.edge_position_dic)

            self.stage = "onbus"
            next_route_edge = next_route_edge.partition("_")[0]
            if action in choices:
                choice_edge = choices[action]

                if choice_edge == next_route_edge:
                    print(self.stage+' ', end='')
                    self.stage = "final"
                    vehicle.teleport(end_stop.partition("_")[0])
                    sumo.simulationStep()
                    reward += 35

                    state = []
                    state.extend(vedge_loc)
                    state.extend(dest_edge_loc)
                    state.append(sumo.simulation.getTime())
                    state.append(1)
                    state.extend(outmask)
                    choices = vehicle.get_out_dict()
                    
                    return state, reward, self.stage, choices

        self.stage = "done"
        reward += -15
        state = []
        state.extend(vedge_loc)
        state.extend(dest_edge_loc)
        state.append(sumo.simulation.getTime())
        state.append(1)
        state.extend(outmask)
        choices = vehicle.get_out_dict()
        return state, reward, self.stage, choices

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
