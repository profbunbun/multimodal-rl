from .outmask import OutMask


class StageReset:

    def __init__(self, out_dict, index_dict, edge_position):
        self.out_mask = OutMask()
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.edge_position = edge_position
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.stage = "reset"
        self.make_choice_flag = True
        
    
    def step(self, vehicle, person, sumo):
        
    
        sumo.simulationStep()
        
        vedge = vehicle.get_road()
        pedge = person.get_road()
        choices = vehicle.get_out_dict()
        destination_edge = pedge

        (vedge_loc,
         dest_edge_loc,
         outmask,
         edge_distance
         ) = self.out_mask.get_outmask(vedge,
                                       destination_edge,
                                       choices,
                                       self.edge_position)

        new_dist_check = 1
        state = []
        state.extend(vedge_loc)
        state.extend(dest_edge_loc)
        state.append(sumo.simulation.getTime())
        # state.append(edge_distance)
        state.append(new_dist_check)
        state.extend(outmask)
        self.stage = "pickup"
        return state, self.stage, choices, vedge, edge_distance

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
