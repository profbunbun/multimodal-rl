from .outmask import OutMask
from .bus_stop import StopFinder


class Stage4:
    def __init__(self, edge_position_dic):
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.make_choice_flag = False
        self.stage = "final"
        self.old_edge = None
        self.agent_step = 0
        self.edge_position_dic = edge_position_dic
        self.route_flag = 0
        self.old_dist = None
        self.state = []

    def nullstep(self, vehicle, sumo):
        vedge = vehicle.get_road()

        while not self.make_choice_flag and self.stage != "done":
            sumo.simulationStep()
            vedge = vehicle.get_road()
            
            if ":" in vedge or self.old_edge == vedge:
                self.make_choice_flag = False
            else:
                self.make_choice_flag = True

        self.old_edge = vedge

    def step(self, action, validator, vehicle, person, sumo):
        self.agent_step += 1
        reward = 0
        vedge = vehicle.get_road()
        dest = person.destination

        vedge_loc = self.edge_position_dic[vedge]
        dest_loc = self.edge_position_dic[dest]
        self.old_dist = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], dest_loc[0], dest_loc[1]
        )
        
        
        self.nullstep(vehicle, sumo)

        vedge_loc = self.edge_position_dic[vedge]
        dest_loc = self.edge_position_dic[dest]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], dest_loc[0], dest_loc[1]
        )
        if self.old_dist > edge_distance:
            new_dist_check = 1
            reward += 1
        elif self.old_dist < edge_distance:
            new_dist_check = -1
            reward += -1.5
        else:
            new_dist_check = 0
            reward += -1
         
        
        self.old_dist = edge_distance
        choices = vehicle.get_out_dict()

        if validator == 1:
            if self.make_choice_flag:
                # self.agent_step += 1
                vehicle.set_destination(action)
                sumo.simulationStep()
                self.make_choice_flag = False

            vedge = vehicle.get_road()  # repeat
            choices = vehicle.get_out_dict()
            
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                edge_distance,
            ) = self.out_mask.get_outmask(vedge, dest, choices, self.edge_position_dic)



            vedge = vehicle.get_road()

            self.stage = "final"
            dest = dest.partition("_")[0]

            if vedge == dest:
                print(self.stage+' ',)
                self.stage = "done"
                reward += 45

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(sumo.simulation.getTime())
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.old_edge = vedge

            
            choices = vehicle.get_out_dict()
            return self.state, reward, self.stage, choices

        self.stage = "done"
        reward += -15
        self.make_choice_flag = False
        
        choices = vehicle.get_out_dict()
        return self.state, reward, self.stage, choices

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)