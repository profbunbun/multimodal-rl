from .outmask import OutMask
from .bus_stop import StopFinder


class Stage2:
    def __init__(self, edge_position_dic):
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.make_choice_flag = True
        self.stage = "dropoff"
        self.old_edge = None
        self.agent_step = 0
        self.edge_position_dic = edge_position_dic
        self.route_flag = 0
        self.old_dist = None
        self.state = []

    def nullstep(self, vehicle, dest, sumo):
        sumo.simulationStep()
        vedge = vehicle.get_road()

        if ":" in vedge or self.old_edge == vedge:
            self.make_choice_flag = False
        else:
            self.make_choice_flag = True
            vedge_loc = self.edge_position_dic[vedge]
            dest_loc = self.edge_position_dic[dest]
            self.old_dist = self.manhat_dist(
                vedge_loc[0], vedge_loc[1], dest_loc[0], dest_loc[1]
                )

        self.old_edge = vedge

    def step(self, action, validator, vehicle, person, sumo):
        self.agent_step += 1

        reward = 0
        vedge = vehicle.get_road()
        vedge_loc = self.edge_position_dic[vedge]
        pedge = person.get_road()
        dest = self.finder.find_begin_stop(
            pedge, self.edge_position_dic, sumo
        ).partition("_")[0]
        dest_loc = self.edge_position_dic[dest]
        self.old_dist = self.manhat_dist(
                vedge_loc[0], vedge_loc[1], dest_loc[0], dest_loc[1]
                )
        dest_loc = self.edge_position_dic[dest]
        
        while not self.make_choice_flag and self.stage != "done":
            self.nullstep(vehicle, dest,  sumo)

        vedge_loc = self.edge_position_dic[vedge]
        dest_loc = self.edge_position_dic[dest]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], dest_loc[0], dest_loc[1]
        )
        if self.old_dist > edge_distance:
            new_dist_check = 1
            reward += 1

        else:
            new_dist_check = -1
            reward += -1

        choices = vehicle.get_out_dict()

        if validator == 1:
            if self.make_choice_flag:
                vehicle.set_destination(action)
                sumo.simulationStep()
                self.make_choice_flag = False

            vedge = vehicle.get_road()  # repeat
            pedge = person.get_road()  # repeat
            dest = self.finder.find_begin_stop(pedge, self.edge_position_dic, sumo)
            choices = vehicle.get_out_dict()
            
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                edge_distance,
            ) = self.out_mask.get_outmask(vedge, dest, choices, self.edge_position_dic)

            vedge = vehicle.get_road()

            self.stage = "dropoff"
            dest = dest.partition("_")[0]

            if vedge == dest:
                print(self.stage+' ', end='')
                self.stage = "onbus"
                reward += 30

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(sumo.simulation.getTime())
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.old_edge = vedge
            self.old_dist = edge_distance
            
            choices = vehicle.get_out_dict()
            return self.state, reward, self.stage, choices

        self.stage = "done"
        reward += -15
        self.make_choice_flag = False

        choices = vehicle.get_out_dict()
        return self.state, reward, self.stage, choices

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
