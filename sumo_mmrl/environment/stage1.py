from .outmask import OutMask
from .bus_stop import StopFinder


class Stage1:
 
    def __init__(self, edge_position_dic):
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.make_choice_flag = False
        self.stage = "pickup"
        self.old_edge = None
        self.agent_step = 0
        self.edge_position_dic = edge_position_dic
        self.route_flag = 0
        self.state = []
        self.old_dist = None

    def nullstep(self, vehicle, person, sumo):

        sumo.simulationStep()
        vedge = vehicle.get_road()
        
        if ":" in vedge or self.old_edge == vedge:
            self.make_choice_flag = False
        else:
            self.make_choice_flag = True
            pedge = person.get_road()
            vedge_loc = self.edge_position_dic[vedge]
            pedge_loc = self.edge_position_dic[pedge]
            self.old_dist = self.manhat_dist(
                vedge_loc[0], vedge_loc[1], pedge_loc[0], pedge_loc[1]
                )

        self.old_edge = vedge

    def step(self, action, validator, vehicle, person, sumo):
        # self.make_choice_flag = True
        while not self.make_choice_flag and self.stage != "done":
            self.nullstep(vehicle, person,  sumo)

        reward = 0
        vedge = vehicle.get_road()
        pedge = person.get_road()
        vedge_loc = self.edge_position_dic[vedge]
        pedge_loc = self.edge_position_dic[pedge]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], pedge_loc[0], pedge_loc[1]
        )
        if self.old_dist >= edge_distance:
            new_dist_check = 1
            reward += .5
        else:
            new_dist_check = -1
            reward += -1
        
        choices = vehicle.get_out_dict()

        if validator == 1:
            if self.make_choice_flag:
                vehicle.set_destination(action)
                sumo.simulationStep()
                # reward += -1
                self.make_choice_flag = False

            vedge = vehicle.get_road()
            pedge = person.get_road()
            choices = vehicle.get_out_dict()
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                edge_distance,
            ) = self.out_mask.get_outmask(
                vedge, pedge, choices, self.edge_position_dic
            )

            vedge = vehicle.get_road()

            self.stage = "pickup"

            if vedge == pedge:
                print(self.stage+' ', end='')
                # self.stage = "done"
                self.stage = "dropoff"
                
                self.make_choice_flag = True
                new_dest = self.finder.find_begin_stop(vedge,
                                                       self.edge_position_dic,
                                                       sumo).partition("_")[0]
                (
                    vedge_loc,
                    dest_edge_loc,
                    outmask,
                    edge_distance,
                    ) = self.out_mask.get_outmask(
                        vedge, new_dest, choices, self.edge_position_dic
                        )

                reward += 90

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(self.agent_step)
            # self.state.append(sumo.simulation.getTime())
            # self.state.append(edge_distance)
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.old_edge = vedge
            self.old_dist = edge_distance
            self.agent_step += 1
            choices = vehicle.get_out_dict()
            return self.state, reward, self.stage, choices

        self.stage = "done"
        reward += -15
        self.make_choice_flag = False
        self.agent_step += 1
        choices = vehicle.get_out_dict()
        return self.state, reward, self.stage, choices

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
