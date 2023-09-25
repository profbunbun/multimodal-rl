from .outmask import OutMask
from .find_stop import StopFinder


class Stage2:
 
    def __init__(self, edge_position_dic):
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.make_choice_flag = False
        self.done = False
        self.old_edge = None
        self.agent_step = 0
        self.edge_position_dic = edge_position_dic
        self.route_flag = 0
        self.state = []

    def nullstep(self, vehicle, sumo):

        sumo.simulationStep()
        vedge = vehicle.get_road()

        if ":" in vedge or self.old_edge == vedge:
            self.make_choice_flag = False
        else:
            self.make_choice_flag = True

        self.old_edge = vedge

    def step(self, action, validator, vehicle, person, sumo):
        print("part 2")

        reward = -0.1
        vedge = vehicle.get_road()
        pedge = person.get_road()
        vedge_loc = self.edge_position_dic[vedge]
        pedge_loc = self.edge_position_dic[pedge]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], pedge_loc[0], pedge_loc[1]
        )
        old_dist = edge_distance
        choices = vehicle.get_out_dict()

        if validator != -1:
            if self.make_choice_flag:
                self.agent_step += 1
                vehicle.set_destination(action)
                reward += -0.1
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

            if old_dist >= edge_distance:
                new_dist_check = 1
                reward += 0.1
            else:
                new_dist_check = -1
                reward -= 0.1

            vedge = vehicle.get_road()

            self.done = False

            if vedge == pedge:
                self.done = True
                reward += 50

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(sumo.simulation.getTime())
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.old_edge = vedge
            while not self.make_choice_flag and not self.done:
                self.nullstep(vehicle, sumo)
            return self.state, reward, self.done, choices

        self.done = True
        reward += -15
        self.make_choice_flag = False
        self.agent_step += 1
        return self.state, reward, self.done, choices

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
