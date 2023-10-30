"""  """

import numpy as np
from .connect import SUMOConnection
from .net_parser import NetParser
from .person import Person
from .plot_util import Plotter
from .ride_select import RideSelect
from .outmask import OutMask
from .bus_stop import StopFinder
from .vehicle import Vehicle


class Env:
    def __init__(
        self, path, sumocon, steps_per_episode, num_of_vehic, types, test
    ) -> None:
        self.plotter = Plotter()  # for plotting results
        self.out_mask = OutMask()
        self.finder = StopFinder()

        self.parser = NetParser(  # This is all the stuff to create the
            path + sumocon  # different map network dictionaries
        )

        self.test = test
        self.sumo_con = SUMOConnection(
            path + sumocon
        )  # This handlies our communication
        # with the TRACI simulation interface (??REDUNDANT?)
        self.ruff_rider = RideSelect()  # This is object to select the vehicle
        self.edge_position = (
            self.parser.get_edge_pos_dic()
        )  # This creates the x y positions of all the lanes
        
        self.max = self.parser.get_max_manhattan()

        self.sumo = None  # this becomes the sumo conection used after GUI 
        # or command line is decided
        self.path = path  # path to the experiment folder  for simulation data
        self.steps_per_episode = steps_per_episode  # you get it
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.reward = 0

        self.make_choice_flag = False  # changes if agent is in a valid
        # state to make a decison:
        # not an intersection,
        # one choice on new lane

        self.old_edge = None  # location variable to check
        # against new position. For make choice flag
        self.old_dist = None
        self.rewards = []
        self.epsilon_hist = []

        self.vehicle = None
        self.person = None
        self.pedge = None

        self.p_index = 0

        self.edge_distance = None  # manhatten distance to destination

        self.destination_edge = None  # changes for each stage destination goal

        self.num_of_vehicles = num_of_vehic  # number community
        # vehicles to select from

        self.types = types  # types of passangers
        # and vehicles to pair for a trip
        # self.parser.get_route_edges()
        self.stage = "reset"
        self.bussroute = self.parser.get_route_edges()

    def reset(self):
        
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.make_choice_flag = True
        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()
        self.reward = 0

        vehicles = []
        for v_id in range(self.num_of_vehicles):
            vehicles.append(
                Vehicle(
                    str(v_id),
                    out_dict,
                    index_dict,
                    self.edge_position,
                    self.sumo,
                    v_id + 1,
                )
            )

        people = []
        for p_id in range(1):
            people.append(
                Person(
                    str(p_id),
                    self.sumo,
                    self.edge_position,
                    index_dict,
                    p_id + 1,
                )
            )

        self.person = people[0]
        vid_selected = self.ruff_rider.select(vehicles, self.person)
        self.vehicle = vehicles[int(vid_selected)]
        self.sumo.simulationStep()
        self.old_dist = 0

        vedge = self.vehicle.get_road()
        self.old_edge = vedge
        self.pedge = self.person.get_road()
        choices = self.vehicle.get_out_dict()
        self.destination_edge = self.pedge

        (vedge_loc,
         dest_edge_loc,
         outmask,
         edge_distance) = self.out_mask.get_outmask(
            vedge, self.destination_edge, choices, self.edge_position
        )

        new_dist_check = 1
        state = []
        xmin = self.parser.net_minmax()[0][0]
        xmax = self.parser.net_minmax()[1][0]
        ymin = self.parser.net_minmax()[0][1]
        ymax = self.parser.net_minmax()[1][1]
        vx = self.scale_xy(vedge_loc[0], xmin, xmax, 0, 1)
        vy = self.scale_xy(vedge_loc[1], ymin, ymax, 0, 1)
        destx = self.scale_xy(dest_edge_loc[0], xmin, xmax, 0, 1)
        desty = self.scale_xy(dest_edge_loc[1], ymin, ymax, 0, 1)
        state.append(vx)
        state.append(vy)
        state.append(destx)
        state.append(desty)
        state.append(self.agent_step)
        # state.append(self.sumo.simulation.getTime())
        # state.append(math.log(edge_distance+1))
        state.append(self.new_range(edge_distance, self.max, 10))
        state.append(new_dist_check)
        state.extend(outmask)
        self.stage = "pickup"

        return state, self.stage, choices

    def nullstep(self):
        vedge = self.vehicle.get_road()

        if self.steps >= self.steps_per_episode:
            self.reward += -20
            self.stage = "done"

        while not self.make_choice_flag and self.stage != "done":
            self.sumo.simulationStep()
            vedge = self.vehicle.get_road()

            if ":" in vedge or self.old_edge == vedge:
                self.make_choice_flag = False
            else:
                self.make_choice_flag = True
        self.old_edge = vedge

    def step(self, action, validator):
        
        self.reward = 0
        self.steps = int(self.sumo.simulation.getTime())        
        self.agent_step += 1

        self.nullstep()
        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_position[vedge]
        dest_edge_loc = self.edge_position[self.destination_edge]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
        )

        if self.old_dist > edge_distance:
            new_dist_check = 1
            # self.reward += 1
            # self.reward += 2
        else:
            new_dist_check = -1
            # self.reward += -0.01
        # else:
        #     new_dist_check = -1
        #     self.reward += -0.15

        if validator == 1:
            if self.make_choice_flag:
                # self.reward += -0.2
                self.vehicle.set_destination(action)
                self.sumo.simulationStep()
                self.make_choice_flag = False

            choices = self.vehicle.get_out_dict()
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                edge_distance,
            ) = self.out_mask.get_outmask(
                vedge, self.destination_edge, choices, self.edge_position
            )
            if vedge == self.destination_edge:
                self.reward += 30
                match self.stage:
                    case "pickup":
                        self.reward += 10
                        print(self.stage + " ", end="")
                        self.stage = "dropoff"
                        self.make_choice_flag = True
                        self.destination_edge = self.finder.find_begin_stop(
                            self.pedge, self.edge_position, self.sumo
                            ).partition("_")[0]

                    case "dropoff":
                        self.reward += 20
                        print(self.stage + " ", end="")
                        self.stage = "onbus"  # for test. change back to onbus
                        next_route_edge = self.bussroute[1].partition("_")[0]
                        self.destination_edge = next_route_edge
                        self.make_choice_flag = True

                    case "onbus":
                        self.reward += 30
                        print(self.stage + " ", end="")
                        self.stage = "final"
                        self.make_choice_flag = True
                        end_stop = self.finder.find_end_stop(
                            self.person.destination,
                            self.edge_position,
                            self.sumo).partition("_")[0]
                        self.vehicle.teleport(end_stop.partition("_")[0])
                        dest = self.person.destination
                        dest_loc = self.edge_position[dest]
                        self.sumo.simulationStep()
                        self.old_dist = self.manhat_dist(
                            vedge_loc[0], vedge_loc[1], dest_loc[0],
                            dest_loc[1])

                    case "final":
                        self.reward += 40
                        print(self.stage + " ", end="")
                        self.stage = "done"
                        self.make_choice_flag = True
                       
            state = []
            xmin = self.parser.net_minmax()[0][0]
            xmax = self.parser.net_minmax()[1][0]
            ymin = self.parser.net_minmax()[0][1]
            ymax = self.parser.net_minmax()[1][1]
            vx = self.scale_xy(vedge_loc[0], xmin, xmax, 0, 1)
            vy = self.scale_xy(vedge_loc[1], ymin, ymax, 0, 1)
            destx = self.scale_xy(dest_edge_loc[0], xmin, xmax, 0, 1)
            desty = self.scale_xy(dest_edge_loc[1], ymin, ymax, 0, 1)
            state.append(vx)
            state.append(vy)
            state.append(destx)
            state.append(desty)
            state.append(self.agent_step)
            # state.append(math.log(edge_distance+1))
            # state = self.normalize(state, 0, self.max)
            state.append(self.new_range(edge_distance, self.max, 10))
            state.append(new_dist_check)
            state.extend(outmask)
            self.old_edge = vedge
            self.old_dist = edge_distance
            return state, self.reward, self.stage, choices
    
        choices = self.vehicle.get_out_dict()
        (
            vedge_loc,
            dest_edge_loc,
            outmask,
            edge_distance,
        ) = self.out_mask.get_outmask(
            vedge, self.destination_edge, choices, self.edge_position
        )

        self.stage = "done"
        self.reward += -15
        self.make_choice_flag = False

        state = []
        xmin = self.parser.net_minmax()[0][0]
        xmax = self.parser.net_minmax()[1][0]
        ymin = self.parser.net_minmax()[0][1]
        ymax = self.parser.net_minmax()[1][1]
        vx = self.scale_xy(vedge_loc[0], xmin, xmax, 0, 1)
        vy = self.scale_xy(vedge_loc[1], ymin, ymax, 0, 1)
        destx = self.scale_xy(dest_edge_loc[0], xmin, xmax, 0, 1)
        desty = self.scale_xy(dest_edge_loc[1], ymin, ymax, 0, 1)
        state.append(vx)
        state.append(vy)
        state.append(destx)
        state.append(desty)
        state.append(self.agent_step)
        # state.append(math.log(edge_distance+1))
        state.append(self.new_range(edge_distance, self.max, 10))
        state.append(new_dist_check)
        state.extend(outmask)
        self.old_edge = vedge
        self.old_dist = edge_distance
        self.accumulated_reward += self.reward
        return state, self.reward, self.stage, choices

    def render(self, mode):
        if mode == "gui":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()

    def close(self, episode, epsilon, accu):
        steps = self.sumo.simulation.getTime()

        self.sumo.close()
        # acc_r = self.accumulated_reward
        acc_r = accu
        acc_r = float(acc_r)

        self.rewards.append(acc_r)

        self.epsilon_hist.append(epsilon)
        avg_reward = np.mean(self.rewards[-100:])

        print(
            "EP: ",
            episode,
            f"Reward: {acc_r:.5}",
            f" Average Reward  {avg_reward:.3}",
            f"epsilon {epsilon:.5}",
            f" **** step: {steps }",
            f"*** Agent steps: {self.agent_step}",
        )

        x = [i + 1 for i in range(len(self.rewards))]
        file_name = self.path + "/Graphs/sumo-agent" + self.test + ".png"

        self.plotter.plot_learning(x, self.rewards,
                                   self.epsilon_hist, file_name)

    def manhat_dist(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def normalize(self, arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def new_range(self, old_val, old_max, new_max):
        old_min = 0
        new_min = 0
        return (
            ((old_val - old_min) * (new_max - new_min))
            / (old_max - old_min)) + new_min
        
    def scale_xy(self, old_val, old_min, old_max, new_min, new_max):
        
        return (
            ((old_val - old_min) * (new_max - new_min))
            / (old_max - old_min)) + new_min