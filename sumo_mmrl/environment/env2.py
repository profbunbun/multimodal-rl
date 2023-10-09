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

        self.sumo = None  # this becomes the sumo conection used after GUI 
        # or command line is decided
        self.path = path  # path to the experiment folder  for simulation data
        self.steps_per_episode = steps_per_episode  # you get it
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.reward = 0
        self.state = []

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
        pedge = self.person.get_road()
        choices = self.vehicle.get_out_dict()
        self.destination_edge = pedge

        (vedge_loc,
         dest_edge_loc,
         outmask,
         edge_distance) = self.out_mask.get_outmask(
            vedge, self.destination_edge, choices, self.edge_position
        )

        # self.vehicle.random_relocate()
        new_dist_check = 1
        state = []
        state.extend(vedge_loc)
        state.extend(dest_edge_loc)
        state.append(self.sumo.simulation.getTime())
        # state.append(edge_distance)
        state.append(new_dist_check)
        state.extend(outmask)
        self.stage = "pickup"

        return self.state, self.stage, choices

    def nullstep(self):
        vedge = self.vehicle.get_road()
        if self.steps >= self.steps_per_episode:
            self.accumulated_reward += -15
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
        
        self.steps = int(self.sumo.simulation.getTime())
        if self.steps >= self.steps_per_episode:
            self.accumulated_reward += -15
            self.stage = "done"
        self.agent_step += 1
        self.reward = 0

        self.nullstep()
        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_position[vedge]
        dest_edge_loc = self.edge_position[self.destination_edge]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
        )

        if self.old_dist > edge_distance:
            new_dist_check = 1
            self.reward += 2
        elif self.old_dist < edge_distance:
            new_dist_check = -1
            self.reward += -1.5
        else:
            new_dist_check = 0
            self.reward += -1

        if validator == 1:
            if self.make_choice_flag:
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
                match self.stage:
                    case "pickup":
                        print(self.stage + " ", end="")
                        self.stage = "dropoff"
                        self.make_choice_flag = True
                        self.reward += 100
                        self.destination_edge = self.finder.find_begin_stop(
                            self.pedge, self.edge_position, self.sumo
                            ).partition("_")[0]

                    case "dropoff":
                        print(self.stage + " ", end="")
                        self.stage = "done"  # for test. change back to onbus
                        self.make_choice_flag = True
                        self.reward += 100

                    case "onbus":
                        print(self.stage + " ", end="")
                        self.stage = "final"
                        self.make_choice_flag = True
                        self.reward += 100

                    case "final":
                        print(self.stage + " ", end="")
                        self.stage = "done"
                        self.make_choice_flag = True
                        self.reward += 100

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(self.sumo.simulation.getTime())
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.old_edge = vedge
            self.old_dist = edge_distance
            choices = self.vehicle.get_out_dict()
            return self.state, self.reward, self.stage, choices
    
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
        self.reward += -50
        self.make_choice_flag = False

        self.state = []
        self.state.extend(vedge_loc)
        self.state.extend(dest_edge_loc)
        self.state.append(self.sumo.simulation.getTime())
        self.state.append(new_dist_check)
        self.state.extend(outmask)
        self.old_edge = vedge
        self.old_dist = edge_distance
        choices = self.vehicle.get_out_dict()
        self.accumulated_reward += self.reward

        return self.state, self.reward, self.stage, choices

    def render(self, mode):
        if mode == "gui":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()

    def close(self, episode, epsilon):
        steps = self.sumo.simulation.getTime()

        self.sumo.close()
        acc_r = self.accumulated_reward
        acc_r = float(self.accumulated_reward)

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