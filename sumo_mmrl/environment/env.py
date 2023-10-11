"""  """

import numpy as np
from .connect import SUMOConnection
from .net_parser import NetParser
from .person import Person
from .plot_util import Plotter
from .ride_select import RideSelect
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage_reset import StageReset
from .vehicle import Vehicle


class Basic:
    def __init__(self, path, sumocon, steps_per_episode, num_of_vehic, types, test) -> None:
        self.plotter = Plotter()  # for plotting results

        self.parser = NetParser(  # This is all the stuff to create the
            path + sumocon  # different map network dictionaries
        )
        
        self.test=test
        self.sumo_con = SUMOConnection(
            path + sumocon
        )  # This handlies our communication
        # with the TRACI simulation interface (??REDUNDANT?)
        self.ruff_rider = RideSelect()  # This is object to select the vehicle
        self.edge_position = (
            self.parser.get_edge_pos_dic()
        )  # This creates the x y positions of all the lanes
        self.stage_reset = StageReset(
            self.parser.get_out_dic(),  # Reset stage sets everything to zero and starts simulation
            self.parser.get_edge_index(),  # Takes the different network dictionaries
            self.edge_position,
        )

        self.stage_1 = Stage1(
            self.edge_position
        )  # this stage does to pic the person up
        self.stage_2 = Stage2(
            self.edge_position
        )  # this stage drops them of at the bus stop
        self.stage_3 = Stage3(
            self.edge_position,
            self.parser.get_route_edges()
        )  # this stage drops them of at the bus stop
        self.stage_4 = Stage4(
            self.edge_position
        )  # this stage drops them of at the bus stop
        self.sumo = None  # this becomes the sumo conection used after GUI or command line is decided
        self.path = path  # path to the experiment folder  for simulation data
        self.steps_per_episode = steps_per_episode  # you get it
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.state = []

        self.stage = "reset"  # Stage flag changes as agent
        # succeeds in each stach.
        # reset is start and changes automatic

        self.make_choice_flag = False  # changes if agent is in a valid
        # state to make a decison:
        # not an intersection,
        # one choice on new lane

        self.old_edge = None  # location variable to check  # against new position. For make choice flag

        self.rewards = []
        self.epsilon_hist = []

        self.vehicle = None
        self.person = None

        self.p_index = 0

        self.edge_distance = None  # manhatten distance to destination

        self.destination_edge = None  # changes for each stage destination goal

        self.num_of_vehicles = num_of_vehic  # number community
        # vehicles to select from

        self.types = types  # types of passangers
        # and vehicles to pair for a trip
        self.parser.get_route_edges()

    def reset(self):
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.stage = "reset"
        self.make_choice_flag = True
        self.stage_1.agent_step = 0
        self.stage_2.agent_step = 0
        self.stage_3.agent_step = 0
        self.stage_4.agent_step = 0

        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()

        vehicles = []
        people = []

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

        for p_id in range(1):
            people.append(
                Person(str(p_id),
                       self.sumo,
                       self.edge_position,
                       index_dict,
                       p_id + 1,
                       )
            )

        self.person = people[0]
        vid_selected = self.ruff_rider.select(vehicles, self.person)
        self.vehicle = vehicles[int(vid_selected)]

        # self.vehicle.random_relocate()

        (
            self.state,
            self.stage,
            choices,
            vedge,
            self.edge_distance,
        ) = self.stage_reset.step(self.vehicle, self.person, self.sumo)
        self.old_edge = vedge
        return self.state, self.stage, choices

    def step(self, action, validator):

        self.stage_2.agent_step = 0
        self.stage_3.agent_step = 0
        self.stage_4.agent_step = 0
        self.steps = int(self.sumo.simulation.getTime())

        if self.stage == "pickup":
            (self.state, reward, self.stage, choices) = self.stage_1.step(
                action, validator, self.vehicle, self.person, self.sumo
            )
            self.agent_step = self.stage_1.agent_step
            self.accumulated_reward += reward
 
        elif self.stage == "dropoff":
            (self.state, reward, self.stage, choices) = self.stage_2.step(
                action, validator, self.vehicle, self.person, self.sumo
            )
            self.agent_step += self.stage_2.agent_step
            self.accumulated_reward += reward

        elif self.stage == "onbus":
            (self.state, reward, self.stage, choices) = self.stage_3.step(
                action, validator, self.vehicle, self.person, self.sumo
            )
            self.agent_step += self.stage_3.agent_step
            self.accumulated_reward += reward

        elif self.stage == "final":
            (self.state, reward, self.stage, choices) = self.stage_4.step(
                action, validator, self.vehicle, self.person, self.sumo
            )
            self.agent_step += self.stage_4.agent_step
            self.accumulated_reward += reward

        if self.steps >= self.steps_per_episode:
            self.accumulated_reward += -150
            self.stage = "done"

        return self.state, self.accumulated_reward, self.stage, choices

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
        file_name = self.path + "/Graphs/sumo-agent"+self.test+".png"

        self.plotter.plot_learning(x, self.rewards, self.epsilon_hist, file_name)
