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


class Basic:
    def __init__(self, path, sumocon, steps_per_episode, num_of_vehic, types, test) -> None:
        self.plotter = Plotter()  # for plotting results
        self.out_mask = OutMask()
        self.finder = StopFinder()

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
            self.parser.get_edge_pos_dic())  # This creates the x y positions of all the lanes
       
       
        self.sumo = None  # this becomes the sumo conection used after GUI or command line is decided
        self.path = path  # path to the experiment folder  for simulation data
        self.steps_per_episode = steps_per_episode  # you get it
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.state = []


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
        # self.parser.get_route_edges()
        self.stages = ['RESET','PICKUP','DROPOFF','']
        

    def reset(self):
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.make_choice_flag = True
        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()
        
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
        self.sumo.simulationStep()
        
        vedge = self.vehicle.get_road()
        pedge = self.person.get_road()
        choices = self.vehicle.get_out_dict()
        destination_edge = pedge

        (vedge_loc,
         dest_edge_loc,
         outmask,
         edge_distance
         ) = self.out_mask.get_outmask(vedge,
                                       destination_edge,
                                       choices,
                                       self.edge_position)

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

    def step(self, action, validator):

      
        if self.stage == "pickup":
            
 
        elif self.stage == "dropoff":
          

        elif self.stage == "onbus":
           

        elif self.stage == "final":


        self.steps = int(self.sumo.simulation.getTime())

        if self.steps >= self.steps_per_episode:
            self.accumulated_reward += -15
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
