'''RL Enviroment'''
from .stage_manager import StageManager
import numpy as np
from .connect import SUMOConnection
# from .net_parser import NetParser
from .ride_select import RideSelect
from .outmask import OutMask
from .bus_stop import StopFinder
from .observation import Observation
from .vehicle_manager import VehicleManager 
from .person_manager import PersonManager 
from .reward_calculator import RewardCalculator
from ..utilities.env_utils import Utils
from .step_manager import StepManager
from functools import wraps
import time


class Env:


    def __init__(self, config, edge_locations, bussroute, out_dict, index_dict):


        self.config = config  
        self.path = config['training_settings']['experiment_path']
        self.sumo_config_path = self.path + config['training_settings']['sumoconfig']
        self.num_of_vehicles = config['env']['num_of_vehicles']
        self.types_of_passengers = config['env']['types_of_passengers']
        self.graph_path = self.path + config['graph_output']['graph_path']
        self.life = config['training_settings']['initial_life']
        self.penalty = config['training_settings']['penalty']
        self.smoothing_window = config['training_settings']['smoothing_window']
        
        self.obs = Observation()
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.sumo_con = SUMOConnection(self.sumo_config_path)
        self.ride_selector = RideSelect()  
        self.edge_locations = edge_locations
        self.sumo = None   
        self.agent_step = 0
        self.accumulated_reward = 0
        self.make_choice_flag = False 
        self.old_edge = None  
        self.old_dist = None
        self.rewards = []
        self.epsilon_hist = []
        self.vehicle = None
        self.person = None
        self.p_index = 0
        self.distcheck = 0
        self.edge_distance = None
        self.destination_edge = None
        self.stage = "reset"
        self.bussroute = bussroute
        self.out_dict = out_dict
        self.reward_calculator = RewardCalculator(self.edge_locations)
        self.index_dict = index_dict
        

    def reset(self):
  
        self.agent_step = 0
        self.accumulated_reward = 0
        self.life = self.config['training_settings']['initial_life']
        self.make_choice_flag = True
        
        

        self.vehicle_manager = VehicleManager(self.config['env']['num_of_vehicles'], self.edge_locations, self.sumo, self.out_dict, self.index_dict)
        self.person_manager = PersonManager(self.config['env']['num_of_people'], self.edge_locations, self.sumo, self.index_dict)
        self.stage_manager = StageManager(self.finder, self.edge_locations, self.sumo, self.bussroute)
        self.step_manager = StepManager(self.sumo)

        self.stage = self.stage_manager.get_initial_stage()
        vehicles = self.vehicle_manager.create_vehicles()
        people = self.person_manager.create_people()
        self.person = people[0]
        vid_selected = self.ride_selector.select(vehicles, self.person)
        self.vehicle = vehicles[int(vid_selected)]
        self.sumo.simulationStep()
        self.old_dist = 0
        vedge = self.vehicle.get_road()
        self.old_edge = vedge
        choices = self.vehicle.get_out_dict()
        self.destination_edge = self.person.get_road()
        dest_loc = self.edge_locations[self.destination_edge]
        state = self.obs.get_state(self.sumo, self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)

        return state, self.stage, choices, 


    def step(self, action, validator):
        '''
        Performs a step in the environment based on the given action.

        :param action: Action to be performed.
        :type action: int
        :param validator: Validates if the step should be performed.
        :type validator: int
        :return: New state, reward, stage, and choices.
        :rtype: tuple
        '''

        self.agent_step += 1

        if self.life <= 0:
            self.stage = "done"
            validator = 0

        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_locations[vedge]
        dest_edge_loc = self.edge_locations[self.destination_edge]

        edge_distance = Utils.manhattan_distance(
            vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
        )


        if validator == 1:
            if self.make_choice_flag:
                vedge = self.step_manager.perform_step(self.vehicle, action, self.destination_edge)
                self.make_choice_flag = False
                self.life -= 0.01

            self.make_choice_flag, self.old_edge = self.step_manager.null_step(self.vehicle, self.make_choice_flag, self.old_edge)

            vedge = self.vehicle.get_road()
            choices = self.vehicle.get_out_dict()
            dest_loc = self.edge_locations[self.destination_edge]

            reward, self.make_choice_flag, self.distcheck, self.life = self.reward_calculator.calculate_reward(
                self.old_dist, edge_distance, self.destination_edge, vedge, self.make_choice_flag, self.life)

            self.stage, self.destination_edge = self.stage_manager.update_stage(
                self.stage, self.destination_edge, vedge, self.person, self.vehicle
            )
            if self.stage == "done": 
                reward += 0.99
                print("successfull dropoff")
            choices = self.vehicle.get_out_dict()

            state = self.obs.get_state(self.sumo,self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)
            self.old_edge = vedge
            self.old_dist = edge_distance
            return state, reward, self.stage, choices

        choices = self.vehicle.get_out_dict()

        self.stage = "done"
        reward = self.penalty
        self.make_choice_flag = False
        dest_loc = self.edge_locations[self.destination_edge]
        state = self.obs.get_state(self.sumo,self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)
        self.accumulated_reward += reward
        return state, reward, self.stage, choices

    # @timeit
    def render(self, mode):
        '''
        Renders the environment based on the given mode.

        :param mode: Mode to render the environment ('gui', 'libsumo', 'no_gui').
        :type mode: str
        '''
        if mode == "gui":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()
    
    # @timeit
    def close(self, episode, accu, current_epsilon):
        '''
        Closes the environment and prints out the graph of rewards.

        :param episode: Current episode number.
        :type episode: int
        :param accu: Accumulated reward.
        :type accu: float
        :param current_epsilon: Current epsilon value for exploration.
        :type current_epsilon: float
        :return: Average reward.
        :rtype: float
        '''
        steps = self.sumo.simulation.getTime()
        self.sumo.close()
        acc_r = float(accu)
        self.rewards.append(acc_r)

        self.epsilon_hist.append(current_epsilon)

        avg_reward = np.mean(self.rewards[-100:])
        # smoothed_rewards = Utils.smooth_data(self.rewards, 100)

        print_info = {
            "EP": episode,
            "Reward": f"{acc_r:.5}",
            "Avg Reward": f"{avg_reward:.3}",
            "epsilon": f"{current_epsilon:.3}",
            "time": f"{steps}",
            "steps": f"{self.agent_step}",
            }
        print(", ".join(f"{k}: {v}" for k, v in print_info.items()))

        x = list(range(1, len(self.rewards) + 1))
        file_name = self.path + "/Graphs/sumo-agent.png"
        # Utils.plot_learning_curve(x, smoothed_rewards, self.epsilon_hist, file_name)
        return
    
    def quiet_close(self):
        '''
        Closes the environment without printing out the graph of rewards.
        '''
        self.sumo.close()
        return
   