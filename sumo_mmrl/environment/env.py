'''RL Enviroment'''
from .stage_manager import StageManager
import numpy as np
from .connect import SUMOConnection
from .net_parser import NetParser
from .plot_util import Plotter
from .ride_select import RideSelect
from .outmask import OutMask
from .bus_stop import StopFinder
from .observation import Observation
from .vehicle_manager import VehicleManager  # Import VehicleManager
from .person_manager import PersonManager # Import PersonManager
from .reward_calculator import RewardCalculator
from .utils import Utils
from .step_manager import StepManager


class Env:
    '''RL Enviroment'''
    def __init__(self, path, sumocon, num_of_vehic, types):
        self.obs = Observation()
        self.plotter = Plotter()  
        self.out_mask = OutMask()
        self.finder = StopFinder()

        self.parser = NetParser( 
            path + sumocon  
        )

        self.sumo_con = SUMOConnection(
            path + sumocon
        )


        self.ride_selector = RideSelect()  
        self.edge_position = (
            self.parser.get_edge_pos_dic()
        )  # This creates the x y positions of all the lanes

        self.sumo = None  
        
        self.path = path  

        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.reward = 0
        self.make_choice_flag = False 
        self.old_edge = None  
        self.old_dist = None
        self.rewards = []
        self.epsilon_hist = []

        self.vehicle = None
        self.person = None

        self.p_index = 0
        self.distcheck = 0

        self.edge_distance = None  # manhatten distance to destination

        self.destination_edge = None  # changes for each stage destination goal

        self.num_of_vehicles = num_of_vehic  # number community
        # vehicles to select from

        self.types = types  # types of passangers
        # and vehicles to pair for a trip
        # self.parser.get_route_edges()
        self.stage = "reset"
        self.bussroute = self.parser.get_route_edges()
        self.life = 20
        self.reward_calculator = RewardCalculator(self.edge_position)
        

    

    def reset(self):
        '''reset everything '''
        
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.life = 20
        self.make_choice_flag = True
        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()
        self.vehicle_manager = VehicleManager(1, self.edge_position, self.sumo, out_dict, index_dict)
        self.person_manager = PersonManager(1, self.edge_position, self.sumo, index_dict)
        self.stage_manager = StageManager(self.finder, self.edge_position, self.sumo, self.bussroute)
        self.step_manager = StepManager(self.sumo)
        self.stage = self.stage_manager.get_initial_stage()
        self.reward = 0

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

        dest_loc = self.edge_position[self.destination_edge]

        state = self.obs.get_state(self.sumo, self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)

        return state, self.stage, choices, 

    
    def step(self, action, validator):
        '''next step for environment'''

        self.steps = int(self.sumo.simulation.getTime())
        self.agent_step += 1

        # Using StepManager for null steps
        self.make_choice_flag, self.old_edge = self.step_manager.null_step(self.vehicle, self.make_choice_flag, self.old_edge)

        if self.life <= 0:
            self.stage = "done"
            validator = 0

        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_position[vedge]
        dest_edge_loc = self.edge_position[self.destination_edge]

        edge_distance = Utils.manhattan_distance(
            vedge_loc[0], vedge_loc[1], dest_edge_loc[0], dest_edge_loc[1]
        )

        # Calculate reward
        self.reward, self.make_choice_flag, self.distcheck, self.life = self.reward_calculator.calculate_reward(
            self.old_dist, edge_distance, self.stage, self.destination_edge, vedge, self.make_choice_flag, self.life)

        if validator == 1:
            if self.make_choice_flag:
                # Using StepManager to perform action
                self.best_choice = self.step_manager.perform_step(self.vehicle, action, self.destination_edge)
                self.make_choice_flag = False
                self.life -= 1

            # Update stage using StageManager
            self.stage, self.destination_edge = self.stage_manager.update_stage(
                self.stage, self.destination_edge, vedge, self.person
            )

            choices = self.vehicle.get_out_dict()

            dest_loc = self.edge_position[self.destination_edge]
            state = self.obs.get_state(self.sumo,self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)

            self.old_edge = vedge
            self.old_dist = edge_distance
            return state, self.reward, self.stage, choices

        choices = self.vehicle.get_out_dict()

        self.stage = "done"
        self.reward = -0.15
        self.make_choice_flag = False

        dest_loc = self.edge_position[self.destination_edge]

        state = self.obs.get_state(self.sumo,self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)


        self.accumulated_reward += self.reward
        return state, self.reward, self.stage, choices




    def render(self, mode):
        '''how to render based on OS and commad line'''
        if mode == "gui":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()
    

    def close(self, episode, accu, current_epsilon):
        '''close connection and print graph'''
        steps = self.sumo.simulation.getTime()
        self.sumo.close()
        acc_r = float(accu)
        self.rewards.append(acc_r)

        self.epsilon_hist.append(current_epsilon)

        avg_reward = np.mean(self.rewards[-100:])
        smoothed_rewards = self.smooth_data(self.rewards, 100)

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
        self.plotter.plot_learning(x, smoothed_rewards, self.epsilon_hist, file_name)
        return avg_reward
    def smooth_data(self, data, window_size):
        """Smooth data using a simple moving average."""
        if not data: return []

        smoothed = []
        running_total = 0
        non_numeric_count = 0

        for i, value in enumerate(data):
            if value is not None:
                running_total += value
            else:
                non_numeric_count += 1

            if i >= window_size:
                if data[i - window_size] is not None:
                    running_total -= data[i - window_size]
                else:
                    non_numeric_count -= 1

            count = min(i + 1, window_size) - non_numeric_count
            if count > 0:
                smoothed.append(running_total / count)
            else:
                smoothed.append(None)  # or some default/fallback value

        if non_numeric_count > 0:
            print(f"Warning: Found {non_numeric_count} non-numeric values. Some smoothing results may be None.")

        return smoothed
    
    def get_steps_per_episode(self):
        return self.sumo.simulation.getTime()
    
    def get_global_step(self):
        return self.agent_step
    
    def get_destination_edge_id(self):
        return self.destination_edge
    
    def get_vehicle_location_edge_id(self):
        return self.vehicle.get_lane()
    
    def get_best_choice(self):
        return self.best_choice
    
    def get_out_lanes(self):
        return self.vehicle.get_out_dict()
    
    def get_life(self):
        return self.life
    
    
    
    