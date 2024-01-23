'''RL Enviroment'''
from .stage_manager import StageManager
import numpy as np
from .connect import SUMOConnection
from .net_parser import NetParser
from .ride_select import RideSelect
from .outmask import OutMask
from .bus_stop import StopFinder
from .observation import Observation
from .vehicle_manager import VehicleManager 
from .person_manager import PersonManager 
from .reward_calculator import RewardCalculator
from ..utilities.env_utils import Utils
from .step_manager import StepManager


class Env:
    """
RL Environment for managing and simulating the interaction between vehicles, passengers, and the SUMO environment.

Attributes:
    :param dict config: Configuration settings for the environment.
    :param str path: Path to the experiment files.
    :param str sumo_config_path: Path to the SUMO configuration files.
    :param int num_of_vehicles: Number of vehicles in the environment.
    :param list types_of_passengers: Types of passengers in the environment.
    :param str graph_path: Path to the graph output files.
    :param int life: Initial life of the agent.
    :param float penalty: Penalty for wrong actions.
    :param int smoothing_window: Window size for smoothing the rewards.
    :param Observation obs: Observation object to capture the environment's state.
    :param OutMask out_mask: Object to manage the action space.
    :param StopFinder finder: Object to manage bus stops.
    :param NetParser parser: Object to parse the network files.
    :param SUMOConnection sumo_con: Object to manage the SUMO connection.
    :param RideSelect ride_selector: Object to select the ride for the passengers.
    :param dict edge_position: Dictionary of edge positions.
    :param SUMO sumo: SUMO simulation object.
    :param int steps: Current simulation step.
    :param int agent_step: Current step of the agent.
    :param float accumulated_reward: Accumulated reward of the agent.
    :param bool make_choice_flag: Flag to indicate if the agent should make a choice.
    :param str old_edge: ID of the previous edge the vehicle was on.
    :param float old_dist: Previous distance to the destination.
    :param list rewards: List of rewards per episode.
    :param list epsilon_hist: History of epsilon values for exploration.
    :param Vehicle vehicle: Current vehicle object.
    :param Person person: Current person object.
    :param int p_index: Index of the current person.
    :param float distcheck: Distance checker for the current step.
    :param float edge_distance: Distance to the destination edge.
    :param str destination_edge: ID of the destination edge.
    :param str stage: Current stage of the environment.
    :param list bussroute: List of edges for the bus route.
    :param RewardCalculator reward_calculator: Object to calculate rewards.

Methods:
    reset(): 
        Resets the environment to the initial state.

    step(action, validator): 
        Performs a step in the environment based on the given action.

    render(mode): 
        Renders the environment based on the given mode.

    close(episode, accu, current_epsilon): 
        Closes the environment and prints out the graph of rewards.

    get_steps_per_episode(): 
        Returns the number of steps per episode.

    get_global_step(): 
        Returns the global step count.

    get_destination_edge_id(): 
        Returns the ID of the destination edge.

    get_vehicle_location_edge_id(): 
        Returns the ID of the vehicle's current edge.

    get_best_choice(): 
        Returns the best choice made by the agent.

    get_out_lanes(): 
        Returns the outgoing lanes from the current edge.

    get_life(): 
        Returns the current life of the agent.
"""
    def __init__(self, config):
        '''
        Initializes the RL Environment with the given configuration.

        :param config: Configuration settings for the environment.
        :type config: dict
        '''
        

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
        self.parser = NetParser(self.sumo_config_path)
        self.sumo_con = SUMOConnection(self.sumo_config_path)
        self.ride_selector = RideSelect()  
        self.edge_position = (
            self.parser.get_edge_pos_dic()
        )  
        self.sumo = None   
        self.steps = 0
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
        self.bussroute = self.parser.get_route_edges()
        self.reward_calculator = RewardCalculator(self.edge_position)
        

    

    def reset(self):
        '''
        Resets the environment to the initial state.

        :return: Initial state, stage, and choices.
        :rtype: tuple
        '''

        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.life = self.config['training_settings']['initial_life']
        self.make_choice_flag = True
        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()

        self.vehicle_manager = VehicleManager(self.config['env']['num_of_vehicles'], self.edge_position, self.sumo, out_dict, index_dict)
        self.person_manager = PersonManager(self.config['env']['num_of_people'], self.edge_position, self.sumo, index_dict)
        self.stage_manager = StageManager(self.finder, self.edge_position, self.sumo, self.bussroute)
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
        dest_loc = self.edge_position[self.destination_edge]
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

        self.steps = int(self.sumo.simulation.getTime())
        self.agent_step += 1
        # self.make_choice_flag, self.old_edge = self.step_manager.null_step(self.vehicle, self.make_choice_flag, self.old_edge)

        if self.life <= 0:
            self.stage = "done"
            validator = 0

        vedge = self.vehicle.get_road()
        vedge_loc = self.edge_position[vedge]
        dest_edge_loc = self.edge_position[self.destination_edge]

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
            dest_loc = self.edge_position[self.destination_edge]

            reward, self.make_choice_flag, self.distcheck, self.life = self.reward_calculator.calculate_reward(
                self.old_dist, edge_distance, self.stage, self.destination_edge, vedge, self.make_choice_flag, self.life)

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
        dest_loc = self.edge_position[self.destination_edge]
        state = self.obs.get_state(self.sumo,self.agent_step, self.vehicle, dest_loc, self.life, self.distcheck)
        self.accumulated_reward += reward
        return state, reward, self.stage, choices




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
        smoothed_rewards = Utils.smooth_data(self.rewards, 100)

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
        Utils.plot_learning_curve(x, smoothed_rewards, self.epsilon_hist, file_name)
        return
    
    def quiet_close(self):
        '''
        Closes the environment without printing out the graph of rewards.
        '''
        self.sumo.close()
        return
   
    
    def get_steps_per_episode(self):
        '''
        Returns the number of steps per episode.

        :return: Number of steps.
        :rtype: int
        '''
        return self.sumo.simulation.getTime()
    
    def get_global_step(self):
        '''
        Returns the global step count.

        :return: Global step count.
        :rtype: int
        '''
        return self.agent_step
    
    def get_destination_edge_id(self):
        '''
        Returns the ID of the destination edge.

        :return: Destination edge ID.
        :rtype: str
        '''
        return self.destination_edge
    
    def get_vehicle_location_edge_id(self):
        '''
        Returns the ID of the vehicle's current edge.

        :return: Vehicle location edge ID.
        :rtype: str
        '''
        return self.vehicle.get_lane()
    
    
    
    def get_out_lanes(self):
        '''
        Returns the outgoing lanes from the current edge.

        :return: Outgoing lanes.
        :rtype: dict
        '''
        return self.vehicle.get_out_dict()
    
    def get_life(self):
        '''
        Returns the current life of the agent.

        :return: Life of the agent.
        :rtype: int
        '''
        return self.life
    def get_legal_actions(self):
        '''
        Returns the legal actions of the agent.

        :return: Legal actions.
        :rtype: list
        '''
        return self.vehicle.get_out_dict()
    
