""" import stuff """
import numpy as np
from .vehicle import Vehicle
from .person import Person
from .connect import SUMOConnection
from .plot_util import Plotter
from .net_parser import NetParser
from .outmask import OutMask
from .find_stop import StopFinder
from .routemask import RouteMask
from .stage1 import Stage1
from .stage_reset import StageReset


class Basic:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, path, sumocon, steps_per_episode, num_of_vehic, types) -> None:
        self.plotter = Plotter()
        self.parser = NetParser(path + sumocon)
        self.sumo_con = SUMOConnection(path + sumocon)
        self.out_mask = OutMask()
        self.finder = StopFinder()
        self.route_mask = RouteMask()
        self.stage_reset = StageReset(self.parser.get_out_dic(),
                                      self.parser.get_edge_index())

        self.edge_position = self.parser.get_edge_pos_dic()
        self.stage_1 = Stage1(self.edge_position)
        self.sumo = None
        self.path = path
        self.steps_per_episode = steps_per_episode
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.state = []

        self.done = False
        self.make_choice_flag = False

        self.old_edge = None

        self.rewards = []
        self.epsilon_hist = []

        self.vehicle = None
        self.person = None

        self.p_index = 0

        self.edge_distance = None

        self.destination_edge = None

        self.route_flag = 0

        self.num_of_vehicles = num_of_vehic
        self.types = types
        
    def reset(self):
        """
        reset _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.steps = 0
        self.agent_step = 0
        self.accumulated_reward = 0
        self.done = False
        self.make_choice_flag = True
        self.stage_1.agent_step = 0

        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()

        vehicles = []
        people = []
        
        for v_id in range(self.num_of_vehicles):
            vehicles.append(
                Vehicle(str(v_id), out_dict, index_dict,
                        self.edge_position, self.sumo, self.types)
            )
        self.vehicle = vehicles[0]

        for p_id in range(1):
            people.append(Person(str(p_id), self.sumo,
                                 self.edge_position, index_dict, self.types))

        self.person = people[0]
        self.vehicle.random_relocate()
        self.sumo.simulationStep()
        vedge = self.vehicle.get_road()
        pedge = self.person.get_road()
        choices = self.vehicle.get_out_dict()
        self.destination_edge = pedge
        
        (vedge_loc,
         dest_edge_loc,
         outmask,
         self.edge_distance
         ) = self.out_mask.get_outmask(vedge,
                                       self.destination_edge,
                                       choices,
                                       self.edge_position)

        new_dist_check = 1
        
        # print(self.sumo.simulation.getTime())
        self.state = []
        self.state.extend(vedge_loc)
        self.state.extend(dest_edge_loc)
        self.state.append(self.sumo.simulation.getTime())
        self.state.append(new_dist_check)
        self.state.extend(outmask)
        
        self.old_edge = vedge
        return self.state, self.done, choices

    def step(self, action, validator):
        """
        step _summary_

        _extended_summary_

        Keyword Arguments:
            action -- _description_ (default: {None})

        Returns:
            _description_
        """
        
        (self.state,
         reward,
         self.done,
         choices) = self.stage_1.step(action,
                                      validator,
                                      self.vehicle,
                                      self.person,
                                      self.sumo)
         
        self.agent_step = self.stage_1.agent_step
        
        self.steps = int(self.sumo.simulation.getTime())

        if self.steps >= self.steps_per_episode:
            reward += -45
            self.done = True

        self.accumulated_reward += reward
        return self.state, reward, self.done, choices

    def render(self, mode):
        """
        render _summary_

        _extended_summary_

        Arguments:
            mode -- _description_
        """
        if mode == "gui":
            self.sumo = self.sumo_con.connect_gui()

        elif mode == "libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()

        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()

    def close(self, episode, epsilon):
        """
        close _summary_

        _extended_summary_
        """
        
        self.sumo.close()
        acc_r = self.accumulated_reward
        acc_r = float(self.accumulated_reward)

        self.rewards.append(acc_r)

        self.epsilon_hist.append(epsilon)
        avg_reward = np.mean(self.rewards[-100:])

        print(
            "EP: ",
            episode,
            f"Reward: {acc_r:.3}",
            f" Average Reward  {avg_reward:.3}",
            f"epsilon {epsilon:.5}",
            f" **** step: {self.steps}",
            f"*** Agent steps: {self.stage_1.agent_step}",
        )

        x = [i + 1 for i in range(len(self.rewards))]
        file_name = self.path + "/Graphs/sumo-agent.png"

        self.plotter.plot_learning(x, self.rewards, self.epsilon_hist, file_name)
        # print(len(self.best_route))
        # print(self.best_route)
        # print(len(self.route))
        # print(self.route)
