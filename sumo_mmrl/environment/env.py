""" import stuff """
import math
import numpy as np
from .vehicle import Vehicle
from .person import Person
from ..connector.connect import SUMOConnection
from ..connector.utility import Utility


class Basic:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, path, sumocon, steps_per_episode) -> None:
        self.util = Utility()
        self.sumo_con = SUMOConnection(path + sumocon)
        (   self.out_dict,
            self.index_dict,
            self.edge_list,
            self.edge_position,
        ) = self.sumo_con.get_edges_info()

        self.steps_per_episode = steps_per_episode

        self.done = False
        self.make_choice_flag = False
        self.vehicle = None
        self.person = None
        self.route = []
        self.done = False
        self.steps = 0
        self.agent_step = 0
        self.old_edge = None
        self.sumo = None
        self.path = path
        self.accumulated_reward = 0
        self.rewards = []
        self.eps_history = []
        self.vehicles = []
        self.choices = None
                            #  s    t    r    l
        self.distance_mask = [-100,-100,-100,-100]
        self.state = np.array([])
        
        # for v_id in range(10):
        #     self.vehicles.append(Vehicle(str(v_id), self.out_dict, self.index_dict, self.sumo))

    def reset(self):
        """
        reset _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.steps = 0
        self.agent_step = 0
        self.route = []
        self.vehicles = []
        self.distance_mask = [-100,-100,-100,-100]
        self.accumulated_reward = 0
        for v_id in range(1):
            self.vehicles.append(Vehicle(str(v_id), self.out_dict, self.index_dict, self.sumo))
        self.vehicle = self.vehicles[0]
        
        # self.vehicle.random_relocate()
        self.sumo.simulationStep()
        self.steps += 1

        vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
        pedge = self.sumo.person.getRoadID("p_0")
        
        vedge_loc=self.edge_position[vedge]
        pedge_loc=self.edge_position[pedge]
        edge_distance=math.dist(vedge_loc, pedge_loc)
        # self.route.append(vedge)

        vloc = self.vehicle.location()

        self.person = Person("p_0", self.sumo)

        self.choices = self.vehicle.get_out_dict()
        ploc = self.person.location()
        # distance = math.dist(vloc, ploc)

        for key, value in self.choices.items():
            if key == 's':
                sloc = self.edge_position[value]
                s_dist = math.dist(sloc,pedge_loc)
                if s_dist < edge_distance:
                    self.distance_mask[0] = 1
                else:
                    self.distance_mask[0] = -1   
            elif key == 't':
                tloc = self.edge_position[value]
                t_dist = math.dist(tloc,pedge_loc)
                if t_dist < edge_distance:
                    self.distance_mask[1] = 1
                else:
                    self.distance_mask[1] = -1
            
            elif key == 'r':
                rloc = self.edge_position[value]
                r_dist = math.dist(rloc,pedge_loc)
                if r_dist < edge_distance:
                    self.distance_mask[2] = 1
                else:
                    self.distance_mask[2] = -1
            
            elif key == 'l':
                lloc = self.edge_position[value]
                l_dist = math.dist(lloc,pedge_loc)
                if l_dist < edge_distance:
                    self.distance_mask[3] = 1
                else:
                    self.distance_mask[3] = -1
            



        self.state= np.array([])

        self.state= np.append(self.state, vedge_loc)
        self.state= np.append(self.state, pedge_loc)

        self.state= np.append(self.state, self.steps)
        self.state= np.append(self.state, edge_distance)
        self.state= np.append(self.state,self.distance_mask)

        self.done = False
        self.make_choice_flag = True

        self.choices = self.vehicle.get_out_dict()

        # self.best_route=self.sumo.simulation.findRoute(self.vedge,self.pedge)

        self.old_edge = vedge
        return self.state, self.done, self.choices, self.distance_mask

    def nullstep(self):
        """
        nullstep _summary_

        _extended_summary_
        """
        self.sumo.simulationStep()
        self.steps += 1

        vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)

        if ":" in vedge or self.old_edge == vedge:
            self.make_choice_flag = False
        else:
            self.make_choice_flag = True

        if self.steps >= self.steps_per_episode:
            self.done = True

        self.old_edge = vedge

    def step(self, action, validator):
        """
        step _summary_

        _extended_summary_

        Keyword Arguments:
            action -- _description_ (default: {None})

        Returns:
            _description_
        """
        reward = 0
        if validator != -1:
            
            self.distance_mask = [-100,-100,-100,-100]
            if self.make_choice_flag:
                self.vehicle.set_destination(action)
                reward += -0.05
                self.accumulated_reward += reward
                self.agent_step += 1
                self.make_choice_flag = False

            self.sumo.simulationStep()
            self.steps += 1
            vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
            pedge = self.sumo.person.getRoadID("p_0")
            vedge_loc=self.edge_position[vedge]
            pedge_loc=self.edge_position[pedge]
            edge_distance=math.dist(vedge_loc, pedge_loc)
            self.choices = self.vehicle.get_out_dict()
            for key, value in self.choices.items():
                if key == 's':
                    sloc = self.edge_position[value]
                    s_dist = math.dist(sloc,pedge_loc)
                    if s_dist < edge_distance:
                        self.distance_mask[0] = 1
                    else:
                        self.distance_mask[0] = -1   
                if key == 't':
                    tloc = self.edge_position[value]
                    t_dist = math.dist(tloc,pedge_loc)
                    if t_dist < edge_distance:
                        self.distance_mask[1] = 1
                    else:
                        self.distance_mask[1] = -1
                
                if key == 'r':
                    rloc = self.edge_position[value]
                    r_dist = math.dist(rloc,pedge_loc)
                    if r_dist < edge_distance:
                        self.distance_mask[2] = 1
                    else:
                        self.distance_mask[2] = -1
                
                if key == 'l':
                    lloc = self.edge_position[value]
                    l_dist = math.dist(lloc,pedge_loc)
                    if l_dist < edge_distance:
                        self.distance_mask[3] = 1
                    else:
                        self.distance_mask[3] = -1

            vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
            pedge = self.sumo.person.getRoadID("p_0")
            self.route.append(vedge)

            self.done = False

            if vedge == pedge:
                self.done = True
                reward += 3
                self.accumulated_reward += reward
            if self.steps >= self.steps_per_episode:
                self.done = True
                reward += -1
                self.accumulated_reward += reward
            self.state= np.array([])
            self.state= np.append(self.state, vedge_loc)
            self.state= np.append(self.state, pedge_loc)
            self.state= np.append(self.state, self.agent_step)
            self.state= np.append(self.state, edge_distance)
            self.state= np.append(self.state,self.distance_mask)

            self.old_edge = vedge
            while not self.make_choice_flag and not self.done:
                self.nullstep()
            return self.state, reward, self.done, self.choices, self.distance_mask
        else:
            self.done = True
            reward += -10
            self.accumulated_reward += reward
            self.make_choice_flag=False
                  
           
            
            return self.state, reward, self.done, self.choices, self.distance_mask

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

    def c_l_output(self, episode, epsilon):
        """
        c_l_output _summary_

        _extended_summary_
        """
        acc_r = self.accumulated_reward
        acc_r = float(self.accumulated_reward)

        self.rewards.append(acc_r)

        self.eps_history.append(epsilon)
        avg_reward = np.mean(self.rewards[-100:])

        print(
            "EP: ",
            episode,
            f"Reward: {acc_r:.3}",
            f" Average Reward  {avg_reward:.3}",
            f"epsilon {epsilon:.5}",
            f" **** step: {self.steps}",
            f"*** Agent steps: {self.agent_step}",
        )

    def chart_output(self):
        """
        chart_output _summary_

        _extended_summary_
        """
        x = [i + 1 for i in range(len(self.rewards))]
        file_name = self.path + "/Graphs/sumo-agent.png"

        self.util.plot_learning(x, self.rewards, self.eps_history, file_name)

    def close(self, episode, epsilon):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo.close()
        self.c_l_output(episode, epsilon)
        self.chart_output()
        # print(len(self.best_route))
        # print(self.best_route)
        # print(len(self.route))
        # print(self.route)
