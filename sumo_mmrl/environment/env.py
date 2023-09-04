""" import stuff """
import numpy as np
from .vehicle import Vehicle
from .person import Person
from .connect import SUMOConnection
from .plot_util import Utility
from .net_parser import NetParser
from .outmask import OutMask


class Basic:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, path, sumocon, steps_per_episode) -> None:
        self.util = Utility()
        self.parser = NetParser(path + sumocon)
        self.sumo_con = SUMOConnection(path + sumocon)
        self.out_mask = OutMask()
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

        self.edge_position = self.parser.get_edge_pos_dic()
        self.edge_distance = None

    def reset(self):
        """
        reset _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.steps = 0
        self.agent_step = 0
        vehicles = []
        people = []
        self.accumulated_reward = 0
        self.p_index = 0

        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()


        for v_id in range(1):
            vehicles.append(Vehicle(str(v_id), out_dict, index_dict, self.edge_position, self.sumo))
        self.vehicle = vehicles[0]

        for p_id in range(1):
            people.append(Person(str(p_id), self.sumo, self.edge_position, index_dict))
        self.person = people[self.p_index]

        self.sumo.simulationStep()
        self.steps += 1

        vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
        pedge = self.sumo.person.getRoadID(self.person.person_id)


        new_dist_check = 1
        choices=self.vehicle.get_out_dict()

        vedge_loc, pedge_loc, outmask, self.edge_distance = self.out_mask.get_outmask(vedge, pedge, choices, self.edge_position)
        
        self.state = []

        self.state.extend(vedge_loc)
        self.state.extend(pedge_loc)
        self.state.append(self.steps)
        self.state.append(new_dist_check)
        self.state.extend(outmask)

        self.done = False
        self.make_choice_flag = True



        self.old_edge = vedge
        return self.state, self.done, choices

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
        old_dist = self.edge_distance
        reward = 0
        choices = self.vehicle.get_out_dict()

        if validator != -1:

            if self.make_choice_flag:
                self.vehicle.set_destination(action)
                reward += -0.1
                self.accumulated_reward += reward
                self.agent_step += 1
                self.make_choice_flag = False

            self.sumo.simulationStep()
            self.steps += 1
            vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
            pedge = self.sumo.person.getRoadID(self.person.person_id)
            
            choices = self.vehicle.get_out_dict()
            vedge_loc, pedge_loc, outmask, self.edge_distance = self.out_mask.get_outmask(vedge, pedge, choices, self.edge_position)


            if old_dist > self.edge_distance:
                new_dist_check = 1
            else:
                new_dist_check = -1
            

            vedge = self.sumo.vehicle.getRoadID(self.vehicle.vehicle_id)
            pedge = self.sumo.person.getRoadID(self.person.person_id)


            self.done = False

            if new_dist_check == 1:
                reward += 0.1

            if vedge == pedge:
                self.done = True

                self.vehicle.pickup()
                # print(self.sumo_con.busstopCheck())
                # print("Pickup ", self.p_index)
                reward += 45
                self.vehicle.set_destination(self.sumo_con.busstopCheck()[0])

                self.accumulated_reward += reward
            if self.steps >= self.steps_per_episode:
                self.done = True
                reward += -10
            self.accumulated_reward += reward
            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(pedge_loc)
            self.state.append(self.steps)
            self.state.append(new_dist_check)
            self.state.extend(outmask)

            self.old_edge = vedge
            while not self.make_choice_flag and not self.done:
                self.nullstep()
            return self.state, reward, self.done, choices
        else:
            self.done = True
            reward += -15
            self.accumulated_reward += reward
            self.make_choice_flag = False

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

    def c_l_output(self, episode, epsilon):
        """
        c_l_output _summary_

        _extended_summary_
        """
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
            f"*** Agent steps: {self.agent_step}",
        )

    def chart_output(self):
        """
        chart_output _summary_

        _extended_summary_
        """
        x = [i + 1 for i in range(len(self.rewards))]
        file_name = self.path + "/Graphs/sumo-agent.png"

        self.util.plot_learning(x, self.rewards, self.epsilon_hist, file_name)

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
