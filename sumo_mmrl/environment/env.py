""" import stuff """
import numpy as np
from .vehicle import Vehicle
from .person import Person
from .connect import SUMOConnection
from .plot_util import Utility
from .net_parser import NetParser
from .outmask import OutMask
from .find_stop import StopFinder
from .routemask import RouteMask
from .stage1 import Stage1


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
        self.finder = StopFinder()
        self.route_mask = RouteMask()
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

        out_dict = self.parser.get_out_dic()
        index_dict = self.parser.get_edge_index()

        vehicles = []
        people = []

        for v_id in range(1):
            vehicles.append(
                Vehicle(str(v_id), out_dict, index_dict,
                        self.edge_position, self.sumo)
            )
        self.vehicle = vehicles[0]

        for p_id in range(1):
            people.append(Person(str(p_id), self.sumo,
                                 self.edge_position, index_dict))
        self.person = people[0]

        self.vehicle.random_relocate()
        self.sumo.simulationStep()
        self.steps += 1
        vedge = self.vehicle.get_road()

        pedge = self.person.get_road()
        p_destination = self.person.get_destination()

        choices = self.vehicle.get_out_dict()
        self.destination_edge = pedge
        (
            vedge_loc,
            dest_edge_loc,
            outmask,
            self.edge_distance,
        ) = self.out_mask.get_outmask(
            vedge, self.destination_edge, choices, self.edge_position
        )

        new_dist_check = 1
        closest_end_stop = self.finder.find_end_stop(
            p_destination, self.edge_position, self.sumo
        )

        closest_begin_stop = self.finder.find_begin_stop(
            pedge, self.edge_position, self.sumo
        )
        # line=self.finder.get_line(self.closest_begin_stop)
        lineroute = self.finder.get_line_route(self.sumo)
        routemask = self.route_mask.get_route_mask(
            vedge, choices, self.route_flag, lineroute
        )
        print(self.sumo.simulation.getTime())
        self.state = []
        self.state.extend(vedge_loc)
        self.state.extend(dest_edge_loc)
        self.state.append(self.sumo.simulation.getTime())
        self.state.append(new_dist_check)
        self.state.extend(outmask)
        self.state.append(self.route_flag)
        self.state.extend(routemask)

        self.old_edge = vedge
        return self.state, self.done, choices

    def nullstep(self):
        """
        nullstep _summary_

        _extended_summary_
        """
        self.sumo.simulationStep()
        self.steps += 1

        vedge = self.vehicle.get_road()

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
            vedge = self.vehicle.get_road()

            choices = self.vehicle.get_out_dict()
            (
                vedge_loc,
                dest_edge_loc,
                outmask,
                self.edge_distance,
            ) = self.out_mask.get_outmask(
                vedge, self.destination_edge, choices, self.edge_position
            )

            if old_dist > self.edge_distance:
                new_dist_check = 1
            else:
                new_dist_check = -1

            vedge = self.vehicle.get_road()

            self.done = False

            if new_dist_check == 1:
                reward += 0.1

            if vedge == self.destination_edge:
                self.done = True

                # self.vehicle.pickup()
                # print(self.sumo_con.busstopCheck())
                # print("Pickup ", self.p_index)
                reward += 45
                # self.vehicle.set_destination(self.sumo_con.busstopCheck()[0])

                self.accumulated_reward += reward
            if self.steps >= self.steps_per_episode:
                self.done = True
                reward += -10
            self.accumulated_reward += reward

            lineroute = self.finder.get_line_route(self.sumo)
            routemask = self.route_mask.get_route_mask(
                vedge, choices, self.route_flag, lineroute
            )
            print(self.sumo.simulation.getTime())

            self.state = []
            self.state.extend(vedge_loc)
            self.state.extend(dest_edge_loc)
            self.state.append(self.sumo.simulation.getTime())
            self.state.append(new_dist_check)
            self.state.extend(outmask)
            self.state.append(self.route_flag)
            self.state.extend(routemask)

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
            f"*** Agent steps: {self.agent_step}",
        )

        x = [i + 1 for i in range(len(self.rewards))]
        file_name = self.path + "/Graphs/sumo-agent.png"

        self.util.plot_learning(x, self.rewards, self.epsilon_hist, file_name)
        # print(len(self.best_route))
        # print(self.best_route)
        # print(len(self.route))
        # print(self.route)
