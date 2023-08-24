""" import stuff """
import math
import numpy as np
from Objects.vehicle import Vehicle
from Objects.person import Person
from Connector.connect import SUMOConnection


class Basic:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, sumocon: str, steps_per_episode) -> None:
        self.sumo_con = SUMOConnection(sumocon)
        (
            self.out_dict,
            self.index_dict,
            self.edge_list,
            self.edge_position,
        ) = self.sumo_con.get_edges_info()

        self.steps_per_episode = steps_per_episode
        self.episode_count = 0
        self.action = 0
        self.done = False
        self.make_choice_flag = False

        self.vehicle = None
        self.route = []

    def reset(self):
        """
        reset _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.sumo.simulationStep()
        # self.sumo
        self.reward = 0

        self.done = False
        self.steps = 0
        self.agent_step = 0
        # Error here in self.out_dict, two vars with same name causing an issue
        self.vehicle = Vehicle("1", self.out_dict, self.index_dict, self.sumo)
        self.vehicle.random_relocate()
        self.sumo.simulationStep()

        self.vedge = self.sumo.vehicle.getRoadID("1")
        self.route.append(self.vedge)

        self.vloc = self.vehicle.location()

        self.person = Person("p_0", self.sumo)
        self.pedge = self.sumo.person.getRoadID("p_0")

        self.choices = self.vehicle.get_out_dict()

        self.ploc = self.person.location()

        self.new_distance = math.dist(self.vloc, self.ploc)

        self.vehicle_lane_index = self.index_dict[self.vedge]
        self.person_lane_index = self.index_dict[self.pedge]

        state = np.array([])
        # state = np.append(state, self.vehicle_lane_index)
        # state = np.append(state, self.person_lane_index)
        state = np.append(state, self.vloc)
        state = np.append(state, self.ploc)

        state = np.append(state, self.steps)
        state = np.append(state, self.new_distance)

        self.done = False
        self.make_choice_flag = True

        self.lane = self.sumo.vehicle.getLaneID("1")
        self.choices = self.vehicle.get_out_dict()
        #
        #
        #
        #
        # self.best_route=self.sumo.simulation.findRoute(self.vedge,self.pedge)

        self.old_edge = self.vedge
        return state, self.reward, self.done, self.choices

    def nullstep(self):
        """
        nullstep _summary_

        _extended_summary_
        """
        self.steps += 1
        # self.reward = 0
        self.sumo.simulationStep()

        if ":" in self.vedge or self.old_edge == self.vedge:
            self.make_choice_flag = False
        else:
            self.make_choice_flag = True

        self.old_edge = self.vedge
        self.vedge = self.sumo.vehicle.getRoadID("1")
        pass

    def step(self, action=None):
        """
        step _summary_

        _extended_summary_

        Keyword Arguments:
            action -- _description_ (default: {None})

        Returns:
            _description_
        """
        # self.reward = 0
        self.old_distance = self.new_distance
        self.vloc = self.vehicle.location()
        self.ploc = self.person.location()
        self.choices = self.vehicle.get_out_dict()
        self.new_distance = math.dist(self.vloc, self.ploc)

        if self.make_choice_flag:
            self.vehicle.set_destination(action)
            self.reward += -0.01
            self.agent_step += 1
            self.make_choice_flag = False

        self.steps += 1

        self.action = action

        self.sumo.simulationStep()

        self.lane = self.sumo.vehicle.getLaneID("1")

        self.old_edge = self.vedge
        self.vedge = self.sumo.vehicle.getRoadID("1")
        self.pedge = self.sumo.person.getRoadID("p_0")
        self.route.append(self.vedge)

        if ":" in self.vedge:
            self.vehicle_lane_index = self.vehicle_lane_index
        else:
            self.vehicle_lane_index = self.index_dict[self.vedge]

        self.person_lane_index = self.index_dict[self.pedge]

        self.done = False

        if self.vedge == self.pedge:
            # self.reward += 20
            self.done = True

        if self.steps > self.steps_per_episode:
            # self.reward += -10
            self.done = True

        state = np.array([])
        # state = np.append(state, self.vehicle_lane_index)
        # state = np.append(state, self.person_lane_index)
        state = np.append(state, self.vloc)
        state = np.append(state, self.ploc)
        state = np.append(state, self.agent_step)
        state = np.append(state, self.new_distance)

        reward = np.array([])
        reward = np.append(reward, self.reward)

        return state, reward, self.done, self.choices

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

        return

    def close(self):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo.close()
        # print(len(self.best_route))
        # print(self.best_route)
        # print(len(self.route))
        # print(self.route)
        return
