import sys
import os
import math
import torch as T
import numpy as np
import gymnasium as gym
from Objects.vehicle import Vehicle
from Objects.person import Person
from Util.utility import Utility


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
    
import sumolib
import libsumo as traci

util = Utility()

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class Basic():
    metadata = {'render.modes': ['human']}
    CONNECTION_LABEL = 0

    def __init__(self, net_file: str,
                 route_file: str,gui
                 ) -> None:
        self.steps_per_episode = 2000
        self.episode_count = 0
        self.episode_step = 0
        self.action = 0

        self.done = False
        self.no_choice = False

        self.out_dict = None
        self.index_dict = None
        self.edge_list = None
        self.network = util.getNetInfo(net_file)
        [self.out_dict, self.index_dict,
            self.edge_list] = util.getEdgesInfo(self.network)

        self._net = net_file
        self._route = route_file

        self.use_gui = gui
        self.speed = None
        self.render_mode = None

        self.vehicle = None

        self.label = str(Basic.CONNECTION_LABEL)
        Basic.CONNECTION_LABEL += 1
        self.sumo = None

    def reset(self):

        if self.use_gui :
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
            sumo_cmd = [
            "sumo-gui",
            "-d "+"5",
            
            "-c",
            
            "Nets/3x3.sumocfg", "--start", "--quit-on-end", "--no-step-log", "--no-warnings", "--no-duration-log",]

        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
            sumo_cmd = [
            "sumo",
            # "-d "+"5",
            
            "-c",
            "Nets/3x3.sumocfg", "--start", "--quit-on-end", "--no-step-log", "--no-warnings", "--no-duration-log",]


        speed = self.speed

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci

        self.sumo.simulationStep()
        self.reward = 0

        self.episode_step += 1
        self.done = False
        self.steps = 0
        self.agent_step = 0

        self.vehicle = Vehicle("1", self._net, self._route,
                               self.out_dict, self.index_dict, self.sumo)
        self.person = Person("p_0", self._net, self._route, self.sumo)

        self.vedge = self.sumo.vehicle.getRoadID("1")
        self.pedge = self.sumo.person.getRoadID("p_0")

        self.vehicle_lane_index = self.index_dict[self.vedge]
        self.person_lane_index = self.index_dict[self.pedge]

        self.vloc = self.vehicle.location()
        self.ploc = self.person.location()
        self.new_distance = (math.dist(self.vloc, self.ploc))

        state = np.array([])
        state = np.append(state, self.vehicle_lane_index)
        state = np.append(state, self.person_lane_index)
        # state = np.append(state, self.agent_step)
        state = np.append(state, self.new_distance)

        self.done = False
        self.no_choice = False

        self.lane = self.sumo.vehicle.getLaneID("1")
        self.out_dict = self.vehicle.get_out_dict()

        return state, self.reward, self.no_choice, self.lane, self.out_dict

    def nullstep(self):

        self.old_edge = self.vedge
        self.sumo.simulationStep()
        self.vedge = self.sumo.vehicle.getRoadID("1")

        if self.old_edge == self.vedge:
            self.no_choice = True
        else:
            self.no_choice = False  
        pass

    def step(self, action=None):
        self.reward = 0
        self.old_distance = self.new_distance

        self.steps += 1

        self.action = action
        self.old_edge = self.vedge
        self.sumo.simulationStep()
        self.vedge = self.sumo.vehicle.getRoadID("1")

        if self.old_edge == self.vedge and ':' not in self.vedge:
            self.no_choice = True
        else:
            self.no_choice = False

        self.lane = self.sumo.vehicle.getLaneID("1")
        self.done = True
        self.use_gui = False
        self.vloc = self.vehicle.location()
        self.ploc = self.person.location()
        self.new_distance = (math.dist(self.vloc, self.ploc))

        if self.new_distance > self.old_distance:
            self.reward += -.3
        if self.new_distance < self.old_distance:

            self.reward += .2

        self.vehicle.set_destination(action)
        self.reward += -.1
        self.agent_step += 1

        self.vedge = self.sumo.vehicle.getRoadID("1")
        self.pedge = self.sumo.person.getRoadID("p_0")

        if ":" not in self.vedge:
            self.vehicle_lane_index = self.index_dict[self.vedge]
        else:
            self.vehicle_lane_index = self.vehicle_lane_index

        self.person_lane_index = self.index_dict[self.pedge]

        done = False

        if self.vedge == self.pedge:
            self.reward += 10
            done = True
        # if self.steps > self.steps_per_episode:
        #     self.reward += -10
        #     done = True
        #     self.sumo.close()
            

        state = np.array([])
        state = np.append(state, self.vehicle_lane_index)
        state = np.append(state, self.person_lane_index)
        # state = np.append(state, self.agent_step)
        state = np.append(state, self.new_distance)

        reward = np.array([])
        reward = np.append(reward, self.reward)

        return state, reward, done

    def render(self, mode='human'):
        self.speed = 5
        self.use_gui = True
        return

    def close(self):
        # self.vehicle.close()
        # self.person.close()
        self.sumo.close()
        return
