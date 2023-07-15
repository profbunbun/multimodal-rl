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

import traci


util=Utility()

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ



class Basic():
    metadata = {'render.modes': ['human']}
    CONNECTION_LABEL = 0 
    def __init__(self, net_file: str,
        route_file: str,
        use_gui: bool =False,
        steps_per_episode: int = 5000,
        ) -> None:
        self.steps_per_episode = steps_per_episode
        self.episode_count=0
        self.episode_step = 0
        self.action=0
        
        self.done=False
        self.no_choice=False
        
        
        self.out_dict = None
        self.index_dict = None
        self.edge_list = None
        self.network = util.getNetInfo(net_file)
        [self.out_dict, self.index_dict, self.edge_list] = util.getEdgesInfo(self.network)
        
        self._net = net_file
        self._route = route_file
        
        self.use_gui =use_gui
        self.speed=None
        self.render_mode=None
        
        self.vehicle=None
        
        self.label = str(Basic.CONNECTION_LABEL)
        Basic.CONNECTION_LABEL += 1
        self.sumo = None
        
        if LIBSUMO:
                traci.start(
                    [sumolib.checkBinary("sumo"), "-n", self._net]
                )  
                conn = traci
        else:
                traci.start(
                    [sumolib.checkBinary("sumo"), "-n", self._net],
                    label="init_connection" + self.label,
                )
                conn = traci.getConnection("init_connection" + self.label)
        conn.close()
    
   
      
    
    def reset(self):
        
       
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
            
        speed=self.speed
    
        sumo_cmd = [
            self._sumo_binary,
            "-d "+"5",
            "-c",
             "Nets/3x3.sumocfg","--start", "--quit-on-end","--no-step-log","--no-warnings","--no-duration-log",]
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        self.sumo.simulationStep()
        self.reward = 0
        
        self.episode_step += 1
        self.done=False
        self.steps = 0
        self.agent_step=0
        # print (self.index_dict)
        
        self.vehicle=Vehicle("1",self._net,self._route,self.out_dict, self.index_dict,self.sumo)
        self.person=Person("p_0",self._net,self._route,self.sumo)
        
      
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.pedge=self.sumo.person.getRoadID("p_0")
        
        self.vehicle_lane_index=self.index_dict[self.vedge]
        self.person_lane_index=self.index_dict[self.pedge]
        
        self.vloc=self.vehicle.location()
        self.ploc=self.person.location()
        self.new_distance= (math.dist(self.vloc,self.ploc))
       
        state=np.array([])
        state=np.append(state,self.vehicle_lane_index)
        state=np.append(state,self.person_lane_index)
        state=np.append(state,self.agent_step)
        state=np.append(state,self.new_distance)
       
        
        self.done=False
        self.no_choice=False
       
      
        self.lane=self.sumo.vehicle.getLaneID("1")
        self.out_dict =self.vehicle.get_out_dict()  
        

        
        return state,self.reward, self.no_choice,self.lane, self.out_dict
        
        
        
    def nullstep(self):
        
        self.old_edge=self.vedge
        self.sumo.simulationStep()
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.no_choice=False
        
        
        if self.old_edge == self.vedge:
            self.no_choice=True
        pass    
        
    def step(self, action=None):
        self.reward = 0
        self.old_distance=self.new_distance
      
        
        self.steps+= 1
        
        self.action=action
        self.old_edge=self.vedge
        self.sumo.simulationStep()
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.no_choice=False
       
        if self.old_edge == self.vedge:
            self.no_choice=True
            
        self.lane=self.sumo.vehicle.getLaneID("1")
        self.done=True
        self.use_gui=False
        self.vloc=self.vehicle.location()
        self.ploc=self.person.location()
        self.new_distance= (math.dist(self.vloc,self.ploc))
       
        if self.new_distance >= self.old_distance:
                self.reward+= -.5
        if self.new_distance < self.old_distance:
            
                self.reward+= .6
                
        
        self.vehicle.set_destination(action)
        self.reward+= -.7
        self.agent_step+=1
        
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.pedge=self.sumo.person.getRoadID("p_0")
            
        if ":" not in self.vedge:
                self.vehicle_lane_index=self.index_dict[self.vedge]
        else:
                self.vehicle_lane_index=self.vehicle_lane_index
                
        self.person_lane_index=self.index_dict[self.pedge]
        
            
        done=False
    
       
            
        if self.vedge==self.pedge:
                self.reward+=10
                done=True
        if self.steps>self.steps_per_episode:
                self.reward+=-10
                done=True
                
        
            
        state=np.array([])
        state=np.append(state,self.vehicle_lane_index)
        state=np.append(state,self.person_lane_index)
        state=np.append(state,self.agent_step)
        state=np.append(state,self.new_distance)
      
        reward=np.array([])
        reward=np.append(reward,self.reward)    
        
        return state,reward,done
    
    
    def render(self, mode='human'):
        self.speed = 5
        self.use_gui=True
        return
    
    def close(self):
        # self.vehicle.close()
        # self.person.close()
        self.sumo.close()
        return


