import sys
import os
import math
import torch as T
import numpy as np
import gymnasium as gym
from Objects.vehicle import Vehicle
from Objects.person import Person
from Util.util import getMinMax,translate
from Util import network_map_data_structures






if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
import traci




LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ



class Basic():
    metadata = {'render.modes': ['human']}
    CONNECTION_LABEL = 0 
    def __init__(self, net_file: str,
        route_file: str,
        use_gui: bool =False,
        steps_per_episode: int = 8000,
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
        self.network = network_map_data_structures.getNetInfo(net_file)
        [self.out_dict, self.index_dict, self.edge_list] = network_map_data_structures.getEdgesInfo(self.network)
        
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
        # print (self.index_dict)
        
        self.vehicle=Vehicle("1",self._net,self._route,self.out_dict, self.index_dict)
        self.person=Person("p_0",self._net,self._route)
        
      
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
        state=np.append(state,self.steps)
        state=np.append(state,self.new_distance)
        state = T.from_numpy(state)
        state=state.double()
        
        self.done=False
        self.no_choice=False
       
      
        self.lane=self.sumo.vehicle.getLaneID("1")
        self.out_dict =self.vehicle.get_out_dict()  
        

        
        return state,self.reward, self.no_choice,self.lane, self.out_dict
        
        
        
        
        
    def step(self, action):
        self.old_distance=self.new_distance
        self.reward+=-0.005
        self.steps+= 1
        self.action=action
        self.old_edge=self.vedge
        self.sumo.simulationStep()
        self.vedge=self.sumo.vehicle.getRoadID("1")
        
        if self.old_edge == self.vedge:
            self.no_choice=True
            
        self.lane=self.sumo.vehicle.getLaneID("1")
        self.done=True
        self.use_gui=False
        self.vloc=self.vehicle.location()
        self.ploc=self.person.location()
        self.new_distance= (math.dist(self.vloc,self.ploc))
        
        if self.new_distance > self.old_distance:
                self.reward+=-.05
        if self.new_distance < self.old_distance:
                self.reward+=-.01
                
        if not self.no_choice:
            self.vehicle.set_destination(np.max(action))
        if self.no_choice and action !=-1:
            self.reward+=-.02
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.pedge=self.sumo.person.getRoadID("p_0")
            
        if ":" not in self.vedge:
                self.vehicle_lane_index=self.index_dict[self.vedge]
        else:
                self.vehicle_lane_index=self.vehicle_lane_index
                
        self.person_lane_index=self.index_dict[self.pedge]
        
            
        done=False
    
       
            
        if self.vedge==self.pedge:
                self.reward+=2.5
                done=True
        if self.steps>self.steps_per_episode:
                self.reward+=-2
                done=True
                
        
            
        state=np.array([])
        state=np.append(state,self.vehicle_lane_index)
        state=np.append(state,self.person_lane_index)
        state=np.append(state,self.steps)
        state=np.append(state,self.new_distance)
        state = T.from_numpy(state)
        state=state.double()
        reward=np.array([])
        reward=np.append(reward,self.reward)    
        reward=T.from_numpy(reward)
        reward=reward.double()
        return state,reward,done
    
    
    def render(self, mode='human'):
        self.speed = 5
        self.use_gui=True
        return
    
    def close(self):
        self.sumo.close()
        return


