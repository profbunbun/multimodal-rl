import sys
import os
import torch as T
import numpy as np
import gymnasium as gym
from obj.vehicle import Vehicle
from obj.person import Person
from core.util import getMinMax,translate
from core import network_map_data_structures






if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
import traci




LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ



class Basic(gym.Env,):
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
        
        
        self.length_dict = None
        self.out_dict = None
        self.index_dict = None
        self.edge_list = None
        self.network = network_map_data_structures.getNetInfo(net_file)
        [self.length_dict, self.out_dict, self.index_dict, self.edge_list] = network_map_data_structures.getEdgesInfo(self.network)
        self.__current_target_xml_file__ = ""
        
        
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
        
        self.reward = 0
        self.episode_step += 1
        self.done=False
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        speed=self.speed
    
        sumo_cmd = [
            self._sumo_binary,
            "-d "+"5",
            "-c",
             "nets/3x3/3x3.sumocfg","--start", "--quit-on-end","--no-step-log","--no-warnings","--no-duration-log",]
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        self.sumo.simulationStep()
        self.steps = 0
        # print (self.index_dict)
        
        self.vehicle=Vehicle("1",self._net,self._route,self.out_dict, self.index_dict)
        self.person=Person("p_0",self._net,self._route)
        
      
        self.vedge=self.sumo.vehicle.getRoadID("1")
        self.pedge=self.sumo.person.getRoadID("p_0")
        
        self.vehicle_lane_index=self.index_dict[self.vedge]
        self.person_lane_index=self.index_dict[self.pedge]
        
        
        state=np.array([])
        state=np.append(state,self.vehicle_lane_index)
        state=np.append(state,self.person_lane_index)
        state=np.append(state,self.action)
        state=np.append(state,self.reward)
        state = T.from_numpy(state)
        
        self.done=False
        self.no_choice=False
       
      
        self.lane=self.sumo.vehicle.getLaneID("1")
        self.out_dict =self.vehicle.get_out_dict()  
        

        
        return state,self.reward, self.no_choice,self.lane, self.out_dict
        
        
        
        
        
    def step(self, action):
        if action != None:
            
            # print(action)
            self.reward+=-0.001
            self.steps+= 1
            oldvedge=self.vedge
            self.action=action
            # print(self.action)
            self.sumo.simulationStep()
            self.vedge=self.sumo.vehicle.getRoadID("1")
            self.lane=self.sumo.vehicle.getLaneID("1")
            
            self.done=True
            self.use_gui=False
            self.vloc=self.vehicle.location()
            
            self.ploc=self.person.location()
            
            self.vehicle.set_destination(action)
            
            
    
            self.vedge=self.sumo.vehicle.getRoadID("1")
            self.pedge=self.sumo.person.getRoadID("p_0")
            
            if ":" not in self.vedge:
                self.vehicle_lane_index=self.index_dict[self.vedge]
            else:
                self.vehicle_lane_index=self.vehicle_lane_index
                
            self.person_lane_index=self.index_dict[self.pedge]
        
            
            done=False
            self.no_choice=False
            info={}
            
            if self.vedge==self.pedge:
                self.reward+=2.5
                done=True
            if self.steps>self.steps_per_episode:
                self.reward+=-1
                done=True
                
            if oldvedge==self.vedge:
                
                self.no_choice=True
                
            if ":" in self.lane:
                self.no_choice=True
            
            state=np.array([])
            state=np.append(state,self.vehicle_lane_index)
            state=np.append(state,self.person_lane_index)
            state=np.append(state,self.action)
            state=np.append(state,self.reward)
            state = T.from_numpy(state)
            reward=np.array([])
            reward=np.append(reward,self.reward)    
            reward=T.from_numpy(reward)
        else:
            # print(action)
            self.reward+=-0.001
            self.steps+= 1
            oldvedge=self.vedge
            self.action=-1
            # print(self.action)
            self.sumo.simulationStep()
            self.vedge=self.sumo.vehicle.getRoadID("1")
            self.lane=self.sumo.vehicle.getLaneID("1")
            
            self.done=True
            self.use_gui=False
            self.vloc=self.vehicle.location()
            
            self.ploc=self.person.location()
            
            # self.vehicle.set_destination(action)
            
            
    
            self.vedge=self.sumo.vehicle.getRoadID("1")
            self.pedge=self.sumo.person.getRoadID("p_0")
            
            if ":" not in self.vedge:
                self.vehicle_lane_index=self.index_dict[self.vedge]
            else:
                self.vehicle_lane_index=self.vehicle_lane_index
                
            self.person_lane_index=self.index_dict[self.pedge]
        
            
            done=False
            self.no_choice=False
            info={}
            
            if self.vedge==self.pedge:
                self.reward+=2.5
                done=True
            if self.steps>self.steps_per_episode:
                self.reward+=-1
                done=True
                
            if oldvedge==self.vedge:
                
                self.no_choice=True
                
            if ":" in self.lane:
                self.no_choice=True
            
            state=np.array([])
            state=np.append(state,self.vehicle_lane_index)
            state=np.append(state,self.person_lane_index)
            state=np.append(state,self.action)
            state=np.append(state,self.reward)
            state = T.from_numpy(state)
            reward=np.array([])
            reward=np.append(reward,self.reward)    
            reward=T.from_numpy(reward)
        
        return state,reward,done,info,self.no_choice,self.lane
    
    
    
    def render(self, mode='human'):
        self.speed = 5
        self.use_gui=True
        # print('render')
        return
    
    def close(self):
        self.sumo.close()
        
        # print('close')
        return


