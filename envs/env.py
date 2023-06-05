import sys
import os

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



class Basic(gym.Env):
    metadata = {'render.modes': ['human']}
    CONNECTION_LABEL = 0 
    def __init__(self, net_file: str,
        route_file: str,
        use_gui: bool =False,
        ) -> None:
        
    
      self.length_dict = None
      self.out_dict = None
      self.index_dict = None
      self.edge_list = None
      self.network = network_map_data_structures.getNetInfo(net_file)
      [self.length_dict, self.out_dict, self.index_dict, self.edge_list] = network_map_data_structures.getEdgesInfo(self.network)
      self.__current_target_xml_file__ = ""
      self.done=False
      self._net = net_file
      self._route = route_file
      self.use_gui =use_gui
      self.speed=None
      self.render_mode=None
      self.episode_count=0
      self.episode_step = 0
      self.vehicle=None
      self.min,self.max,self.diff=getMinMax(self._net)
      low=np.array([self.min,self.min])
      high=np.array([self.max,self.max])
      self.observation_space=gym.spaces.Box(low,high, dtype=np.float32)
      self.action_space=gym.spaces.Discrete(3)
      
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
        
        self.episode_step += 1
        self.done=False
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        speed=self.speed
    
        sumo_cmd = [
            self._sumo_binary,
            "-d "+str(speed),
            "-c",
             "nets/3x3/3x3.sumocfg","--start", "--quit-on-end",]
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        self.sumo.simulationStep()
        self.vehicle=Vehicle("1",self._net,self._route,self.out_dict, self.index_dict)
        self.person=Person("p_0",self._net,self._route)
        self.vloc=self.vehicle.location()
        self.vloc=translate(self.vloc[0],self.min,self.max,0,100),translate(self.vloc[1],self.min,self.max,0,100)
        
        self.ploc=self.person.location()
        self.ploc=translate(self.ploc[0],self.min,self.max,0,100),translate(self.ploc[1],self.min,self.max,0,100)
        
        self.episode_step = 0
        # self.vehicle.set_destination()
        # self.vehicle.pickup()
        state=np.array([self.vloc,self.ploc])
        return state
        
        
        
        
        
        
    def step(self, action=None):
        
        
        
       
        self.sumo.simulationStep()
        
        
        self.done=True
        self.use_gui=False
        vloc=self.vehicle.location()
        ploc=self.person.location()
        self.vehicle.set_destination(action)
        # print(self.sumo.vehicle.getRoute("1"))
        # self.vehicle.pickup()
        state=np.array([vloc,ploc])
        return state
       
    
    
    
    def render(self, mode='human'):
        self.speed = 150
        self.use_gui=True
        # print('render')
        return
    
    def close(self):
        self.sumo.close()
        
        # print('close')
        return


