import sys
import os

import numpy as np
import gymnasium as gym
from obj.vehicle import Vehicle
from obj.person import Person
from core.util import getMinMax



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
        
    
      self.__current_target_xml_file__ = ""
      self.done=False
      self._net = net_file
      self._route = route_file
      self.use_gui =use_gui
      self.speed=None
      self.render_mode=None
      self.episode_count=0
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
        
        self.episode_count+=1
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
        self.vehicle=Vehicle("1",self._net,self._route)
        self.person=Person("p_0",self._net,self._route)
        vloc=self.vehicle.location()
        ploc=self.person.location()
        self.vehicle.set_destination()
        self.vehicle.pickup()
        state=np.array([vloc,ploc])
        return state
        
        
        
        
        
        
    def step(self, action=None):
        
       
        self.sumo.simulationStep()
        
        
        self.done=True
        self.use_gui=False
        vloc=self.vehicle.location()
        ploc=self.person.location()
        # self.vehicle.set_destination()
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


