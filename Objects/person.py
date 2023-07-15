import os
import sys
from Util.utility import Utility
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
from sumolib import net
import traci
util=Utility()
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

    
class Person:
    CONNECTION_LABEL = 0 
    def __init__(self,person_id,net_file,route_file,sumo) -> None:
        
        self.person_id =person_id
        self.destination = None
        self._net = net_file
        self._route = route_file
        self.min=0
        self.max=0
        self.diff=0
        
        self.sumo = sumo
      
        
        self.min,self.max,self.diff=util.getMinMax(self._net)
       
        pass
    
    
    
    def location(self):
        
        self.ppos=self.sumo.person.getPosition(self.person_id)
        
        return self.ppos
        
    def set_destination(self):
        pass
    
    def set_pickup(self):
        pass
    
    def pickup(self):
        pass
    def close(self):
        self.sumo.close()
        return
    
    