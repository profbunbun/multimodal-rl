import os
import sys
from core.util import getMinMax
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
from sumolib import net
import traci
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
        return int(round((rightMin + (valueScaled * rightSpan)),0))
    
class Person:
    CONNECTION_LABEL = 0 
    def __init__(self,person_id,net_file,route_file) -> None:
        
        self.person_id =person_id
        self.destination = None
        self._net = net_file
        self._route = route_file
        self.min=0
        self.max=0
        self.diff=0
        
        
        self.label = str(Person.CONNECTION_LABEL)
        # Person.CONNECTION_LABEL += 1
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
           
      
   
            
        self.min,self.max,self.diff=getMinMax(self._net)
        conn.close()
        
        
        pass
    
    
    
    def location(self):
        self.sumo = traci.getConnection(self.label) 
        self.ppos=self.sumo.person.getPosition(self.person_id)
        self.ppos=translate(self.ppos[0],self.min,self.max,0,100),translate(self.ppos[1],self.min,self.max,0,100)
        print(self.ppos)
        
        return self.ppos
        
    def set_destination(self):
        
  
        
        pass
    def set_pickup(self):
        pass
    def pickup(self):
   
        pass
    
    