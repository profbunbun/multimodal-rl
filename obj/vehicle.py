import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
from sumolib import net
import traci
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class Vehicle:
    CONNECTION_LABEL = 0 
    def __init__(self,vehicle_id,net_file,route_file) -> None:
        
        self.vehicle_id =vehicle_id
        self.destination = None
        self._net = net_file
        self._route = route_file
        
        
        self.label = str(Vehicle.CONNECTION_LABEL)
        Vehicle.CONNECTION_LABEL += 1
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
        
        
        pass
    
    def location(self):
        self.sumo = traci.getConnection(self.label) 
        self.ppos=self.sumo.vehicle.getPosition(self.vehicle_id)
        
        print(self.ppos)
        
        pass
        
    def set_destination(self):
        
        fleet=self.sumo.vehicle.getTaxiFleet(0)
       
        self.sumo.vehicle.changeTarget("1",edgeID="1e")
        print(fleet)
        
        pass
    def set_pickup(self):
        pass
    def pickup(self):
        reservation=self.sumo.person.getTaxiReservations(0)
        reservation_id=reservation[0]
        self.sumo.vehicle.dispatchTaxi("1","0")
        print(reservation_id)
        pass
    
    
    def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
        return int(round((rightMin + (valueScaled * rightSpan)),0))