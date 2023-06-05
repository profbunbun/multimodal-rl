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

import random
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
        return int(round((rightMin + (valueScaled * rightSpan)),0))

class Vehicle:
    CONNECTION_LABEL = 0 
    def __init__(self,vehicle_id,net_file,route_file,out_dict,index_dict) -> None:
        
        self.vehicle_id =vehicle_id
        self.destination = None
        self._net = net_file
        self._route = route_file
        self.min=0
        self.max=0
        self.diff=0
        self.out_dict=out_dict
        self.index_dict=index_dict
        self.choice=None
        self.flag=None
        self.choice_dic=None
        self.make_choice=True
        
        
        
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
        self.sumo = traci.getConnection(self.label)     
        self.min,self.max,self.diff=getMinMax(self._net)
        new_lane=random.choice(list(self.index_dict.keys()))
        print("Start From: " + new_lane)
        
        self.sumo.vehicle.changeTarget("1",edgeID=new_lane)
        self.sumo.vehicle.moveTo("1",new_lane+"_0",5)
        
        
        
        pass
    
    def location(self):
        self.sumo = traci.getConnection(self.label) 
        self.ppos=self.sumo.vehicle.getPosition(self.vehicle_id)
        self.ppos=translate(self.ppos[0],self.min,self.max,0,100),translate(self.ppos[1],self.min,self.max,0,100)
        
        print(self.ppos)
        
        return self.ppos
        
    def set_destination(self,action=None):
        self.sumo = traci.getConnection(self.label) 
        if action==0:
            action=LEFT
        elif action == 1:
            action =STRAIGHT
        elif action == 2:
            action==RIGHT
            
        if action != None:
            self.current_lane=self.sumo.vehicle.getLaneID("1")
            self.cur_loc=self.current_lane.partition("_")[0]
            if self.cur_loc in self.out_dict:
                self.choice_dic=self.out_dict[self.cur_loc]
                self.flag=self.choice_dic.get(action)
            if (self.choice == RIGHT)&(RIGHT in self.choice_dic) :
                    self.flag=self.choice_dic.get(self.choice)
                    self.make_choice=False
                    self.sumo.vehicle.changeTarget("1",edgeID=self.flag)
                    
                    
            elif (self.choice == STRAIGHT) &(STRAIGHT in self.choice_dic):
                    self.flag=self.choice_dic.get(self.choice)
                    self.make_choice=False
                    self.sumo.vehicle.changeTarget("1",edgeID=self.flag)
                    
            elif (self.choice == LEFT)&(LEFT in self.choice_dic):
                    self.flag=self.choice_dic.get(self.choice)
                    self.make_choice=False
                    self.sumo.vehicle.changeTarget("1",edgeID= self.flag)
        
        
        
        
        
        # fleet=self.sumo.vehicle.getTaxiFleet(0)
        # self.sumo.vehicle.changeTarget("1",edgeID="1e")
        # print(fleet)
        pass
    
    def links(self):
        self.sumo = traci.getConnection(self.label) 
        current_lane=self.sumo.vehicle.getLaneID("1")
        choices=len(self.sumo.lane.getLinks(current_lane))
        
        return choices
    
    def set_pickup(self):
        pass
    
    def pickup(self):
        reservation=self.sumo.person.getTaxiReservations(0)
        reservation_id=reservation[0]
        # self.sumo.vehicle.dispatchTaxi("1","0")
        print(reservation_id)
        pass
    
    
    