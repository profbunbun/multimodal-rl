
from Connector.utility import Utility

util=Utility()


    
class Person:
    
    def __init__(self,person_id,sumo) -> None:
        self.person_id =person_id
        self.sumo = sumo
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
    
    