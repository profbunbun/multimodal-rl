""" Observation Class"""
from .outmask import OutMask


class Observation:
    '''get observation of current state and clean it'''
    def __init__(self):
        '''init'''
        self.out_mask = OutMask()

    
    def get_state(self, sumo, step, vehicle, destination_loc, life, distcheck):
        '''get observation of current state and clean it'''
        
        bounding_box = sumo.simulation.getNetBoundary()
        self.min_x, self.min_y = bounding_box[0]
        self.max_x, self.max_y = bounding_box[1]

        self.max_manhat_dist = (self.max_x - self.min_x) + (self.max_y - self.min_y)
        
        vloc = vehicle.location()
        choices = vehicle.get_out_dict()
        state = []
        
        state.append(self.normalize(vloc[0], self.min_x, self.max_x))
        state.append(self.normalize(vloc[1], self.min_y, self.max_y))
        state.append(self.normalize(destination_loc[0], self.min_x, self.max_x))
        state.append(self.normalize(destination_loc[1], self.min_y, self.max_y))
        state.append(step)
        state.append(self.manhat_dist(vloc[0], vloc[1], destination_loc[0], destination_loc[1]))  
        state.extend(self.out_mask.get_outmask_valid(choices))
        state.append(life)
        state.append(distcheck)

        return state

    def manhat_dist(self, x1, y1, x2, y2):
        '''Calculate and normalize Manhattan distance'''
        distance = abs(x1 - x2) + abs(y1 - y2)
        # Normalize using the maximum Manhattan distance
        return distance / self.max_manhat_dist
    
    def normalize(self, value, min_value, max_value):
        """Normalize a value to a [0,1] range."""
        # Ensure the denominator isn't zero
        range = max_value - min_value if max_value - min_value != 0 else 1
        return (value - min_value) / range

