""" Observation Class"""
from .outmask import OutMask


class Observation:
    '''
    Observation class to get and clean the current state's observation.

    :ivar OutMask out_mask: Instance of OutMask class for masking output.
    '''
    def __init__(self):
        '''
        Initialize the Observation object.
        '''
        self.out_mask = OutMask()

    
    def get_state(self, sumo, step, vehicle, destination_loc, life, distcheck):
        '''
        Get observation of the current state and clean it.

        :param sumo: SUMO simulation instance.
        :param int step: The current simulation step.
        :param vehicle: The vehicle object for which to get the state.
        :param tuple destination_loc: The destination location (x, y).
        :param int life: Life of the vehicle.
        :param bool distcheck: Flag to check the distance.
        :return: Normalized state observation as a list.
        :rtype: list
        '''
        
        bounding_box = sumo.simulation.getNetBoundary()
        self.min_x, self.min_y = bounding_box[0]
        self.max_x, self.max_y = bounding_box[1]

        # pad max and min with 1 to allow for vehicles to be on the edge
        self.max_manhat_dist = ((self.max_x + 1) - (self.min_x - 1)) + ((self.max_y + 1) - (self.min_y - 1))
        
        vloc = vehicle.location()
        choices = vehicle.get_out_dict()
        state = []
        
        state.append(self.normalize(vloc[0], self.min_x, self.max_x))
        state.append(self.normalize(vloc[1], self.min_y, self.max_y))
        state.append(self.normalize(destination_loc[0], self.min_x, self.max_x))
        state.append(self.normalize(destination_loc[1], self.min_y, self.max_y))
        state.append(step * 0.0001)
        state.append(self.manhat_dist(vloc[0], vloc[1], destination_loc[0], destination_loc[1]))  
        state.extend(self.out_mask.get_outmask_valid(choices))
        state.append(life)
        state.append(distcheck)

        return state

    def manhat_dist(self, x1, y1, x2, y2):
        '''
        Calculate and normalize the Manhattan distance between two points.

        :param float x1: X-coordinate of the first point.
        :param float y1: Y-coordinate of the first point.
        :param float x2: X-coordinate of the second point.
        :param float y2: Y-coordinate of the second point.
        :return: Normalized Manhattan distance.
        :rtype: float
        '''
        distance = abs(x1 - x2) + abs(y1 - y2)
        # Normalize using the maximum Manhattan distance
        return distance / self.max_manhat_dist
    
    def normalize(self, value, min_value, max_value):
        '''
        Normalize a value to a [0,1] range.

        :param float value: The value to normalize.
        :param float min_value: The minimum value of the range.
        :param float max_value: The maximum value of the range.
        :return: Normalized value.
        :rtype: float
        '''
        # Ensure the denominator isn't zero
        range = max_value - min_value if max_value - min_value != 0 else 1
        return (value - min_value) / range

