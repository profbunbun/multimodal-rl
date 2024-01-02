from .vehicle import Vehicle

class VehicleManager:
    """
    VehicleManager class for managing and creating multiple Vehicle instances.

    :param int num_of_vehicles: The number of vehicles to manage.
    :param dict edge_position: Dictionary of edge positions.
    :param sumo: SUMO simulation instance.
    :param dict out_dict: Dictionary of outgoing edges.
    :param dict index_dict: Dictionary of edge indices.
    """
    def __init__(self, num_of_vehicles, edge_position, sumo, out_dict, index_dict):
        """
        Initialize the VehicleManager with the given parameters.
        """
        self.num_of_vehicles = num_of_vehicles
        self.edge_position = edge_position
        self.sumo = sumo
        self.out_dict = out_dict
        self.index_dict = index_dict

    def create_vehicles(self):
        """
        Create and return a list of Vehicle instances.

        :return: List of Vehicle instances.
        :rtype: list
        """
        vehicles = []
        for v_id in range(self.num_of_vehicles):
            vehicles.append(
                Vehicle(
                    str(v_id),
                    self.out_dict,
                    self.index_dict,
                    self.edge_position,
                    self.sumo,
                    v_id + 1,
                )
            )
        return vehicles
