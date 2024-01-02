class RideSelect:
    """
    RideSelect class for selecting vehicles based on specific criteria.
    """


    def __init__(self) -> None:
        """
        Initialize the RideSelect instance.
        """

        pass

    def select(self, vehicle_array, person):
        """
        Select a vehicle from an array based on the person's type.

        :param list vehicle_array: Array of vehicle objects.
        :param Person person: Person object for whom the vehicle is being selected.
        :return: Selected vehicle ID.
        :rtype: str
        """
        
        v_dict = self. make_vehic_atribs_dic(vehicle_array)
        p_type = person.get_type()

        vehicle_id = {i for i in v_dict if v_dict[i]==p_type}

        # print(type(vehicle_id))
        return list(vehicle_id)[0]

    def make_vehic_atribs_dic(self, vehicle_array):
        """
        Create a dictionary of vehicle attributes from an array of vehicles.

        :param list vehicle_array: Array of vehicle objects.
        :return: Dictionary of vehicle attributes.
        :rtype: dict
        """
        
        v_dict = { }
        
        for v in vehicle_array:
            v_dict[v.vehicle_id] = v.get_type()

        # return v_a_dic
        return v_dict