class StageManager:
    def __init__(self, finder, edge_position, sumo, bussroute):
        """
        StageManager class for managing the stages of a journey.

        :param finder: Finder object to search for stops.
        :param dict edge_position: Dictionary of edge positions.
        :param sumo: SUMO simulation instance.
        :param list bussroute: List of bus routes.
        """

        self.finder = finder
        self.edge_position = edge_position
        self.sumo = sumo
        self.bussroute = bussroute
        self.stage = "pickup"

    def update_stage(self, current_stage, destination_edge, vedge, person, vehicle):
        """
        Update the stage of a journey based on the current position and destination.

        :param str current_stage: Current stage of the journey.
        :param str destination_edge: Destination edge ID.
        :param str vedge: Current vehicle edge ID.
        :param Person person: Person object involved in the journey.
        :return: Tuple of the new stage and new destination edge.
        :rtype: tuple
        """
        new_stage = current_stage
        
        new_destination_edge = destination_edge

        if vedge == destination_edge:
            # If the vehicle is at the destination edge, update the stage
            if current_stage == "pickup":
                new_stage = "picked up"
                
                new_destination_edge = self.finder.find_begin_stop(
                    person.get_road(), self.edge_position, self.sumo
                ).partition("_")[0]
                print(new_stage)

            elif current_stage == "picked up":
                new_stage = "onbus"

                end_stop = self.finder.find_end_stop(
                    person.destination, self.edge_position, self.sumo
                ).partition("_")[0]
                new_destination_edge = end_stop
                vehicle.teleport(new_destination_edge)
                self.sumo.simulationStep()

                print(new_stage)

            elif current_stage == "onbus":
                new_stage = "final"
                new_destination_edge = person.destination
                print(new_stage)

            elif current_stage == "final":
                new_stage = "done"
                print(new_stage)
        return new_stage, new_destination_edge

    def get_initial_stage(self):
        """
        Get the initial stage of the journey.

        :return: Initial stage.
        :rtype: str
        """
        return self.stage
