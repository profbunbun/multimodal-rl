
class OutMask:
    """
    OutMask class to generate output masks based on vehicle edge and position data.

    This class provides methods to generate masks for determining the valid movement directions and distance calculations.
    """

    def get_outmask(self, vedge, pedge, choices, edge_position):
        """
        Generate an output mask and edge distance based on current and previous edge locations.

        :param str vedge: Current vehicle edge ID.
        :param str pedge: Previous vehicle edge ID.
        :param dict choices: Dictionary of movement choices from the current edge.
        :param dict edge_position: Dictionary of edge IDs and their corresponding (x, y) positions.
        :return: Tuple of current edge location, previous edge location, output mask, and edge distance.
        :rtype: tuple

        The output mask is a list where -1 indicates invalid movement and 1 indicates valid movement in the order of right, straight, left, and to.
        """

        vedge_loc = edge_position[vedge]
        pedge_loc = edge_position[pedge.partition("_")[0]]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], pedge_loc[0], pedge_loc[1]
        )
        outmask = [-1, -1, -1, -1]

        for key, value in choices.items():
            if key == "r":
                sloc = edge_position[value]
                s_dist = self.manhat_dist(sloc[0], sloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if s_dist < edge_distance:
                    outmask[0] = 1

            elif key == "s":
                tloc = edge_position[value]
                t_dist = self.manhat_dist(tloc[0], tloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if t_dist < edge_distance:
                    outmask[1] = 1

            elif key == "l":
                rloc = edge_position[value]
                r_dist = self.manhat_dist(rloc[0], rloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if r_dist < edge_distance:
                    outmask[2] = 1

            elif key == "t":
                lloc = edge_position[value]
                l_dist = self.manhat_dist(lloc[0], lloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if l_dist < edge_distance:
                    outmask[3] = 1

        return vedge_loc, pedge_loc, outmask, edge_distance

    def get_outmask_valid(self, choices):
        """
        Generate a validity mask for the available movement choices.

        :param dict choices: Dictionary of movement choices from the current edge.
        :return: List representing the output mask for valid movements.
        :rtype: list

        The output mask is a list with values 1 for valid movements and 0 for invalid, in the order of right, straight, left, and to.
        """

        # outmask = [0, 0, 0, 0]
        outmask = [0, 0, 0]

        for choice in choices.items():
            if choice[0] == "r":
                outmask[0] = 1

            elif choice[0] == "s":
                outmask[1] = 1

            elif choice[0] == "l":
                outmask[2] = 1

            # elif choice[0] == "t":
            #     outmask[3] = 1

        return outmask

    def manhat_dist(self, x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        :param float x1: X-coordinate of the first point.
        :param float y1: Y-coordinate of the first point.
        :param float x2: X-coordinate of the second point.
        :param float y2: Y-coordinate of the second point.
        :return: Manhattan distance between the two points.
        :rtype: float
        """
      
        return abs(x1 - x2) + abs(y1 - y2)
