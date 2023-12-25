class Utils:
    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        """
        Calculate the Manhattan distance between two points.

        Parameters:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

        Returns:
        The Manhattan distance between the two points.
        """
        return abs(x1 - x2) + abs(y1 - y2)

    # Add any additional static utility methods below
