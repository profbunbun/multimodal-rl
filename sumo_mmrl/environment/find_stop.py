"""
     _summary_

    _extended_summary_
"""


class StopFinder:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self) -> None:
        self.con = None

    def manhat_dist(self, x1, y1, x2, y2):
        """
        manhat_dist _summary_

        _extended_summary_

        Args:
            x1 (_type_): _description_
            y1 (_type_): _description_
            x2 (_type_): _description_
            y2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return abs(x1 - x2) + abs(y1 - y2)

    def find_bus_locs(self):
        """
        find_bus_locs _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """

        bus_stops = self.con.busstop.getIDList()
        bus_locs = []
        for stop in bus_stops:
            # print(stop)
            stop_loc = [stop, self.con.busstop.getLaneID(stop)]
            bus_locs.append(stop_loc)
        return bus_locs

    def get_stop_dists(self, loc, loc_dic):
        """
        get_stop_dists _summary_

        _extended_summary_

        Args:
            loc (_type_): _description_
            loc_dic (_type_): _description_
            con (_type_): _description_

        Returns:
            _type_: _description_
        """

        stops = self.find_bus_locs()
        dist_dic = {}

        for stop in stops:
            stop_lane = stop[1].partition("_")[0]
            stop_loc = loc_dic[stop_lane]
            dest_loc = loc_dic[loc]
            dist_dic[stop[0]] = self.manhat_dist(
                dest_loc[0], dest_loc[1], stop_loc[0], stop_loc[1]
            )

        return dist_dic

    def find_end_stop(self, end_loc, loc_dic, con):
        """
        find_end_stop

        _extended_summary_

        Args:
            end_loc (_type_): _description_
        """
        self.con = con

        dic = self.get_stop_dists(end_loc, loc_dic)
        return max(dic, key=dic.get)

    def find_begin_stop(self, begin_loc, loc_dic, con):
        """
        find_begin_stop _summary_

        _extended_summary_

        Args:
            begin_loc (_type_): _description_
            loc_dic (_type_): _description_
            con (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.con = con

        dic = self.get_stop_dists(begin_loc, loc_dic)
        return max(dic, key=dic.get)

    def get_line(self, stop_id):
        """
        get_line _summary_

        _extended_summary_

        Args:
            stop_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        # traci.busstop.getParameterWithKey()
        return self.con.busstop.getParameter(
            stop_id, "lines"
        )  # getParameterWithKey(stop_id,"busStop")

    def get_line_route(self, con):
        """
        get_line_route _summary_

        _extended_summary_

        Args:
            con (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.con = con
        # traci.route.getEdges()?
        return self.con.route.getEdges("bus_1")
