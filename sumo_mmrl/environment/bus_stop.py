

class StopFinder:
    '''
    StopFinder class for finding bus stops and distances in the SUMO environment.

    Attributes:
        con (connection object): Connection object for interacting with SUMO.
    '''


    def __init__(self) -> None:
        '''
        Initializes the StopFinder object.
        '''
        self.con = None

    def manhat_dist(self, x1, y1, x2, y2):

        return abs(x1 - x2) + abs(y1 - y2)

    def find_bus_locs(self):
        '''
        Finds the locations of all bus stops.

        :return: A list of bus stop locations.
        :rtype: list
        '''

        bus_stops = self.con.busstop.getIDList()
        bus_locs = []
        for stop in bus_stops:
            # print(stop)
            stop_loc = [stop, self.con.busstop.getLaneID(stop)]
            bus_locs.append(stop_loc)
        return bus_locs

    def get_stop_dists(self, loc, loc_dic):
        '''
        Gets the distances from a location to all bus stops.

        :param loc: The current location.
        :type loc: str
        :param loc_dic: Dictionary of locations.
        :type loc_dic: dict
        :return: A dictionary of distances to each bus stop.
        :rtype: dict
        '''
     
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
        '''
        Finds the nearest bus stop to the end location.

        :param end_loc: The end location.
        :type end_loc: str
        :param loc_dic: Dictionary of locations.
        :type loc_dic: dict
        :param con: Connection object.
        :type con: connection object
        :return: Lane ID of the nearest bus stop.
        :rtype: str
        '''

        self.con = con

        dic = self.get_stop_dists(end_loc, loc_dic)
        stop = min(dic, key=dic.get)
        lane = con.busstop.getLaneID(stop)
        return lane

    def find_begin_stop(self, begin_loc, loc_dic, con):
        '''
        Finds the nearest bus stop to the beginning location.

        :param begin_loc: The beginning location.
        :type begin_loc: str
        :param loc_dic: Dictionary of locations.
        :type loc_dic: dict
        :param con: Connection object.
        :type con: connection object
        :return: Lane ID of the nearest bus stop.
        :rtype: str
        '''

        self.con = con

        dic = self.get_stop_dists(begin_loc, loc_dic)
        stop = min(dic, key=dic.get)

        lines = self.get_line(stop)
        
        lane = con.busstop.getLaneID(stop)
        return lane

    def get_line(self, stop_id):
        '''
        Retrieves the line parameter of a given bus stop.

        :param stop_id: ID of the bus stop.
        :type stop_id: str
        :return: Line parameter of the bus stop.
        :rtype: str
        '''
  
        # traci.busstop.getParameterWithKey()
        return self.con.busstop.getParameter(
            stop_id, "line"
        )  # getParameterWithKey(stop_id,"busStop")

    def get_line_route(self, con):
        '''
        Retrieves the edges of the route for a specific line.

        :param con: Connection object.
        :type con: connection object
        :return: Edges of the route.
        :rtype: list
        '''
    
        self.con = con
        # traci.route.getEdges()?
        return self.con.route.getEdges("bus_1")

