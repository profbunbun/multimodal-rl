
import sys
import os
import traci
import libsumo



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")


class SUMOConnection:
    '''
    SUMOConnection class for managing connections to the SUMO simulation.

    Attributes:
        label (str): Unique label for the connection.
        sumocfg (str): Path to the SUMO configuration file.
        sumo_ (SUMO object): The SUMO simulation object.
        sumo_cmd (list): Command list to start SUMO.
    '''


    
    CONNECTION_LABEL = 0

    def __init__(self, sumocfg: str) -> None:
        '''
        Initializes the SUMOConnection object.

        :param sumocfg: Path to the SUMO configuration file.
        :type sumocfg: str
        '''
        self.label = str(SUMOConnection.CONNECTION_LABEL)
        SUMOConnection.CONNECTION_LABEL += 1
        self.sumocfg = sumocfg
        self.sumo_ = None
        self.sumo_cmd = None

    def connect_gui(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.sumo_cmd = [
            "sumo-gui",
            "-c",
            self.sumocfg,
            "-d",
            "50",
            "--start",
            "--quit-on-end",
            "--human-readable-time",
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci
        # self.sumo_.addStepListener(self.listener)
        
        return self.sumo_

    def connect_libsumo_no_gui(self):

        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        libsumo.start(self.sumo_cmd, label=self.label)
        self.sumo_ = libsumo
        # self.sumo_.addStepListener(self.listener)
        return self.sumo_

    def connect_no_gui(self):
  
        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci

        return self.sumo_

    def close(self):
 
        self.sumo_.close()
    
    def busstopCheck(self):

        lanes = []
        stops = self.sumo_.busstop.getIDList()
        for stop in stops:
            lanes.append(self.sumo_.busstop.getLaneID(stop))
        return lanes

    def get_junction_list(self):
        return traci.constants.junction.getIDList()

    def get_edge_list(self):
        return traci.constants.edge.getIDList()

    def get_lane_list(self):
        return traci.constants.lane.getIDList()
    
    
    
