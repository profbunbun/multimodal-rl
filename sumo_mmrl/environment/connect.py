"""import stuff"""
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
    """_summary_

    Returns:
        _type_: _description_
    """

    CONNECTION_LABEL = 0

    def __init__(self, sumocfg: str) -> None:
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
        """
        connect_libsumo_no_gui _summary_

        _extended_summary_

        Returns:
            _description_
        """
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
        """
        connect_no_gui _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci
        # self.sumo_.addStepListener(self.listener)
        return self.sumo_

    def close(self):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo_.close()
    
    def busstopCheck(self):
        '''
        busstopCheck _summary_

        _extended_summary_
        '''        
        # getting all bus stops on the map\
        lanes=[]
        stops = self.sumo_.busstop.getIDList()
        for stop in stops:
            lanes.append(self.sumo_.busstop.getLaneID(stop))
        return lanes
    
    