import sys
import os
import libsumo as traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
    

class connection():
    
    def __init__(self) -> None:
        pass