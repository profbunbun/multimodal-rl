"""module stuff"""
from Connector.utility import Utility

util = Utility()


class Person:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, person_id, sumo) -> None:
        self.person_id = person_id
        self.sumo = sumo
        pass

    def location(self):
        """
        location _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.ppos = self.sumo.person.getPosition(self.person_id)

        return self.ppos

    def set_destination(self):
        """
        set_destination _summary_

        _extended_summary_
        """
        pass

    def set_pickup(self):
        """
        set_pickup _summary_

        _extended_summary_
        """
        pass

    def pickup(self):
        """
        pickup _summary_

        _extended_summary_
        """
        pass

    def close(self):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo.close()
        return
