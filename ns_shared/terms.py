from enum import Enum




class State:
    """Class defining primary states/phases"""

    IDLE = Enum()

    Phase1 = Enum()
    Phase2 = Enum()
    Phase3 = Enum()
    Phase4 = Enum()


class MicroState:
    """this was gonna be used to flag the microstates of the shovelling red blue blocks
    task, not sure if I'll use it, but it's here"""

    pass
