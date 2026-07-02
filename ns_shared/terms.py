import logging
import threading
from enum import Enum

logger = logging.Logger(__name__)


class PhaseType(Enum):
    """Class defining the two PhaseTypes (primarily for GUI usage.)"""

    AUTONOMOUS = "Autonomous"
    MANUAL = "Manual"
    POSE = "Manual (Pose)"


class Phase(Enum):
    """Class defining primary states/phases"""

    Phase1 = ("Phase 1", PhaseType.AUTONOMOUS)
    Phase2 = ("Phase 2", PhaseType.AUTONOMOUS)
    Phase2A = ("Phase 2A", PhaseType.MANUAL)
    Phase3 = ("Phase 3", PhaseType.POSE)
    Phase4 = ("Phase 4", PhaseType.AUTONOMOUS)
    Phase4A = ("Phase 4A", PhaseType.MANUAL)


class MicroState(Enum):
    """this was gonna be used to flag the microstates of the shovelling red blue blocks
    task, not sure if I'll use it, but it's here"""

    pass


class RobotModel(Enum):
    """Class storing the two robot models"""

    SBBot = "SBBot"
    ENGBot = "ENGBot"


class BlockColour(Enum):
    """Class storing upper and lower bounds of block colours (Phase 2,4)"""

    RED = ((0, 120, 70), (10, 255, 255))
    BLUE = ((100, 150, 50), (140, 255, 255))  # Example blue bounds


class PhaseState:
    def __init__(self, timeline_config: list):
        self.lock = threading.Lock()
        if not timeline_config:
            self.phase_queue = []
        else:
            self.phase_queue = timeline_config
        self.current_phase_index = None
        self.is_running = threading.Event()


class PeripheralStatus(Enum):
    CONNECTED = "Connected"
    CONNECTING = "Connecting..."
    DISCONNECTED = "Disconnected"


class PeripheralConnectionCommand(Enum):
    CONNECT = "Connect"
    DISCONNECT = "DISCONNECT"
