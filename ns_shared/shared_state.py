import logging
import threading
import time

from ns_shared.config import DEFAULT_TIMELINE_CONFIG
from ns_shared.terms import PeripheralConnectionCommand, PeripheralStatus, PhaseState

logger = logging.Logger(__name__)


class SharedState:
    def __init__(self):
        # Always create a seperate lock for each variable to prevent chaos.
        self.sb_camera_frame_lock = threading.Lock()
        self.sb_camera_frame = None

        self.sb_gui_camera_frame_lock = threading.Lock()
        self.sb_gui_camera_frame = None

        self.eng_camera_frame_lock = threading.Lock()
        self.eng_camera_frame = None

        self.eng_gui_camera_frame_lock = threading.Lock()
        self.eng_gui_camera_frame = None

        self.raw_webcam_camera_frame_lock = threading.Lock()
        self.raw_webcam_camera_frame = None

        self.webcam_camera_frame_lock = threading.Lock()
        self.webcam_camera_frame = None

        # self.phase_list_lock = threading.Lock()
        # self.phase_list = list(DEFAULT_TIMELINE_CONFIG)

        # self.current_phase_lock = threading.Lock()
        # self.current_phase = None

        self.phase_state = PhaseState(DEFAULT_TIMELINE_CONFIG)

        self.drive_command_lock = threading.Lock()
        # Mecanum/Omni vectors: x (strafe), y (forward), r (rotation)
        self.drive_x = 0.0
        self.drive_y = 0.0
        self.drive_r = 0.0

        # Button triggers for discrete actions (like opening a mechanism or emergency stopping)
        self.controller_buttons = {
            "cross": False,
            "circle": False,
            "square": False,
            "triangle": False,
            "d_up": False,
            "d_down": False,
            "d_left": False,
            "d_right": False,
        }

        # gui <-> backend peripheral command stuff
        self.peripheral_sbbot_status = PeripheralStatus.DISCONNECTED
        self.peripheral_sbbot_status_lock = threading.Lock()

        self.peripheral_engbot_status = PeripheralStatus.DISCONNECTED
        self.peripheral_engbot_status_lock = threading.Lock()

        self.peripheral_controller_status = PeripheralStatus.DISCONNECTED
        self.peripheral_controller_status_lock = threading.Lock()
