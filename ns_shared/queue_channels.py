import logging
import queue
import threading

logger = logging.Logger(__name__)


class QueueChannels:
    def __init__(self):
        self.kill_flag = threading.Event()
        self.vibrate_flag = threading.Event()
        self.force_stop_phase_flag = threading.Event()

        self.turbo_drive_flag = threading.Event()

        self.pose_recog_active_flag = threading.Event()
        self.sbbot_camera_active_flag = threading.Event()
        self.engbot_camera_active_flag = threading.Event()

        self.block_detection_active_flag = threading.Event()

        # self.gui_start_flag = threading.Event()
        # self.gui_stop_flag = threading.Event()
        # self.gui_left_flag = threading.Event()
        # self.gui_right_flag = threading.Event()

        self.block_detection_data = queue.Queue(1)

        # these 3 handle disconnect/connect commands
        # self.peripheral_sbbot_command_queue = queue.Queue(1)
        # self.peripheral_engbot_command_queue = queue.Queue(1)
        # self.peripheral_controller_command_queue = queue.Queue(1)
