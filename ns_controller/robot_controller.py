from ugot import ugot

import ns_shared
import ns_robot


class RobotController:
    def __init__(self, sbbot: ns_robot.RobotHardware, engbot: ns_robot.RobotHardware, QueueChannels, SharedState):
        self.sbbot = sbbot
        self.engbot = engbot
        self.queue_channels = QueueChannels
        self.sharedState = SharedState

        self.connect_to_ugot()

    def connect_to_ugot(self):
        # use sbbot instance to scan
        devices = self.sbbot._sdk.scan_device()
        for key, value in devices.items():
            match key:
                case ns_shared.SBBOT_NAME:
                    self.sbbot._sdk.initialize(value)
                case ns_shared.ENGBOT_NAME:
                    self.engbot._sdk.initialize(value)

    def mainloop(self):
        """
        The main purpose of this slave mainloop is to listen to the process_manager mainloop,
        be a good boy, and just be easily killable by process_manager."""
        while not self.queue_channels.kill_flag.is_set():
            pass
        self.sbbot._sdk.balance_stop_balancing()
        self.engbot._sdk.mecanum_stop()
        # now it shld die

    def phase1(self):
        pass

    def phase2(self):
        pass

    def phase3(self):
        pass

    def phase4(self):
        pass
