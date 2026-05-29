from ugot import ugot
import ns_shared


class RobotController:
    def __init__(self, sbbot: ugot.UGOT, engbot: ugot.UGOT, QueueChannels, SharedState):
        self.sbbot = sbbot
        self.engbot = engbot
        self.queue_channels = QueueChannels
        self.sharedState = SharedState

    def connect_to_ugot(self):
        # use sbbot instance to scan
        devices = self.sbbot.scan_device()
        for key,value in devices.items():
            match key:
                case ns_shared.SBBOT_NAME:
                    self.sbbot.initialize(value)
                case ns_shared.ENGBOT_NAME:
                    self.engbot.initialize(value)


    def mainloop(self):
        """
        The main purpose of this slave mainloop is to listen to the process_manager mainloop,
        be a good boy, and just be easily killable by process_manager."""
        while not self.queue_channels.kill_flag.is_set():
            pass
        self.sbbot.balance_stop_balancing()
        self.engbot.mecanum_stop()
        # now it shld die

    def phase1(self):
        pass

    def phase2(self):
        pass

    def phase3(self):
        pass

    def phase4(self):
        pass
