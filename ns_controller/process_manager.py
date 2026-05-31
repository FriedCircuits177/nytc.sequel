import threading

from torch._C import parse_schema

import ns_controller
import ns_perception
import ns_robot


class ProcessManager:
    def __init__(self, SBBot: ns_robot.RobotHardware, ENGBot: ns_robot.RobotHardware, QueueChannels, SharedState):
        self.SBBot = SBBot
        self.ENGBot = ENGBot
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.robot_controller = ns_controller.RobotController(SBBot, ENGBot, QueueChannels, SharedState)
        self.sbbot_camera = ns_perception.Camera(self.SBBot, QueueChannels, SharedState)
        self.engbot_camera = ns_perception.Camera(
            self.ENGBot, QueueChannels, SharedState
        )

        self.threads = [
            threading.Thread(target=self.robot_controller.mainloop),
            threading.Thread(target=self.sbbot_camera.mainloop),
            threading.Thread(target=self.engbot_camera.mainloop),
        ]

        for _ in self.threads:
            _.start()

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            pass
        # now do cleanup
        for _ in self.threads:
            _.join()
        # and now he dies peacefully with all his children
