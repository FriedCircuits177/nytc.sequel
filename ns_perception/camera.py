import base64

import pygame
from turbojpeg import TurboJPEG

from ns_shared import QueueChannels, SharedState
import ns_robot

class Camera:
    def __init__(
        self, robot: ns_robot.RobotHardware, queue_channels: QueueChannels, shared_state: SharedState
    ):
        self.robot = robot
        self.tj = TurboJPEG()
        self.queue_channels = queue_channels
        self.shared_state = shared_state

    def readEncodedFrame(self) -> str:
        """Returns raw base64 encoded JPEG frame from robot camera."""
        input_data = self.robot._sdk.VISION.vision_pb2.ReadCameraDataRequest()
        input_data.extra = ""
        return self.robot._sdk.VISION.client.readCameraData(
            input_data
        )  # reads from low-level unary_unary channel

    def b64_to_surface_turbo(self, b64_string: str) -> pygame.Surface:
        """Converts raw base64 encoded JPEG into pygame surface"""
        jpg_bytes = base64.b64decode(b64_string)
        rgb = self.tj.decode(jpg_bytes)  # returns a numpy array (H, W, 3)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            pass
